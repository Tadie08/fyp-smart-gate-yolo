import cv2
import numpy as np
import onnxruntime as ort
import pytesseract
import requests
import base64
import re
import time
from picamera2 import Picamera2
from behaviour import init_db, gate_decision

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
MODEL_PATH = "models/best_nano.onnx"
EDGE_CONF_THRESHOLD = 0.75    # Above this = use Tesseract locally
CLOUD_CONF_THRESHOLD = 0.30   # Between 0.30-0.75 = use Plate Recognizer
IMG_SIZE = 640

PLATE_RECOGNIZER_TOKEN = "e8628359fce05e8b1ef7f79a7de7d9e76dbe49ea"

WHITELIST = ["ABC123", "ML773", "WXY1234", "YZ3527"]

# Counters for cloud reduction metric
stats = {
    "total": 0,
    "cache_hits": 0,
    "edge_ocr": 0,
    "cloud_ocr": 0,
    "no_detection": 0
}

# ─────────────────────────────
# LOAD YOLO MODEL
# ─────────────────────────────
print("Loading YOLO model...")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
session = ort.InferenceSession(MODEL_PATH, sess_options=opts)
input_name = session.get_inputs()[0].name
print("YOLO model loaded!")

# ─────────────────────────────
# INIT DATABASE
# ─────────────────────────────
init_db()

# Simple in-memory cache (Redis replacement for now)
plate_cache = {}
CACHE_TTL = 300  # 5 minutes

# ─────────────────────────────
# INIT CAMERA
# ─────────────────────────────
print("Starting camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)
picam2.set_controls({"AwbMode": 4})
time.sleep(1)
print("Camera ready!")

# ─────────────────────────────
# PREPROCESS
# ─────────────────────────────
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img /= 255.0
    return img

# ─────────────────────────────
# POSTPROCESS
# ─────────────────────────────
def postprocess(outputs, conf_threshold=0.15):
    predictions = outputs[0][0]
    detections = []
    for pred in predictions:
        confidence = pred[4]
        if confidence >= conf_threshold:
            x_center, y_center, w, h = pred[0], pred[1], pred[2], pred[3]
            detections.append({
                "bbox": (x_center, y_center, w, h),
                "confidence": float(confidence)
            })
    return detections

# ─────────────────────────────
# CROP PLATE REGION
# ─────────────────────────────
def crop_plate(frame, bbox):
    h, w = frame.shape[:2]
    x_center, y_center, bw, bh = bbox
    x1 = max(0, int((x_center - bw/2) * w / 640))
    y1 = max(0, int((y_center - bh/2) * h / 480))
    x2 = min(w, int((x_center + bw/2) * w / 640))
    y2 = min(h, int((y_center + bh/2) * h / 480))
    return frame[y1:y2, x1:x2]

# ─────────────────────────────
# EDGE OCR — Tesseract
# ─────────────────────────────
def edge_ocr(plate_crop):
    if plate_crop.size == 0:
        return "UNKNOWN"
    plate_crop = cv2.resize(plate_crop, (0, 0), fx=4, fy=4)
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    _, thresh1 = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = cv2.bitwise_not(thresh1)
    thresh3 = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    results = []
    for thresh in [thresh1, thresh2, thresh3]:
        text = pytesseract.image_to_string(thresh, config=config).strip()
        text = ''.join(c for c in text.upper() if c.isalnum())
        if len(text) >= 3:
            results.append(text)
    if not results:
        return "UNKNOWN"
    from collections import Counter
    return Counter(results).most_common(1)[0][0]

# ─────────────────────────────
# CLOUD OCR — Plate Recognizer
# ─────────────────────────────
def cloud_ocr(plate_crop):
    if plate_crop.size == 0:
        return "UNKNOWN"
    try:
        # Encode image
        _, buffer = cv2.imencode('.jpg',
                    cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR))
        img_bytes = buffer.tobytes()

        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            headers={'Authorization': f'Token {PLATE_RECOGNIZER_TOKEN}'},
            files={'upload': ('plate.jpg', img_bytes, 'image/jpeg')},
            timeout=5
        )

        data = response.json()
        if data.get('results'):
            plate = data['results'][0]['plate'].upper()
            plate = ''.join(c for c in plate if c.isalnum())
            return plate if len(plate) >= 3 else "UNKNOWN"
    except Exception as e:
        print(f"Cloud OCR error: {e}")
    return "UNKNOWN"

# ─────────────────────────────
# HYBRID SCHEDULER
# ─────────────────────────────
def hybrid_ocr(frame, bbox, yolo_confidence):
    plate_crop = crop_plate(frame, bbox)
    cv2.imwrite("captures/last_crop.jpg",
                cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR))

    # Check in-memory cache first
    # (Redis will replace this in Objective 2)
    now = time.time()
    for plate, (cached_time, cached_text) in list(plate_cache.items()):
        if now - cached_time > CACHE_TTL:
            del plate_cache[plate]

    # Hybrid decision
    if yolo_confidence >= EDGE_CONF_THRESHOLD:
        # High confidence — use edge OCR
        stats["edge_ocr"] += 1
        ocr_method = "EDGE"
        plate_text = edge_ocr(plate_crop)
    else:
        # Low confidence — escalate to cloud
        stats["cloud_ocr"] += 1
        ocr_method = "CLOUD"
        plate_text = cloud_ocr(plate_crop)
        # Fall back to edge if cloud fails
        if plate_text == "UNKNOWN":
            plate_text = edge_ocr(plate_crop)
            ocr_method = "EDGE-FB"

    # Cache the result
    if plate_text != "UNKNOWN":
        plate_cache[plate_text] = (now, plate_text)

    return plate_text, ocr_method

# ─────────────────────────────
# PRINT STATS
# ─────────────────────────────
def print_stats():
    total = stats["total"]
    if total == 0:
        return
    cloud_pct = (stats["cloud_ocr"] / total) * 100
    edge_pct = (stats["edge_ocr"] / total) * 100
    reduction = 100 - cloud_pct
    print(f"\n{'='*60}")
    print(f"  HYBRID SCHEDULER STATS")
    print(f"  Total detections : {total}")
    print(f"  Edge OCR         : {stats['edge_ocr']} ({edge_pct:.1f}%)")
    print(f"  Cloud OCR        : {stats['cloud_ocr']} ({cloud_pct:.1f}%)")
    print(f"  Cloud reduction  : {reduction:.1f}% vs cloud-only baseline")
    print(f"{'='*60}\n")

# ─────────────────────────────
# MAIN LOOP
# ─────────────────────────────
print("\n🚗 Smart Gate HYBRID System — Press Ctrl+C to stop\n")
print(f"{'Time':<10} {'Plate':<12} {'Conf':<8} {'OCR':<8} {'Risk':<8} {'Flag':<18} {'Decision'}")
print("-" * 82)

frame_count = 0
detection_interval = 30

try:
    while True:
        frame = picam2.capture_array()
        frame_count += 1

        if frame_count % detection_interval != 0:
            continue

        start = time.time()
        input_data = preprocess(frame)
        outputs = session.run(None, {input_name: input_data})
        detections = postprocess(outputs)
        latency = (time.time() - start) * 1000

        stats["total"] += 1

        if detections:
            best = max(detections, key=lambda x: x["confidence"])
            yolo_conf = best["confidence"]

            if yolo_conf >= CLOUD_CONF_THRESHOLD:
                # Run hybrid OCR
                plate_text, ocr_method = hybrid_ocr(
                    frame, best["bbox"], yolo_conf)

                # Behaviour module
                decision, risk_score, flag = gate_decision(
                    plate_text, WHITELIST)

                # Save frame
                timestamp = time.strftime("%H%M%S")
                cv2.imwrite(f"captures/frame_{timestamp}.jpg",
                           cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                print(f"{timestamp:<10} {plate_text:<12} {yolo_conf:.3f}{'':>3} "
                      f"{ocr_method:<8} {risk_score:<8.2f} {flag:<18} {decision}")
            else:
                stats["no_detection"] += 1
                print(f"{time.strftime('%H%M%S'):<10} {'Low conf':<12} "
                      f"{yolo_conf:.3f}{'':>3} {'—':<8} {'—':<8} {'—':<18} {'—'}")
        else:
            stats["no_detection"] += 1
            print(f"{time.strftime('%H%M%S'):<10} {'No plate':<12} {'—':<8} "
                  f"{'—':<8} {'—':<8} {'—':<18} {'—'}")

except KeyboardInterrupt:
    print("\n\nSystem stopped.")
    print_stats()
    picam2.stop()
