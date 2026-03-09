from picamera2 import Picamera2
import cv2
import numpy as np
import onnxruntime as ort
import time

# Load model
session = ort.InferenceSession("models/best_nano.onnx")
input_name = session.get_inputs()[0].name

# Start camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

# Capture frame
print("Capturing frame...")
frame = picam2.capture_array()
cv2.imwrite("debug_frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
print("Frame saved as debug_frame.jpg")

# Run inference with LOW threshold to see ALL detections
img = cv2.resize(frame, (640, 640))
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0).astype(np.float32)
img /= 255.0

outputs = session.run(None, {input_name: img})
predictions = outputs[0][0]

print("\nAll detections (confidence > 0.1):")
found = False
for pred in predictions:
    conf = pred[4]
    if conf > 0.1:
        print(f"  Confidence: {conf:.3f} | bbox: {pred[0]:.1f}, {pred[1]:.1f}, {pred[2]:.1f}, {pred[3]:.1f}")
        found = True

if not found:
    print("  Nothing detected at all — plate may not be visible to camera")

picam2.stop()
