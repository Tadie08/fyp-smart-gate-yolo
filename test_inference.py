import onnxruntime as ort
import numpy as np
import cv2
import time

# Load model
print("Loading model...")
session = ort.InferenceSession("models/best_nano.onnx")
import onnxruntime as ort
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.inter_op_num_threads = 4
session = ort.InferenceSession("models/best_nano.onnx", sess_options=opts)
input_name = session.get_inputs()[0].name
print("Model loaded!")

# Create a dummy image (simulates a camera frame)
dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Preprocess
def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img /= 255.0
    return img

# Warmup run
print("Warming up...")
input_data = preprocess(dummy_img)
session.run(None, {input_name: input_data})

# Measure latency over 10 runs
print("Measuring latency...")
times = []
for i in range(10):
    start = time.time()
    input_data = preprocess(dummy_img)
    outputs = session.run(None, {input_name: input_data})
    end = time.time()
    latency = (end - start) * 1000
    times.append(latency)
    print(f"Run {i+1}: {latency:.1f}ms")

print(f"\nAverage latency: {np.mean(times):.1f}ms")
print(f"Min latency: {np.min(times):.1f}ms")
print(f"Max latency: {np.max(times):.1f}ms")
print(f"\nTarget <150ms: {'✅ PASSED' if np.mean(times) < 150 else '❌ Need optimization'}")
