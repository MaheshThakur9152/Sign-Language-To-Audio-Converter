import time
import cv2
import numpy as np
try:
    from cvzone.HandTrackingModule import HandDetector
except Exception:
    HandDetector = None
from keras.models import load_model

MODEL_PATH = 'cnn8grps_rad1_model.h5'
SAMPLE_IMG = 'AtoZ_3.1/Z/0.jpg'

print('Loading model...')
model = load_model(MODEL_PATH)
print('Model loaded')

# Prepare white template
white = np.ones((400, 400, 3), dtype=np.uint8) * 255
white_input = white.astype('float32') / 255.0
white_input = np.expand_dims(white_input, 0)

# Warmup inference
print('Warming up inference...')
for _ in range(3):
    _ = model.predict(white_input)

# Benchmark inference
iters = 50
t0 = time.time()
for _ in range(iters):
    _ = model.predict(white_input)

t1 = time.time()
print(f'Inference: {iters} runs, total {t1-t0:.3f}s, avg {(t1-t0)/iters*1000:.2f} ms')

# Hand detector benchmark (optional)
if HandDetector is None:
    print('cvzone/mediapipe not available; skipping hand-detection benchmark')
else:
    print('Loading hand detector...')
    hd = HandDetector(maxHands=1)
    img = cv2.imread(SAMPLE_IMG)
    if img is None:
        raise RuntimeError(f'Could not read sample image: {SAMPLE_IMG}')

    # Warmup detection
    for _ in range(3):
        _ = hd.findHands(img, draw=False)

    # Benchmark detection
    iters = 50
    t0 = time.time()
    for _ in range(iters):
        _ = hd.findHands(img, draw=False)

    t1 = time.time()
    print(f'HandDetector: {iters} runs, total {t1-t0:.3f}s, avg {(t1-t0)/iters*1000:.2f} ms')

print('Done')
