from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import math
import base64
from gtts import gTTS
import io
import traceback

app = FastAPI()

# Enable CORS for your web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

@app.post("/tts")
async def text_to_speech(payload: TextPayload):
    try:
        if not payload.text:
            return {"audio": ""}
        tts = gTTS(text=payload.text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_b64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
        return {"audio": audio_b64}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- LOAD MODEL ---
model = None
try:
    model = tf.keras.models.load_model("cnn8grps_rad1_model.h5")
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"CRITICAL: Model failed to load: {e}")

# --- INIT DETECTORS ---
detector = HandDetector(maxHands=1)
detector2 = HandDetector(maxHands=1)
offset = 29
imgSize = 400

# ==========================================
# 1. NEW: GLOBAL STATE FOR SENTENCE BUILDING
# ==========================================
current_sentence = ""       # The full sentence
last_prediction = None      # The character seen in the previous frame
stability_counter = 0       # How many times we've seen the same char
STABILITY_THRESHOLD = 5     # Frames required to "lock in" a letter
last_added_char = None      # The last character we actually added to the sentence

def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

# ==========================================
# 2. HELPER: RESET ENDPOINT
# ==========================================
@app.post("/reset")
def reset_sentence():
    global current_sentence, last_prediction, stability_counter, last_added_char
    current_sentence = ""
    last_prediction = None
    stability_counter = 0
    last_added_char = None
    return {"status": "cleared", "sentence": ""}

@app.post("/backspace")
def backspace_sentence():
    global current_sentence
    current_sentence = current_sentence[:-1]
    return {"status": "updated", "sentence": current_sentence}

# ==========================================
# 3. EXISTING LOGIC (Minimally Modified)
# ==========================================
def predict_gesture(test_image, pts):
    try:
        if model is None: return "Error"
        
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        
        # Get raw probabilities
        prediction = model.predict(white, verbose=0)[0]
        prob = np.array(prediction, dtype='float32')

        # --- FIX 1: CONFIDENCE THRESHOLD ---
        # If the highest probability is less than 80%, return nothing
        if np.max(prob) < 0.8:
            return ""  # Return empty string to indicate uncertainty

        # Get top 3 predictions for your logic
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (pts[5][0] < pts[4][0]):
                ch1 = 0

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if distance(pts[8], pts[16]) < 52:
                ch1 = 2

        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
                ch1 = 3

        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[4][0] > pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[2][1] + 15 < pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if distance(pts[4], pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (distance(pts[4], pts[11]) > 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[4][0] < pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[1][0] < pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[4][0] > pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[3][0] < pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1] < pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[18][1] > pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[5][0] > pts[16][0]:
                ch1 = 6

        # condition for [yj][x]
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if distance(pts[8], pts[16]) > 50:
                ch1 = 6

        # con for [l][x]
        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if distance(pts[4], pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[5][0] - pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0], [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (distance(pts[4], pts[11]) < 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[5][0] - pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[4][0] < pts[5][0] + 15) and ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])) and pts[4][1] > pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and pts[0][0] + fg < pts[20][0]) and not (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50:
                ch1 = 1

        # con for [w]
        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
                ch1 = 1

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        if ch1 == 0:
            ch1 = 'S'
            if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                ch1 = 'A'
            if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
                ch1 = 'T'
            if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                ch1 = 'E'
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
                ch1 = 'M'
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if distance(pts[12], pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (distance(pts[8], pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if distance(pts[8], pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                if pts[8][1] < pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                ch1 = 'B'
            else:
                if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] == pts[16][1]:
                    ch1 = 'D'
                else:
                    if pts[12][0] > pts[4][0] and pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                        ch1 = 'D'
                    else:
                        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
                            ch1 = 'R'
                        else:
                            if pts[4][0] > pts[12][0] and pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                                ch1 = 'U'
                            else:
                                if pts[4][0] > pts[12][0] and pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                                    ch1 = 'U'
                                else:
                                    if pts[4][0] < pts[12][0] and pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                                        ch1 = 'V'
                                    else:
                                        if pts[4][0] < pts[12][0] and pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
                                            ch1 = 'W'
                                        else:
                                            if pts[4][0] > pts[12][0] and pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] == pts[20][1]:
                                                ch1 = 'K'
                                            else:
                                                if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] == pts[20][1] and pts[4][0] > pts[12][0]:
                                                    ch1 = 'K'
                                                else:
                                                    if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] == pts[20][1] and pts[4][0] < pts[12][0]:
                                                        ch1 = 'F'
                                                    else:
                                                        if pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
                                                            ch1 = 'I'

        return ch1
        
    except Exception as e:
        print(f"Logic Error: {e}")
        return "Error"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global current_sentence, last_prediction, stability_counter, last_added_char

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None: return {"prediction": "Error", "sentence": current_sentence}

        hands, img = detector.findHands(img, draw=False)
        
        raw_char = ""  # What the model sees RIGHT NOW
        
        if hands:
            try:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # --- FIX: SAFE CROP & SIZE CHECK ---
                h_img, w_img, _ = img.shape
                y1, y2 = max(0, y - offset), min(h_img, y + h + offset)
                x1, x2 = max(0, x - offset), min(w_img, x + w + offset)

                image = img[y1:y2, x1:x2]

                # Ignore tiny slivers (prevent division by zero)
                if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
                    raw_char = "Adjust Hand"
                    # return {"prediction": "Adjust Hand", "sentence": current_sentence}

                white = np.ones((400, 400, 3), np.uint8) * 255
                
                # 2. Second Detection
                handz, image_out = detector2.findHands(image, draw=False, flipType=True)
                
                if handz:
                    hand_crop = handz[0]
                    pts = hand_crop['lmList']
                    
                    # --- FIX: SAFE RESIZE LOGIC ---
                    crop_h, crop_w, _ = image.shape # Use actual crop size
                    aspectRatio = crop_h / crop_w

                    if aspectRatio > 1:
                        k = imgSize / crop_h
                        wCal = math.ceil(k * crop_w)
                        imgResize = cv2.resize(image, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        # Safe copy
                        end_x = min(wGap + imgResize.shape[1], 400)
                        white[:, wGap:end_x] = imgResize[:, :end_x - wGap]
                    else:
                        k = imgSize / crop_w
                        hCal = math.ceil(k * crop_h)
                        imgResize = cv2.resize(image, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        # Safe copy
                        end_y = min(hGap + imgResize.shape[0], 400)
                        white[hGap:end_y, :] = imgResize[:end_y - hGap, :]

                    # --- DRAWING (Wrapped in Try/Except) ---
                    # 1. Normalize Points to the Crop Box (0.0 to 1.0 range)
                    # This makes the drawing independent of how far the hand is from the camera
                    min_x = min([p[0] for p in pts])
                    min_y = min([p[1] for p in pts])
                    max_x = max([p[0] for p in pts])
                    max_y = max([p[1] for p in pts])
                    w_box = max_x - min_x
                    h_box = max_y - min_y

                    # 2. Draw RELATIVE to the white canvas size (400x400)
                    # Instead of adding 'os' (offset), we map the hand directly to the canvas
                    scale = 350 / max(w_box, h_box) # Leave 25px margin
                    offset_x = (400 - (w_box * scale)) // 2
                    offset_y = (400 - (h_box * scale)) // 2

                    # Create a normalized points list
                    norm_pts = []
                    for p in pts:
                        nx = int((p[0] - min_x) * scale + offset_x)
                        ny = int((p[1] - min_y) * scale + offset_y)
                        norm_pts.append([nx, ny])

                    # 3. Draw using 'norm_pts' instead of 'pts'
                    # (Use your existing drawing loops, but replace 'pts[t][0] + os' with 'norm_pts[t][0]')
                    for t in range(0, 4, 1):
                        cv2.line(white, (norm_pts[t][0], norm_pts[t][1]), (norm_pts[t+1][0], norm_pts[t+1][1]), (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (norm_pts[t][0], norm_pts[t][1]), (norm_pts[t+1][0], norm_pts[t+1][1]), (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (norm_pts[t][0], norm_pts[t][1]), (norm_pts[t+1][0], norm_pts[t+1][1]), (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (norm_pts[t][0], norm_pts[t][1]), (norm_pts[t+1][0], norm_pts[t+1][1]), (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (norm_pts[t][0], norm_pts[t][1]), (norm_pts[t+1][0], norm_pts[t+1][1]), (0, 255, 0), 3)
                    cv2.line(white, (norm_pts[5][0], norm_pts[5][1]), (norm_pts[9][0], norm_pts[9][1]), (0, 255, 0), 3)
                    cv2.line(white, (norm_pts[9][0], norm_pts[9][1]), (norm_pts[13][0], norm_pts[13][1]), (0, 255, 0), 3)
                    cv2.line(white, (norm_pts[13][0], norm_pts[13][1]), (norm_pts[17][0], norm_pts[17][1]), (0, 255, 0), 3)
                    cv2.line(white, (norm_pts[0][0], norm_pts[0][1]), (norm_pts[5][0], norm_pts[5][1]), (0, 255, 0), 3)
                    cv2.line(white, (norm_pts[0][0], norm_pts[0][1]), (norm_pts[17][0], norm_pts[17][1]), (0, 255, 0), 3)

                    for i in range(21):
                        cv2.circle(white, (norm_pts[i][0], norm_pts[i][1]), 2, (0, 0, 255), 1)

                    raw_char = predict_gesture(white, pts)
                
            except Exception as e:
                print(f"Error: {e}")
                raw_char = ""

        else:
            raw_char = "No Hand Detected"

        # ==========================================
        # 4. NEW: SENTENCE CONSTRUCTION LOGIC
        # ==========================================
        
        # Only process if we have a valid character (not empty, not error)
        valid_char = raw_char if raw_char and len(raw_char) == 1 else None

        if valid_char:
            # Check if it matches the last frame
            if valid_char == last_prediction:
                stability_counter += 1
            else:
                stability_counter = 0 # Reset if it flickers
                last_prediction = valid_char

            # If stable enough, and different from what we just added
            if stability_counter >= STABILITY_THRESHOLD:
                if valid_char != last_added_char:
                    current_sentence += valid_char
                    last_added_char = valid_char
                    stability_counter = 0 # Reset to require re-confirmation
        else:
            # If hand is lost/error, reset the "last added" so we can add the same letter again if needed
            if stability_counter > 0:
                stability_counter -= 1
        
        # ==========================================
        # 5. RESPONSE
        # ==========================================
        
        # We only generate audio for the WHOLE sentence when requested, 
        # or maybe just the new letter. For now, let's just return text.
        
        return {
            "prediction": raw_char,       # What is seen right now (e.g., "A")
            "sentence": current_sentence, # The history (e.g., "HELLOA")
            "audio": ""                   # Handle audio on frontend for full sentence
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)