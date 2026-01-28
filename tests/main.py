import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from PIL import Image, ImageTk

# -------------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------------
# MUST match the labels used in your training 
ACTIONS = np.array(['hello', 'thanks', 'iloveyou']) 
MODEL_PATH = 'action.h5' 
SEQUENCE_LENGTH = 30
THRESHOLD = 0.7

# -------------------------------------------------------------------------
# 2. GUI CLASS
# -------------------------------------------------------------------------
class SignLanguageGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("850x600")
        self.window.configure(bg="#f0f0f0")

        # -----------------------------------------------------------------
        # FIX: Rebuild Model Architecture instead of using load_model()
        # -----------------------------------------------------------------
        print("Building Model Architecture...")
        try:
            self.model = Sequential()
            # Input shape matches the training data: (30 frames, 1662 landmarks)
            self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
            self.model.add(LSTM(128, return_sequences=True, activation='relu'))
            self.model.add(LSTM(64, return_sequences=False, activation='relu'))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dense(len(ACTIONS), activation='softmax'))

            print("Loading Weights...")
            # Load only the weights, ignoring the incompatible layer configs
            self.model.load_weights(MODEL_PATH)
            print("Model Loaded Successfully!")
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            # If weights fail, compile=False sometimes helps if passing the full file
            try:
                print("Attempting fallback load...")
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            except:
                print("Could not load model. Check if action.h5 exists.")
                exit()
        
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # UI Elements
        self.header_label = tk.Label(window, text="SIGNOVA - Test Mode", font=("Helvetica", 24, "bold"), bg="#f0f0f0")
        self.header_label.pack(pady=10)

        self.video_label = tk.Label(window)
        self.video_label.pack()

        self.result_frame = tk.Frame(window, bg="#fff", bd=2, relief="groove")
        self.result_frame.pack(fill="x", padx=20, pady=20)
        
        self.pred_label = tk.Label(self.result_frame, text="Waiting...", font=("Helvetica", 32, "bold"), bg="#fff", fg="#007bff")
        self.pred_label.pack(pady=10)

        self.prob_bar = ttk.Progressbar(window, orient="horizontal", length=400, mode="determinate")
        self.prob_bar.pack(pady=5)

        # Variables
        self.sequence = []
        self.cap = cv2.VideoCapture(0)

        # Start Loop
        self.update()
        self.window.mainloop()

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            image, results = self.mediapipe_detection(frame, self.holistic)
            self.draw_styled_landmarks(image, results)
            
            # Prediction Logic
            keypoints = self.extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-SEQUENCE_LENGTH:]

            if len(self.sequence) == SEQUENCE_LENGTH:
                res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                action_idx = np.argmax(res)
                confidence = res[action_idx]

                if confidence > THRESHOLD:
                    self.pred_label.config(text=ACTIONS[action_idx].upper())
                    self.prob_bar["value"] = confidence * 100
                else:
                    self.pred_label.config(text="...")
                    self.prob_bar["value"] = 0

            # Convert for Tkinter
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.window.after(10, self.update)

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        # Draw connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) 
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) 

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageGUI(root, "SIGNOVA")