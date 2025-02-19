# SIGNOVA: American Sign Language (ASL) Recognition 

![SIGNOVA Banner](signova_final_banner.png)

## Introduction
SIGNOVA is a software application that utilizes a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) hand gestures from images. The recognized gestures are displayed as text, allowing for effective communication between sign language users and non-sign language users. The software is particularly beneficial for individuals with hearing or speech impairments, helping to bridge the communication gap.

## Features
- Captures hand gestures using a webcam.
- Processes the image through a trained CNN model.
- Accurately identifies the corresponding ASL sign.
- Displays the recognized sign in text format.
- Works under varying lighting conditions and backgrounds.
- Provides a graphical user interface (GUI) for ease of use.

## System Flow
1. **User Input:** The user makes a sign language gesture in front of a webcam.
2. **Image Acquisition:** The system captures an image of the gesture.
3. **Preprocessing:**
   - Hand is detected using the MediaPipe library.
   - Region of Interest (ROI) is extracted.
   - Image is converted to grayscale and binarized using OpenCV.
4. **Feature Extraction:**
   - Hand landmarks are extracted and analyzed.
5. **Gesture Classification:**
   - CNN processes the extracted features and predicts the corresponding ASL sign.
6. **Output Display:**
   - The predicted character is displayed to the user.

## Project Modules
1. **Data Acquisition:**
   - Capturing images of hand gestures using a webcam.
   - Vision-based method for cost-effective hand tracking.
2. **Data Preprocessing & Feature Extraction:**
   - Using OpenCV for image processing (grayscale conversion, Gaussian blur, binary conversion).
   - Utilizing MediaPipe for hand landmark extraction.
3. **Gesture Classification:**
   - CNN model trained on alphabetic gestures (A-Z) with a dataset of 180 images per alphabet.
   - Improved classification accuracy by grouping similar gestures and refining classifications.
4. **Implementation & Testing:**
   - GUI developed using Tkinter.
   - Prediction accuracy: 97% in general conditions; 99% in optimal lighting/background conditions.

## Technologies Used
- **Python** (Core language)
- **OpenCV** (Image processing)
- **MediaPipe** (Hand landmark detection)
- **Keras & TensorFlow** (CNN model implementation)
- **Tkinter** (GUI development)

## Future Work
- Enhance classification for more complex gestures.
- Develop an Android application for real-time ASL recognition.
- Expand the dataset for improved accuracy across diverse hand shapes and skin tones.

## How to Run the Project
1. Install dependencies:
   ```bash
   pip install opencv-python mediapipe tensorflow keras numpy
