import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
from cvzone.HandTrackingModule import HandDetector

# Load model and labels
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("hand_sign_model.h5")
    with open("labels.json", "r") as f:
        label_map = json.load(f)
    reverse_label_map = {v: k for k, v in label_map.items()}  # keys are ints here
    return model, reverse_label_map

model, reverse_label_map = load_model()

st.title("🤖 Hand Sign Translator")

# Create start and stop buttons, control a session state flag
if 'running' not in st.session_state:
    st.session_state.running = False

def start_translation():
    st.session_state.running = True

def stop_translation():
    st.session_state.running = False

col1, col2 = st.columns(2)
with col1:
    st.button("▶️ Start Translation", on_click=start_translation)
with col2:
    st.button("⏹️ Stop Translation", on_click=stop_translation)

FRAME_WINDOW = st.image([])
prediction_text_placeholder = st.empty()

detector = HandDetector(maxHands=1)
offset = 20
img_size = 300

cap = cv2.VideoCapture(0)

while st.session_state.running:
    success, img = cap.read()
    if not success:
        st.error("Cannot access webcam")
        break

    hands, img = detector.findHands(img)
    prediction_text = "Waiting for hand..."

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgHeight, imgWidth, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgWidth, x + w + offset)
        y2 = min(imgHeight, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255

        aspectRatio = h / w
        if aspectRatio > 1:
            k = img_size / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, img_size))
            wGap = (img_size - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = img_size / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (img_size, hCal))
            hGap = (img_size - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        img_input = imgWhite / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        prediction = model.predict(img_input)
        class_index = np.argmax(prediction)
        class_name = reverse_label_map[class_index]  # <-- Fixed here

        prediction_text = f"Prediction: **{class_name}**"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(img_rgb)
    prediction_text_placeholder.markdown(prediction_text)

cap.release()
