import streamlit as st
st.set_page_config(page_title="Hand Sign Translator Dashboard", layout="wide")

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
    reverse_label_map = {v: k for k, v in label_map.items()}
    return model, reverse_label_map

model, reverse_label_map = load_model()

st.title("🤖 Hand Sign Translator Dashboard")

# Sidebar Controls
st.sidebar.header("⚙️ Controls")
if 'running' not in st.session_state:
    st.session_state.running = False

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

if st.sidebar.button("▶️ Start Translation"):
    st.session_state.running = True
if st.sidebar.button("⏹️ Stop Translation"):
    st.session_state.running = False

st.sidebar.markdown("---")
st.sidebar.markdown("[🎥 Example Video](202505261654.mp4)")


# Layout
col1, col2 = st.columns([2, 1])
FRAME_WINDOW = col1.image([])
prediction_text_placeholder = col2.empty()

# Initialize
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
img_size = 300

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
        confidence = float(prediction[0][class_index])

        if confidence >= confidence_threshold:
            class_name = reverse_label_map[class_index]
            prediction_text = f"Prediction: **{class_name}**\nConfidence: **{confidence:.2f}**"

            # Draw subtitle on frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            text_size = cv2.getTextSize(class_name, font, font_scale, thickness)[0]
            text_x = int((img.shape[1] - text_size[0]) / 2)
            text_y = img.shape[0] - 30

            # White outline
            cv2.putText(img, class_name, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
            # Black text
            cv2.putText(img, class_name, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        else:
            prediction_text = f"Prediction: **Uncertain**\nConfidence: **{confidence:.2f}** (below threshold)"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(img_rgb)
    prediction_text_placeholder.markdown(prediction_text)

cap.release()
