import cv2
import numpy as np
import tensorflow as tf
import json
from cvzone.HandTrackingModule import HandDetector

# Load model
model = tf.keras.models.load_model("hand_sign_model.h5")

# Load label map
with open("labels.json", "r") as f:
    label_map = json.load(f)

# Reverse the label map using integer keys
reverse_label_map = {int(v): k for k, v in label_map.items()}

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
img_size = 300

while True:
    success, img = cap.read()
    if not success:
        print("Failed to access webcam.")
        break

    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgHeight, imgWidth, _ = img.shape

        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgWidth, x + w + offset)
        y2 = min(imgHeight, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            continue  # Skip if crop failed

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

        # Prepare image for prediction
        img_input = imgWhite.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        try:
            prediction = model.predict(img_input)
            class_index = np.argmax(prediction)
            class_name = reverse_label_map[class_index]

            # Draw prediction
            cv2.putText(img, class_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        except Exception as e:
            print("Prediction error:", e)

    cv2.imshow("Prediction", img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
