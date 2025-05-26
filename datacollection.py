import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Hand detector
detector = HandDetector(maxHands=2)

# Image settings
offset = 20
imgSize = 300
counter = 0

# Save folder
folder = "Data/Aavash"
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Clamp crop area to stay within image bounds
        imgHeight, imgWidth, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgWidth, x + w + offset)
        y2 = min(imgHeight, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Show processed windows
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            filename = f'{folder}/Image_{time.time()}.jpg'
            cv2.imwrite(filename, imgWhite)
            print(f"Saved: {filename}")

    # Show webcam feed
    cv2.imshow('Image', img)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
