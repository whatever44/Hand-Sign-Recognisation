import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf

img_size = 300
data_path = "data"
X = []
y = []
labels = sorted([name for name in os.listdir(data_path)
                 if os.path.isdir(os.path.join(data_path, name))])

label_map = {label: idx for idx, label in enumerate(labels)}

with open("labels.json", "w") as f:
    json.dump(label_map, f)

for label in labels:
    folder_path = os.path.join(data_path, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        y.append(label_map[label])

X = np.array(X) / 255.0
y = np.array(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
model.save("hand_sign_model.h5")