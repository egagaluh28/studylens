import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Konfigurasi
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
label_map = {"focus": 0, "distracted": 1, "sleepy": 2, "not_detected": 3}

# Load dataset
def load_data(dataset_path="dataset"):
    images, labels = [], []
    for label in label_map:
        folder = os.path.join(dataset_path, label)
        if not os.path.exists(folder):
            print(f"Folder {folder} tidak ditemukan.")
            continue
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label_map[label])
            except Exception as e:
                print(f"Gagal memuat {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load & split data
print("📦 Memuat dan membagi dataset...")
X, y = load_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Augmentasi
print("🔁 Menyiapkan augmentasi data...")
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# CNN model
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Training CNN
print("🚀 Melatih model CNN...")
model = build_cnn_model((IMG_SIZE, IMG_SIZE, 3), num_classes=4)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS
)

# Simpan model
os.makedirs("models", exist_ok=True)
model.save("models/head_position_cnn.h5")
print("✅ Model CNN disimpan ke models/head_position_cnn.h5")
