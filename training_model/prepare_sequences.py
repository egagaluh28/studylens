import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Konfigurasi
IMAGE_SIZE = (64, 64)
SEQUENCE_LENGTH = 10
DATASET_PATH = "dataset"
LABELS = ["focus", "distracted", "sleepy", "not_detected"]

# Fungsi untuk muat dan resize gambar
def load_images(label_path):
    images = []
    for filename in sorted(os.listdir(label_path)):
        img_path = os.path.join(label_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            img = img / 255.0
            images.append(img)
    return images

# Siapkan data dan label
X = []
y = []

for label_idx, label_name in enumerate(LABELS):
    full_path = os.path.join(DATASET_PATH, label_name)
    images = load_images(full_path)

    # Bagi menjadi sequence
    for i in range(0, len(images) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
        sequence = images[i:i+SEQUENCE_LENGTH]
        if len(sequence) == SEQUENCE_LENGTH:
            X.append(sequence)
            y.append(label_idx)

X = np.array(X)
y = np.array(y)

print(f"Total sequence: {X.shape[0]}")
print(f"Shape per sequence: {X.shape[1:]}")

# Simpan hasilnya
np.save("X.npy", X)
np.save("y.npy", y)
print("Data sequence berhasil disimpan: X.npy, y.npy")
