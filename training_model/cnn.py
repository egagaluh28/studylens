import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Konfigurasi
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
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

# Evaluasi akhir
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"\n📊 Akurasi Validasi: {val_accuracy * 100:.2f}%")
print(f"📉 Loss Validasi: {val_loss:.4f}")

# Plot Accuracy dan Loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix dan Classification Report
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\n📋 Classification Report:")
from sklearn.utils.multiclass import unique_labels

# Membalik label_map: dari {nama: angka} menjadi {angka: nama}
reverse_label_map = {v: k for k, v in label_map.items()}

# Ambil label yang benar-benar muncul di data
actual_labels = unique_labels(y_val, y_pred)
target_names = [reverse_label_map[i] for i in actual_labels]

# Tampilkan classification report
print("\n📋 Classification Report:")
print(classification_report(y_val, y_pred, target_names=target_names))

