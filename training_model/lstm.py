import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels

# 📦 Load Data
print("📦 Memuat data sequence...")
X = np.load("X.npy")
y = np.load("y.npy")

# 🔄 Reshape (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], -1))

# ✅ Normalisasi
X_reshaped = X.reshape(-1, X.shape[-1])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(X.shape)

# 📊 Cek distribusi label
unique, counts = np.unique(y, return_counts=True)
print("\n📊 Distribusi Label:")
for label, count in zip(unique, counts):
    print(f"Label {label}: {count} sampel")

# 🔀 Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)

# 🧠 Build Improved LSTM Model
def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.4),
        LSTM(64),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ⚙️ Setup Model
input_shape = (X.shape[1], X.shape[2])
num_classes = len(np.unique(y))
model = build_lstm_model(input_shape, num_classes)

from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 🔢 Hitung class weight
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# ⏱️ Callbacks
os.makedirs("models", exist_ok=True)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ModelCheckpoint("models/best_lstm_model.h5", save_best_only=True)
]

# 🚀 Train Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weights,
    shuffle=True,
    verbose=1
)

# 💾 Save Final Model
model.save("models/final_head_movement_lstm.h5")
print("✅ Model akhir disimpan ke models/final_head_movement_lstm.h5")

# 📈 Evaluation
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n📊 Akurasi Validasi: {val_acc * 100:.2f}%")
print(f"📉 Loss Validasi: {val_loss:.4f}")

# 📊 Grafik Akurasi & Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.title("Akurasi per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# 🔍 Prediksi & Evaluasi
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 📋 Classification Report (fix untuk mencegah error label tidak cocok)
label_map = {0: "focus", 1: "distracted", 2: "sleepy", 3: "not_detected"}
reverse_label_map = {v: k for k, v in label_map.items()}

actual_labels = unique_labels(y_val, y_pred)
target_names = [label_map[i] for i in actual_labels]

print("\n📋 Classification Report:")
print(classification_report(y_val, y_pred, target_names=target_names))
