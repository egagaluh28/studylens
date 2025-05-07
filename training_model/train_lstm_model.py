import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Load data sequence
print("📦 Memuat data sequence...")
X = np.load("X.npy")  # Bentuk: (samples, timesteps, height, width, channels)
y = np.load("y.npy")

# Ubah bentuk untuk LSTM (flatten image per timestep)
X = X.reshape((X.shape[0], X.shape[1], -1))  # Misal: (samples, 10, 64*64*3)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Build LSTM
def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Training LSTM
print("🚀 Melatih model LSTM...")
input_shape = (X.shape[1], X.shape[2])  # (timesteps, features)
num_classes = len(np.unique(y))

model = build_lstm_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32
)

# Simpan model
os.makedirs("models", exist_ok=True)
model.save("models/head_movement_lstm.h5")
print("✅ Model LSTM disimpan ke models/head_movement_lstm.h5")
