import numpy as np

# Parameter dataset
num_samples = 1000  # Jumlah sampel
num_frames = 30     # Jumlah frame per sampel
num_features = 4    # Jumlah fitur per frame
num_classes = 4     # Jumlah kelas (focus, distracted, sleepy, not_detected)

# Buat data input (X)
X = np.random.rand(num_samples, num_frames, num_features)

# Buat label output (y)
y = np.random.randint(0, num_classes, size=(num_samples,))

# Simpan dataset
np.save("X_lstm.npy", X)
np.save("y_lstm.npy", y)

print("Dataset berhasil dibuat: X_lstm.npy dan y_lstm.npy")