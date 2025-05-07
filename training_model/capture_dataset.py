import cv2
import os

# Label dan deskripsi
labels = {
    "focus": "Menghadap ke depan (Fokus)",
    "distracted": "Menoleh kiri/kanan (Terdistraksi)",
    "sleepy": "Menunduk (Mengantuk)",
    "not_detected": "Wajah tidak terlihat (Tidak terdeteksi)"
}

# Buat direktori dataset dan subfolder label
os.makedirs("dataset", exist_ok=True)
for label in labels:
    os.makedirs(f"dataset/{label}", exist_ok=True)

# Mulai kamera
cap = cv2.VideoCapture(0)

print("📷 Tekan tombol berikut untuk menyimpan gambar:")
print("f = Fokus, d = Terdistraksi, s = Mengantuk, n = Tidak terdeteksi, q = Keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka kamera.")
        break

    cv2.imshow("Capture Dataset", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('f'):
        filename = f"dataset/focus/{len(os.listdir('dataset/focus'))}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[FOKUS] Disimpan: {filename}")

    elif key == ord('d'):
        filename = f"dataset/distracted/{len(os.listdir('dataset/distracted'))}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[TERDISTRAKSI] Disimpan: {filename}")

    elif key == ord('s'):
        filename = f"dataset/sleepy/{len(os.listdir('dataset/sleepy'))}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[MENGANTUK] Disimpan: {filename}")

    elif key == ord('n'):
        filename = f"dataset/not_detected/{len(os.listdir('dataset/not_detected'))}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[TIDAK TERDETEKSI] Disimpan: {filename}")

    elif key == ord('q'):
        print("Keluar dari program.")
        break

cap.release()
cv2.destroyAllWindows()
