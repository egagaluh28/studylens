from flask import Flask, render_template, Response, jsonify
import cv2
import time
from utils.detection import detect_head_position
from utils.analysis import summarize_focus

app = Flask(__name__)

# Inisialisasi kamera
camera = cv2.VideoCapture(0)

# Variabel sesi
session_data = {
    "focus": 0,
    "distracted": 0,
    "not_detected": 0,
    "total_frames": 0,
    "start_time": None,
}

def generate_frames():
    global session_data

    session_data['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S")

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Deteksi arah kepala
            head_status = detect_head_position(frame)

            # Update data sesi
            session_data["total_frames"] += 1
            if head_status == "focus":
                session_data["focus"] += 1
            elif head_status == "distracted":
                session_data["distracted"] += 1
            else:
                session_data["not_detected"] += 1

            # Tambahkan teks ke frame
            cv2.putText(frame, f"Status: {head_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame ke JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Streaming frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/summary')
def summary():
    result = summarize_focus(session_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
