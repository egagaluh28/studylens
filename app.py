from flask import Flask, render_template, Response, jsonify
import cv2
import time
from utils.detection import detect_head_position
from utils.analysis import analyze_head_movement

app = Flask(__name__)

# Inisialisasi kamera dengan resolusi lebih rendah
camera = cv2.VideoCapture(0)  # Use index 0 for the default camera
if not camera.isOpened():
    print("Error: Could not open the camera. Please check the camera connection.")
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Resolusi lebar
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Resolusi tinggi

# Variabel sesi
session_data = {
    "head_positions": [],
    "focus": 0,
    "distracted": 0,
    "sleepy": 0,
    "not_detected": 0,
    "total_frames": 0,
    "start_time": None,
    "end_time": None,
}

def generate_frames():
    global session_data

    # Catat waktu mulai sesi
    if session_data['start_time'] is None:
        session_data['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S")

    while True:
        success, frame = camera.read()
        if not success:
            print("Warning: Failed to read frame from the camera.")
            break
        else:
            # Deteksi posisi kepala
            head_status = detect_head_position(frame)

            # Update data sesi
            session_data["total_frames"] += 1
            session_data["head_positions"].append(head_status)
            if head_status == "focus":
                session_data["focus"] += 1
            elif head_status == "distracted":
                session_data["distracted"] += 1
            elif head_status == "sleepy":
                session_data["sleepy"] += 1
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

    # Catat waktu akhir sesi
    session_data['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")

@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_session', methods=['POST'])
def start_session():
    global session_data
    session_data["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"message": "Session started", "start_time": session_data["start_time"]})

@app.route('/stop_session', methods=['POST'])
def stop_session():
    global session_data
    session_data["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"message": "Session stopped", "end_time": session_data["end_time"]})

@app.route('/summary')
def summary():
    global session_data

    # Calculate total session duration in seconds
    if session_data["start_time"] and session_data["end_time"]:
        start_time_struct = time.strptime(session_data["start_time"], "%Y-%m-%d %H:%M:%S")
        end_time_struct = time.strptime(session_data["end_time"], "%Y-%m-%d %H:%M:%S")
        duration_seconds = time.mktime(end_time_struct) - time.mktime(start_time_struct)
        duration_minutes = duration_seconds / 60
    else:
        duration_minutes = 0

    # Ensure no division by zero
    total_frames = max(session_data["total_frames"], 1)

    # Calculate percentages
    focus_percent = (session_data["focus"] / total_frames) * 100
    distracted_percent = (session_data["distracted"] / total_frames) * 100
    sleepy_percent = (session_data["sleepy"] / total_frames) * 100
    not_detected_percent = (session_data["not_detected"] / total_frames) * 100

    # Analyze head movement for focus level
    focus_level = analyze_head_movement(session_data["head_positions"])

    # Prepare summary result
    result = {
        "today": time.strftime("%Y-%m-%d"),
        "start_time": session_data["start_time"],
        "end_time": session_data["end_time"],
        "duration_minutes": round(duration_minutes, 2),
        "focus_percent": round(focus_percent, 2),
        "distracted_percent": round(distracted_percent, 2),
        "sleepy_percent": round(sleepy_percent, 2),
        "not_detected_percent": round(not_detected_percent, 2),
        "focus_level": focus_level,  # Use the analyzed focus level
        "distractions": session_data["distracted"],
    }

    # Return the result before resetting session data
    response = jsonify(result)

    # Reset session data for the next session
    session_data = {
        "head_positions": [],
        "focus": 0,
        "distracted": 0,
        "sleepy": 0,
        "not_detected": 0,
        "total_frames": 0,
        "start_time": None,
        "end_time": None,
    }

    return response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)