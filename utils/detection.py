import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_head_position(frame):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if not results.detections:
            return "not_detected"

        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = bboxC.xmin * iw, bboxC.ymin * ih, bboxC.width * iw, bboxC.height * ih

            center_x = x + w / 2
            center_percentage = center_x / iw

            aspect_ratio = w / h

            # Logika fokus lebih akurat:
            if 0.85 <= aspect_ratio <= 1.15 and 0.4 <= center_percentage <= 0.6:
                return "focus"
            elif center_percentage < 0.4 or center_percentage > 0.6:
                return "distracted"
            else:
                return "distracted"

        return "not_detected"

