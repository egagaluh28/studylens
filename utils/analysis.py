import numpy as np
from tensorflow.keras.models import load_model

# Load LSTM model
lstm_model = load_model("models/head_movement_lstm.h5")

# utils/analysis.py
def analyze_head_movement(head_positions):
    focus_count = head_positions.count("focus")
    distracted_count = head_positions.count("distracted")
    sleepy_count = head_positions.count("sleepy")
    not_detected_count = head_positions.count("not_detected")

    # Analisis berdasarkan jumlah setiap kategori
    if focus_count > distracted_count and focus_count > sleepy_count:
        return "High Focus"
    elif distracted_count > focus_count and distracted_count > sleepy_count:
        return "Distracted"
    elif sleepy_count > focus_count and sleepy_count > distracted_count:
        return "Sleepy"
    else:
        return "Not Detected"
