def summarize_focus(session_data):
    total = session_data["total_frames"]
    if total == 0:
        return {}

    focus_percent = (session_data["focus"] / total) * 100
    distracted_percent = (session_data["distracted"] / total) * 100
    not_detected_percent = (session_data["not_detected"] / total) * 100

    # Tentukan level fokus berdasarkan persentase fokus
    focus_level = "High" if focus_percent >= 80 else "Medium" if focus_percent >= 60 else "Low"

    # Hitung durasi sesi (dengan asumsi 30 FPS)
    duration_minutes = total / 30 / 60

    return {
        "today": session_data["start_time"],
        "duration_minutes": round(duration_minutes, 2),
        "focus_percent": round(focus_percent, 2),
        "distracted_percent": round(distracted_percent, 2),
        "not_detected_percent": round(not_detected_percent, 2),
        "focus_level": focus_level,
        "distractions": session_data["distracted"],
    }