"""
Violence Detection Flask Backend

Run:  python app.py
Then open:  http://localhost:5000
"""

import os
import uuid
import threading
import json
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Folders ─
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


jobs = {}   # job_id -> { status, progress, message, output_path }



YOLO_MODEL_PATH     = "yolov8m.pt"
VIOLENCE_MODEL_PATH = "C:Users/HP/OneDrive/Desktop/project_2nd/best_r3d18.pt"   # <-- put your trained model here



def run_pipeline(job_id, input_path, output_path):
    """Runs the full YOLO + DeepSort + Violence-Detection pipeline in a thread."""

    jobs[job_id]["status"]  = "running"
    jobs[job_id]["progress"] = 0
    jobs[job_id]["message"] = "Loading models…"

    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn as nn
        import torchvision.models.video as tv_models
        from ultralytics import YOLO
        from deep_sort_realtime.deepsort_tracker import DeepSort

        CONF_THRESHOLD  = 0.5
        MAX_AGE         = 30
        PERSON_CLASS_ID = 0
        NUM_FRAMES      = 16
        IMG_SIZE        = 112
        CLASS_NAMES     = ["NonFight", "Fight"]
        VIOLENCE_THRESH = 0.7
        CONSECUTIVE_HITS= 3
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Build violence model
        def build_model():
            m = tv_models.r3d_18(weights=None)
            m.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2),
            )
            return m

        def get_color(track_id):
            palette = [
                (0,255,0),(255,0,0),(0,0,255),(0,255,255),
                (255,0,255),(255,255,0),(128,0,255),(0,128,255),
                (0,255,128),(255,128,0),
            ]
            return palette[int(track_id) % len(palette)]

        # ── Load YOLO 
        jobs[job_id]["message"] = "Loading YOLOv8…"
        yolo    = YOLO(YOLO_MODEL_PATH)
        tracker = DeepSort(max_age=MAX_AGE)

        # ── Load violence model 
        jobs[job_id]["message"] = "Loading violence model…"
        v_model = build_model()
        ckpt    = torch.load(VIOLENCE_MODEL_PATH, map_location=DEVICE)
        if isinstance(ckpt, dict):
            sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
        else:
            sd = ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        v_model.load_state_dict(sd)
        v_model.to(DEVICE).eval()

        frame_buffer  = []
        score_history = []
        last_label    = "NonFight"
        last_prob     = 0.0

        def update_violence(frame):
            nonlocal frame_buffer, score_history, last_label, last_prob
            frame_buffer.append(frame)
            if len(frame_buffer) > NUM_FRAMES:
                frame_buffer.pop(0)

            if len(frame_buffer) == NUM_FRAMES:
                clip_np = np.array([
                    cv2.cvtColor(cv2.resize(f, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
                    for f in frame_buffer
                ], dtype=np.float32) / 255.0
                clip = torch.tensor(clip_np).permute(3,0,1,2).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = v_model(clip)
                    probs  = torch.softmax(logits, dim=1)[0]
                last_prob  = probs[1].item()
                last_label = CLASS_NAMES[probs.argmax().item()]
                score_history.append(last_prob > VIOLENCE_THRESH)
                if len(score_history) > CONSECUTIVE_HITS:
                    score_history.pop(0)

            confirmed = (len(score_history) == CONSECUTIVE_HITS and all(score_history))
            return last_label, last_prob, confirmed

        #open video 
        cap    = cv2.VideoCapture(input_path)
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx          = 0
        total_persons_seen = set()

        jobs[job_id]["message"] = "Processing frames…"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            output = frame.copy()

            # YOLO detection
            results      = yolo(frame, verbose=False)
            detections   = []
            person_count = 0
            for box in results[0].boxes:
                if int(box.cls[0]) != PERSON_CLASS_ID or float(box.conf[0]) < CONF_THRESHOLD:
                    continue
                person_count += 1
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                detections.append(([x1,y1,x2-x1,y2-y1], float(box.conf[0]), "person"))

            # DeepSort tracking
            tracks     = tracker.update_tracks(detections, frame=frame)
            active_ids = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                active_ids.append(tid)
                total_persons_seen.add(tid)
                x1,y1,x2,y2 = map(int, track.to_ltrb())
                color = get_color(tid)
                cv2.rectangle(output, (x1,y1), (x2,y2), color, 2)
                cv2.putText(output, f"ID:{tid} Person",
                            (x1, max(y1-10,15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # Violence detection
            v_label, v_prob, v_confirmed = update_violence(frame)

            
            cv2.rectangle(output, (0,0), (width,80), (0,0,0), -1)
            cv2.putText(output,
                        f"Persons: {person_count}  |  Tracking: {len(active_ids)}",
                        (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)

            if v_confirmed:
                vtxt   = f"VIOLENCE CONFIRMED ({v_prob*100:.1f}%)"
                vcolor = (0,0,255)
            elif v_label == "Fight":
                vtxt   = f"Possible Fight ({v_prob*100:.1f}%)"
                vcolor = (0,165,255)
            else:
                vtxt   = f"Normal ({(1-v_prob)*100:.1f}%)"
                vcolor = (0,220,0)

            cv2.putText(output, vtxt, (10,62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, vcolor, 2)
            cv2.putText(output, f"Frame {frame_idx}/{total}",
                        (width-180, height-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

            writer.write(output)

            # Update progress
            if total > 0:
                pct = int(frame_idx / total * 100)
                jobs[job_id]["progress"] = pct
                jobs[job_id]["message"]  = f"Frame {frame_idx}/{total} | {v_label} ({v_prob*100:.1f}%)"

        cap.release()
        writer.release()

        jobs[job_id]["status"]        = "done"
        jobs[job_id]["progress"]      = 100
        jobs[job_id]["message"]       = f"Done! {frame_idx} frames · {len(total_persons_seen)} unique persons"
        jobs[job_id]["output_path"]   = output_path

    except Exception as e:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = str(e)



#  Routes

@app.route("/")
def index():
    """Serve the frontend HTML file."""
    return send_file("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accepts a video file, saves it, starts the pipeline in a background thread.
    Returns a job_id the frontend can poll.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file    = request.files["video"]
    job_id  = str(uuid.uuid4())[:8]
    ext     = os.path.splitext(file.filename)[1] or ".mp4"
    in_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_input{ext}")
    out_path= os.path.join(OUTPUT_FOLDER, f"{job_id}_output.mp4")

    file.save(in_path)

    jobs[job_id] = {
        "status":      "queued",
        "progress":    0,
        "message":     "Queued…",
        "output_path": None,
    }

    thread = threading.Thread(target=run_pipeline, args=(job_id, in_path, out_path), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    """Poll this endpoint to get job progress."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@app.route("/download/<job_id>")
def download(job_id):
    """Download the processed output video."""
    if job_id not in jobs or not jobs[job_id].get("output_path"):
        return jsonify({"error": "Output not ready"}), 404
    return send_file(jobs[job_id]["output_path"], as_attachment=True, download_name="result.mp4")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Violence Detection Server")
    print("="*50)
    print("  Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
