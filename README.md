# 🚨 AI-Based Violence Detection System

An AI-powered real-time violence detection system that analyzes video footage using **YOLOv8** for person detection, **DeepSort** for multi-object tracking, and a **3D ResNet (R3D-18)** deep learning model for fight/violence classification. The system is served via a **Flask** web application with a clean browser-based UI.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Configuration](#configuration)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This system processes uploaded video files and detects violent behavior (fighting) in real time. It draws bounding boxes around tracked individuals, assigns persistent IDs, and overlays a live violence classification label on each frame. The processed video can then be downloaded from the browser.

---

## ✨ Features

- 🎯 **Person Detection** — YOLOv8m identifies people in each video frame with configurable confidence thresholds.
- 🔁 **Multi-Object Tracking** — DeepSort assigns persistent IDs to individuals across frames.
- 🥊 **Violence Classification** — A fine-tuned R3D-18 (3D ResNet) model classifies 16-frame clips as `Fight` or `NonFight`.
- ✅ **Confirmed Violence Alerts** — Consecutive-hit logic reduces false positives before flagging `VIOLENCE CONFIRMED`.
- 🌐 **Web Interface** — Upload a video, monitor processing progress in real time, and download the annotated result — all from a browser.
- ⚡ **GPU Accelerated** — Automatically uses CUDA if available, falls back to CPU.
- 🧵 **Asynchronous Processing** — Video analysis runs in a background thread so the server stays responsive.

---

## 🏗️ System Architecture

```
Browser (index.html)
       │
       │  POST /upload  ──►  Flask (app.py)
       │                          │
       │  GET /status/<job_id>    ├─► Background Thread
       │                          │        │
       │  GET /download/<job_id>  │    YOLOv8m (person detection)
       └──────────────────────────┘        │
                                       DeepSort (tracking)
                                           │
                                       R3D-18 (violence classification)
                                           │
                                      Annotated MP4 output
```

---

## 📁 Project Structure

```
AI-Based-Violence-Detection-System/
│
├── app.py                  # Flask backend — detection pipeline & API routes
├── index.html              # Frontend UI — upload, progress bar, download
├── requirements.txt        # Python dependencies
├── yolov8m.pt              # YOLOv8 medium pre-trained weights
├── .gitignore
│
├── Models/                 # Trained violence classification model(s)
│
├── uploads/                # Temporary storage for uploaded input videos
└── outputs/                # Processed & annotated output videos
```

---

## 🔧 Prerequisites

- Python **3.8+**
- pip
- (Recommended) NVIDIA GPU with CUDA for faster inference
- A trained R3D-18 violence detection model (`best_r3d18.pt`)

---

## 🚀 Installation

1. **Clone the repository**

```bash
git clone https://github.com/panchariyarohit486-cmyk/AI-Based-Violence-Detection-System-.git
cd AI-Based-Violence-Detection-System-
```

2. **Create and activate a virtual environment** *(recommended)*

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add your trained violence model**

Place your fine-tuned `best_r3d18.pt` checkpoint inside the `Models/` folder and update the path in `app.py` if needed:

```python
# app.py
VIOLENCE_MODEL_PATH = "Models/best_r3d18.pt"
```

---

## ▶️ Usage

1. **Start the Flask server**

```bash
python app.py
```

You should see:

```
==================================================
 Violence Detection Server
==================================================
 Open http://localhost:5000 in your browser
==================================================
```

2. **Open the web interface**

Navigate to [http://localhost:5000](http://localhost:5000) in your browser.

3. **Upload a video**
   - Click **Choose File** and select an MP4, AVI, or MOV video.
   - Click **Upload & Analyze**.
   - A real-time progress bar shows frame-by-frame status.

4. **Download the result**

Once processing completes, click **Download Result** to get the annotated output video.

---

## ⚙️ How It Works

### Pipeline (per frame)

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **YOLOv8m** | Detects all persons in the frame (`class_id = 0`, confidence ≥ 0.5) |
| 2 | **DeepSort** | Tracks each person across frames with a persistent ID |
| 3 | **Frame Buffer** | Accumulates a rolling window of 16 frames |
| 4 | **R3D-18** | Classifies the 16-frame clip as `Fight` or `NonFight` with a probability score |
| 5 | **Consecutive Hits** | Confirms violence only after 3 consecutive positive detections above threshold (0.7) |
| 6 | **Annotation** | Overlays bounding boxes, IDs, and classification labels on the output frame |

### Violence Label Colors

| State | Label | Color |
|-------|-------|-------|
| Normal | `Normal (XX%)` | 🟢 Green |
| Possible | `Possible Fight (XX%)` | 🟠 Orange |
| Confirmed | `VIOLENCE CONFIRMED (XX%)` | 🔴 Red |

---

## 🧠 Model Details

### YOLOv8m (`yolov8m.pt`)
- Pre-trained on COCO dataset
- Used for real-time person detection
- Only persons (`class_id = 0`) are tracked

### R3D-18 Violence Classifier (`best_r3d18.pt`)
- Architecture: `torchvision.models.video.r3d_18` with a custom classification head
- Input: 16-frame clip × 3 channels × 112×112 pixels
- Output: 2 classes — `NonFight`, `Fight`
- Custom head:
  ```
  Linear(512 → 256) → ReLU → Dropout(0.5) → Linear(256 → 2)
  ```

---

## 🔩 Configuration

Key constants in `app.py` that can be tuned:

| Constant | Default | Description |
|----------|---------|-------------|
| `CONF_THRESHOLD` | `0.5` | Minimum YOLO detection confidence |
| `MAX_AGE` | `30` | DeepSort max frames to keep lost track alive |
| `NUM_FRAMES` | `16` | Clip length for violence classifier |
| `IMG_SIZE` | `112` | Frame resize resolution for the classifier |
| `VIOLENCE_THRESH` | `0.7` | Minimum fight probability to count as a hit |
| `CONSECUTIVE_HITS` | `3` | Number of consecutive hits to confirm violence |

---

## 📤 Output

The annotated output video includes:

- **Colored bounding boxes** around each tracked person with their unique ID
- **Top info bar** showing current person count and active track count
- **Violence status label** with confidence percentage
- **Frame counter** overlaid on the bottom-right corner

---

## 👤 Author

**Rohit Panchariya**
- GitHub: [@panchariyarohit486-cmyk](https://github.com/panchariyarohit486-cmyk)
