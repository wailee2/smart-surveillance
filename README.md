# 🚦 Smart Surveillance & Speed Detection System

> A FastAPI-powered traffic surveillance application that detects vehicles, estimates speed using optical flow + monocular depth, flags red-light violations, reads license plates via OCR, and produces an annotated output video with a full offender report — **runs entirely on CPU, no GPU required.**

---

## 📸 Demo

| Input Video | Annotated Output |
|-------------|-----------------|
| Place your input traffic video at `demo/sample_input.mp4` | Annotated output is saved to `static/outputs/<job_id>_output.mp4` |

> 📂 **`demo/`** — place two videos here before sharing your repo:
> - `demo/sample_input.mp4`   — a short clip of traffic (5–30 s is ideal)
> - `demo/sample_output.mp4`  — the annotated output from the app

The dashboard looks like this when a job completes:

```
┌─────────────────────────────────────────────┐
│  SurveillanceAI  │  Upload │ Results │ Vio  │
├──────────────────┴─────────────────────────┤
│  🎬 Processing Complete                     │
│  Frames: 750 │ Duration: 25s │ Time: 4m2s  │
│  ┌────────┬──────────┬──────┐              │
│  │ Speed  │ Red Light│ Both │ …            │
│  │  3     │    1     │  0   │              │
│  └────────┴──────────┴──────┘              │
│  [▶ Output Video]                          │
│  [⬇ Download Video] [⬇ Download CSV]      │
└────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| **Vehicle Detection** | YOLOv11 + ByteTrack multi-object tracking (car, bus, truck, bike, 3-wheeler) |
| **Speed Estimation** | Monocular depth (Depth Anything V2 Small) + Farneback optical flow on CPU |
| **Red-Light Detection** | YOLOv11 classifier for traffic light state (red / yellow / green) |
| **License Plate OCR** | YOLOv11 plate detector + EasyOCR text extraction |
| **Violation Logging** | Per-track offender log: plate, speed, violation type, snapshot, timestamp |
| **Annotated Video** | Bounding boxes, track IDs, live speed overlays, traffic light state bar |
| **CSV Export** | Machine-readable offender report |
| **REST API** | FastAPI backend with background job queue and SSE-style polling |
| **Clean UI** | Dark-mode responsive dashboard — drag-and-drop upload, live progress |
| **CPU Mode** | Auto-detected; Farneback flow replaces RAFT, larger frame intervals |

---

## 🛠 Technology Stack

| Layer | Tool | Version |
|-------|------|---------|
| **Web Framework** | FastAPI | 0.111 |
| **Server** | Uvicorn (ASGI) | 0.29 |
| **Object Detection** | Ultralytics YOLOv11 | 8.2 |
| **Object Tracking** | ByteTrack (via Ultralytics) | — |
| **Detection Utils** | Supervision | 0.21 |
| **Depth Estimation** | Depth Anything V2 Small (HuggingFace) | — |
| **Optical Flow (CPU)** | OpenCV Farneback | 4.9 |
| **Optical Flow (GPU)** | RAFT Large (torchvision) | — |
| **OCR** | EasyOCR | 1.7 |
| **Deep Learning** | PyTorch (CPU build) | 2.3 |
| **Computer Vision** | OpenCV | 4.9 |
| **Data** | Pandas, NumPy | 2.2 / 1.26 |
| **Frontend** | Vanilla HTML/CSS/JS (no build step) | — |

---

## 💻 Running Without a GPU

This application is designed to work on CPU-only machines. Here is what changes automatically:

| Component | GPU Mode | CPU Mode (auto-selected) |
|-----------|----------|--------------------------|
| Optical Flow | RAFT Large (~30 RAFT iterations) | OpenCV Farneback (single-pass, ~30× faster) |
| Depth Updates | Every 20 frames | Every 40 frames |
| Frame Baseline | 20-frame gap | 30-frame gap |
| YOLO Inference | CUDA tensors | CPU tensors |
| EasyOCR | GPU-accelerated | CPU only |
| Expected Speed | ~5–15 fps | ~0.3–1.5 fps |

**Rule of thumb on CPU:** expect 3–10× real-time. A 30-second clip takes roughly 2–8 minutes depending on resolution and hardware.

### Tips to speed things up on CPU

1. **Lower the resolution** of your input video before uploading:
   ```bash
   ffmpeg -i input.mp4 -vf scale=640:-1 input_small.mp4
   ```
2. **Increase frame intervals** in `.env`:
   ```
   FRAME_INTERVAL_CPU=50
   DEPTH_EVERY_N_CPU=60
   ```
3. **Use a short clip** — the app is intended as a demo, not production batch processing.
4. **Close other applications** to free RAM and CPU cores.
5. **Use a machine with many CPU cores** — PyTorch and OpenCV are multi-threaded.

---

## ⚙️ Prerequisites

- **Python** 3.10 or 3.11 (3.12 not yet fully supported by all dependencies)
- **pip** ≥ 23
- **FFmpeg** (for video output compatibility — optional but recommended)
- **~6 GB disk space** (model weights + HuggingFace cache)
- **~4 GB RAM** minimum (8 GB recommended)
- Your three trained YOLOv11 `.pt` weight files:
  - `weights/traffic_light_best.pt`
  - `weights/vehicle_best.pt`
  - `weights/license_plate_best.pt`

> **No GPU required.** If a CUDA GPU is present, it is used automatically.

---

## 🚀 Step-by-Step Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-surveillance.git
cd smart-surveillance
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

**CPU-only (default, recommended for most users):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**If you have a CUDA GPU (optional speedup):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Place your model weights

Option A — copy your files manually:
```bash
mkdir -p weights
cp /path/to/traffic_light_best.pt  weights/
cp /path/to/vehicle_best.pt        weights/
cp /path/to/license_plate_best.pt  weights/
```

Option B — use the helper script:
```bash
python download_weights.py \
  --traffic /path/to/traffic_light_best.pt \
  --vehicle /path/to/vehicle_best.pt \
  --plate   /path/to/license_plate_best.pt
```

### 5. Configure (optional)

```bash
cp .env.example .env
# Edit .env to adjust SPEED_LIMIT_KMH, thresholds, frame intervals, etc.
```

### 6. Start the server

```bash
python run.py
```

Or directly with uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 7. Open the dashboard

Navigate to [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📖 How to Use

1. **Open** [http://localhost:8000](http://localhost:8000)
2. **Drag and drop** (or click Browse) to select a traffic video (MP4, AVI, MOV)
3. **Adjust** the speed limit and detection confidence if needed
4. Click **Start Processing** — the job runs in the background
5. Watch the **live progress bar** — processing FPS is shown in real time
6. When complete, the **results panel** shows:
   - Frames processed, video duration, processing time
   - Violation counts (speeding / red-light)
   - An inline video player for the **annotated output**
   - Download buttons for the **annotated MP4** and **offender CSV**
7. Scroll down to the **Violation Log** table for per-track details

---

## 📂 Project Structure

```
smart-surveillance/
│
├── app/
│   ├── main.py                  # FastAPI app, lifespan, routes mount
│   ├── core/
│   │   ├── config.py            # All settings (Pydantic BaseSettings + .env)
│   │   └── model_manager.py     # Loads & caches all ML models at startup
│   ├── routers/
│   │   ├── inference.py         # POST /api/process, GET /api/job/{id}, …
│   │   └── health.py            # GET /health
│   └── services/
│       ├── inference_service.py # Full pipeline orchestrator
│       └── speed_pipeline.py    # EMA, Kalman, depth scale, motion estimator, tracker
│
├── templates/
│   └── index.html               # Single-page dashboard (Jinja2)
│
├── static/
│   ├── css/style.css            # Dark-mode responsive stylesheet
│   ├── js/app.js                # Frontend logic (upload, polling, results)
│   ├── uploads/                 # Incoming video files (auto-created)
│   └── outputs/                 # Annotated videos, CSVs, snapshots (auto-created)
│       └── snapshots/           # Per-track vehicle snapshots
│
├── weights/                     # Model weights (YOU provide these .pt files)
│   ├── traffic_light_best.pt
│   ├── vehicle_best.pt
│   └── license_plate_best.pt
│
├── demo/                        # Place demo videos here for the README
│   ├── sample_input.mp4
│   └── sample_output.mp4
│
├── run.py                       # Convenience launcher
├── download_weights.py          # Weight placement helper
├── requirements.txt
├── .env.example                 # Template — copy to .env
└── README.md
```

---

## 🔑 Key Sections Explained

### `app/core/config.py`
Central configuration using **Pydantic `BaseSettings`**. Every tunable parameter lives here and can be overridden via environment variable or `.env` file. Auto-detects CUDA availability and sets `CPU_MODE` accordingly.

### `app/core/model_manager.py`
Loaded once at server startup via FastAPI's `lifespan` context manager. Stores all models as attributes and exposes `is_ready()`. In CPU mode, RAFT is skipped (None) and Farneback is used instead.

### `app/services/speed_pipeline.py`
Contains the full numerical stack:
- `EMAFilter` — exponential moving average for box coords, depth, scale
- `KalmanSpeedFilter` — 1-D Kalman filter for per-track speed smoothing
- `MedianSpeedBuffer` — rolling median for spike rejection
- `DepthScaleEstimator` — Depth Anything V2 → relative depth → metres-per-pixel
- `MotionEstimator` — Farneback (CPU) or RAFT (GPU) dense optical flow
- `SpeedTracker` — combines scale + motion into km/h estimates with stationary detection

### `app/services/inference_service.py`
Frame-by-frame pipeline loop. For each frame:
1. Traffic light inference → sets red-light violation flag
2. Vehicle detection + ByteTrack → tracking IDs with ROI offset
3. License plate detection → plate crop buffered per track
4. Speed tracker update → depth + flow → km/h
5. Annotation overlay written to output video
Post-loop: OCR runs on buffered plate crops, offender CSV is saved.

### `app/routers/inference.py`
REST API for the UI:
- `POST /api/process` — uploads video, starts background task, returns `job_id`
- `GET /api/job/{id}` — returns `{status, progress, fps_proc, result}`
- `GET /api/job/{id}/video` — streams the output MP4
- `GET /api/job/{id}/csv` — streams the offender CSV

---

## 📤 Output Files

| File | Location | Description |
|------|----------|-------------|
| Annotated video | `static/outputs/<job_id>_output.mp4` | Input video with bounding boxes, track IDs, speed overlays, info bar |
| Offender CSV | `static/outputs/<job_id>_offenders.csv` | Per-track violation record |
| Snapshots | `static/outputs/snapshots/track<N>_*.jpg` | Vehicle image at time of violation |

### Offender CSV columns

| Column | Type | Description |
|--------|------|-------------|
| `track_id` | int | ByteTrack persistent tracking ID |
| `vehicle_type` | str | car / bus / truck / bike / 3wheeler |
| `plate_number` | str | OCR-extracted plate text, or `UNREADABLE` |
| `speed_kmh` | float | Display-stable smoothed speed |
| `speed_method` | str | `depth_v2+farneback` (CPU) or `depth_v2+raft` (GPU) |
| `violation_type` | str | `speeding`, `red_light`, or `speeding,red_light` |
| `snapshot_path` | str | Path to the saved vehicle snapshot image |
| `timestamp_frame` | int | Frame number of the snapshot |
| `timestamp_real` | float | Time in seconds from video start |

---

## ⚙️ Config Reference

All values can be set in `.env` or passed as environment variables.

| Key | Default | Description |
|-----|---------|-------------|
| `SPEED_LIMIT_KMH` | `50.0` | Speed above which a vehicle is flagged |
| `CONF_THRESHOLD` | `0.4` | YOLO detection confidence minimum |
| `IOU_THRESHOLD` | `0.5` | Non-maximum suppression IoU threshold |
| `FRAME_INTERVAL_CPU` | `30` | Optical flow baseline frame gap (CPU) |
| `FRAME_INTERVAL_GPU` | `20` | Optical flow baseline frame gap (GPU) |
| `DEPTH_EVERY_N_CPU` | `40` | Run depth model every N frames (CPU) |
| `DEPTH_EVERY_N_GPU` | `20` | Run depth model every N frames (GPU) |
| `KALMAN_PROCESS_NOISE` | `0.5` | Kalman Q — lower = smoother, slower reaction |
| `KALMAN_MEASUREMENT_NOISE` | `8.0` | Kalman R — higher = trusts raw reading less |
| `MEDIAN_BUFFER_FRAMES` | `9` | Rolling median window for spike rejection |
| `SPEED_DISPLAY_DELTA_KMH` | `2.0` | Min change before display speed updates |
| `MIN_FLOW_PIXELS` | `1.5` | Ignore flow below this magnitude (noise gate) |
| `STATIONARY_CHECK_SECONDS` | `0.5` | Window for stationary vehicle detection |
| `OCR_CONFIDENCE_THRESHOLD` | `0.4` | Min EasyOCR confidence to accept text |

---

## ⚠️ Limitations

1. **Speed accuracy is approximate.** The system uses monocular relative depth (no stereo or LiDAR). Speed estimates are best understood as relative indicators rather than precision measurements. Expect ±20–40% error depending on scene geometry.

2. **Farneback flow is noisier than RAFT.** On CPU, Farneback dense flow is used as a compromise between speed and accuracy. RAFT would produce more accurate motion vectors but is 20–50× slower on CPU.

3. **Single camera, no calibration.** No camera intrinsic/extrinsic parameters are used. The depth model produces a relative map, not metric depth. A calibrated camera would significantly improve speed accuracy.

4. **OCR struggles with motion blur and low resolution.** Plates captured at high speed or low resolution may return `UNREADABLE`. Slowing the video or using a higher-resolution source helps.

5. **Tracking ID drift.** ByteTrack can lose a track and reassign a new ID if occlusion is long. This may result in the same physical vehicle appearing as two entries in the log.

6. **CPU processing is slow.** A 60-second 1080p clip can take 10–20 minutes on a modern CPU. This is a demo application, not a real-time system on CPU.

7. **Fixed ROI.** The detection region of interest is fixed to the middle 50% of the frame height. Videos where vehicles occupy the top or bottom 25% may have missed detections.

8. **No multi-lane differentiation.** Speed is estimated per track without lane assignment.

---

## 💡 Possible Solutions to Limitations

| Limitation | Possible Solution |
|------------|-------------------|
| Speed accuracy | Add camera calibration (OpenCV `calibrateCamera`); use stereo depth or a known reference object size |
| Slow CPU processing | Use a dedicated GPU; or deploy on a cloud GPU instance (Google Colab, RunPod, AWS g4dn) |
| Farneback noise | On GPU, RAFT is auto-selected; alternatively use PWC-Net which is faster than RAFT |
| OCR failures | Pre-process plate crops: super-resolution (Real-ESRGAN), histogram equalisation, perspective correction |
| Tracking drift | Tune ByteTrack parameters; use ReID (Re-Identification) models to re-link lost tracks |
| Fixed ROI | Make ROI user-configurable via the UI; add a drag-to-draw ROI selector |
| Single camera | Add a second camera angle for cross-validation; use a GPS-equipped drone for absolute speed |

---

## 🔗 References

- **YOLOv11 (Ultralytics)**: https://docs.ultralytics.com
- **ByteTrack**: Zhang et al., 2022 — https://arxiv.org/abs/2110.06864
- **Depth Anything V2**: Yang et al., 2024 — https://depth-anything-v2.github.io
- **RAFT Optical Flow**: Teed & Deng, 2020 — https://arxiv.org/abs/2003.12039
- **Farneback Optical Flow**: Farneback, 2003 — OpenCV docs
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **FastAPI**: https://fastapi.tiangolo.com
- **Supervision**: https://supervision.roboflow.com

---

## 📜 License

This project was developed as an academic assignment / demo. It is provided for educational purposes. Model weights trained on third-party datasets are subject to their respective dataset licences (Roboflow, COCO, etc.).

---

## 🙋 FAQ

**Q: The server starts but says "Models not found".**
A: Place your `.pt` files in the `weights/` directory. Run `python download_weights.py --help` for options.

**Q: The video processes but all speeds show "--".**
A: The Depth Anything V2 model may not have downloaded yet (first run downloads ~100 MB from HuggingFace). Check your internet connection and wait for the first run to complete.

**Q: EasyOCR takes a long time on first run.**
A: EasyOCR downloads its own model (~50 MB) on first initialisation. This is a one-time download.

**Q: Can I use this on Windows?**
A: Yes. All dependencies support Windows. Use `venv\Scripts\activate` instead of `source venv/bin/activate`.

**Q: How do I use a GPU if I have one?**
A: Install the CUDA PyTorch build (see Step 3). The app auto-detects CUDA and switches to GPU mode — no code changes needed.
"# smart-surveillance" 
