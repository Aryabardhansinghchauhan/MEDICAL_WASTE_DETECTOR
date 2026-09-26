# Medical Waste Detector

A clean, high-performance real-time computer vision application for detecting medical waste items using an Ultralytics YOLO `.pt` model.

## Features

- **Multi-Source Support**: Open local webcams (`--source 0`), video files (`--source video.mp4`), or RTSP stream URLs (`--source rtsp://...`).
- **Dynamic Model Class Loading**: Reads class names directly from `model.names` (falling back to `data/data.yaml`).
- **Decoupled Waste Mapping**: Maps detected objects to waste categories (`WHITE`, `RED`, `REVIEW`, etc.) using `config/waste_mapping.yaml`. Unmapped classes default to `category: REVIEW` and require verification.
- **Resilient Camera Engine**: Threaded frame capture drops queued old frames to eliminate latency and automatically attempts reconnection if live streams disconnect.
- **Stable Rendering**: Consistent class-specific color assignment for bounding boxes and segmentation masks.
- **Smoothed FPS Calculation**: Exponential moving average FPS meter for stable performance readouts.
- **REST API & Web UI Dashboard**: Built-in FastAPI server (`--web`) providing `/health`, `/predict`, live camera MJPEG feed, and clean dashboard UI.
- **MQTT Event Streaming**: Publishes standard JSON event payloads (`json.dumps`) to configured MQTT topics.
- **Logging & Video Recording**: Save per-frame detection events to JSONL (`--save-json runs/detections.jsonl`) and export annotated video recordings (`--output runs/output.mp4`).

---

## Directory Structure

```
medical-waste-detector/
│
├── models/
│   └── best.pt
│
├── data/
│   └── data.yaml
│
├── config/
│   └── waste_mapping.yaml
│
├── runs/
│
├── medical_waste_detector.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage Examples

### 1. Basic Webcam Detection (OpenCV Window)
```bash
python medical_waste_detector.py --weights models/best.pt --source 0 --show
```

### 2. Video File Detection
```bash
python medical_waste_detector.py --weights models/best.pt --source video.mp4 --show
```

### 3. RTSP Camera Stream
```bash
python medical_waste_detector.py --weights models/best.pt --source "rtsp://user:password@camera-ip/stream" --show
```

### 4. GPU / CPU Device Selection
```bash
# Force CPU execution
python medical_waste_detector.py --weights models/best.pt --source 0 --device cpu --show

# Force GPU 0
python medical_waste_detector.py --weights models/best.pt --source 0 --device 0 --show
```

### 5. Web UI Dashboard & REST API
```bash
python medical_waste_detector.py --weights models/best.pt --source 0 --web --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

### 6. Save JSONL & Export Annotated Video
```bash
python medical_waste_detector.py \
  --weights models/best.pt \
  --source 0 \
  --show \
  --save-json runs/detections.jsonl \
  --output runs/output.mp4
```

### 7. Optional MQTT Event Publishing
```bash
python medical_waste_detector.py \
  --weights models/best.pt \
  --source 0 \
  --show \
  --mqtt-broker localhost \
  --mqtt-port 1883 \
  --mqtt-topic medical_waste/detections
```

---

## API Documentation

### `GET /health`
Returns system and model readiness status:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`
Accepts an image file payload (`multipart/form-data`) and returns structured detections:
```json
{
  "detections": [
    {
      "class_name": "syringe",
      "confidence": 0.91,
      "bbox": [120, 80, 340, 420],
      "waste_category": "WHITE",
      "requires_review": true
    }
  ]
}
```

---

## Safety & Compliance Note

This application performs visual detection of objects. The waste mapping (`config/waste_mapping.yaml`) is a configurable software layer. Visual detection alone does NOT determine legal compliance, chemical composition, or contamination status. Disposal decisions must be verified against institutional standard operating procedures (SOPs) and applicable local bio-medical waste regulations.
