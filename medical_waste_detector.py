"""
Medical Waste Detector

Simple real-time YOLO detector for webcam, video, and RTSP sources.

Example CLI usage:
    python medical_waste_detector.py --weights models/best.pt --source 0 --show
    python medical_waste_detector.py --weights models/best.pt --source video.mp4 --show
    python medical_waste_detector.py --weights models/best.pt --source "rtsp://user:pass@ip/stream" --show
    python medical_waste_detector.py --weights models/best.pt --source 0 --web --port 8000

This program does not train a model.
It uses the existing YOLO .pt model.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import queue
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# Optional web API framework
try:
    import uvicorn
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, StreamingResponse

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

LOG = logging.getLogger("medical_waste_detector")

# Stable colors. The same class gets the same color during the run.
COLORS = [
    (0, 200, 255),    # Yellow-Orange
    (0, 180, 80),     # Green
    (255, 120, 0),    # Blue
    (180, 80, 255),   # Purple
    (80, 160, 255),   # Light Blue
    (0, 220, 220),    # Bright Yellow
    (220, 100, 100),   # Cyan
    (120, 220, 120),   # Light Green
    (50, 100, 255),    # Red-Orange
    (200, 50, 200),    # Magenta
]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]
    waste_category: str
    waste_label: str
    requires_review: bool
    review_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WasteRuleEngine:
    """Maps model class names to application-level waste categories.

    The mapping is deliberately separate from YOLO detection.
    Unknown classes are sent to REVIEW instead of being guessed.
    """

    def __init__(self, mapping_file: str | None):
        self.mapping: dict[str, dict[str, Any]] = {}

        if not mapping_file:
            LOG.info("No waste mapping supplied. Unknown categories will be REVIEW.")
            return

        path = Path(mapping_file)
        if not path.exists():
            LOG.warning("Mapping file not found: %s", path)
            return

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            mappings = data.get("mappings", {})

            for name, rule in mappings.items():
                self.mapping[self._normalise(name)] = rule or {}

            LOG.info("Loaded %d waste mappings from %s", len(self.mapping), path)
        except Exception as exc:
            LOG.error("Could not load mapping file: %s", exc)

    @staticmethod
    def _normalise(value: str) -> str:
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    def resolve(self, class_name: str) -> tuple[str, str, bool, str | None]:
        key = self._normalise(class_name)
        rule = self.mapping.get(key)

        if not rule:
            return (
                "REVIEW",
                "Manual verification",
                True,
                "No mapping configured for this model class",
            )

        category = str(rule.get("category", "REVIEW")).upper()
        label = str(rule.get("label", category))
        requires_review = bool(rule.get("requires_review", True))
        reason = rule.get("review_reason")

        if category == "REVIEW" and not reason:
            reason = "Manual verification required"

        return category, label, requires_review, reason


class FPSMeter:
    """Calculates smoothed FPS using an exponential moving average (EMA)."""

    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self.last_time: float | None = None
        self.fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()

        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 0:
                instant = 1.0 / dt
                if self.fps == 0:
                    self.fps = instant
                else:
                    self.fps = self.alpha * instant + (1.0 - self.alpha) * self.fps

        self.last_time = now
        return self.fps


class LatestFrameCapture:
    """Keeps only the newest camera frame.

    This prevents a live camera from building a long queue of old frames.
    Handles disconnection and reconnection for live streams.
    """

    def __init__(
        self,
        source: int | str,
        width: int = 1280,
        height: int = 720,
        reconnect: bool = True,
        reconnect_seconds: float = 2.0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.reconnect = reconnect
        self.reconnect_seconds = reconnect_seconds

        self.is_live = self._is_live_source(source)
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.finished = False
        self.connected = False

        self.cap: cv2.VideoCapture | None = None
        try:
            self.cap = self._open_capture()
            self.connected = True
        except Exception as exc:
            if not self.is_live:
                raise exc
            LOG.warning("Initial camera open failed: %s. Will retry...", exc)

        self.thread = threading.Thread(
            target=self._reader,
            name="camera-reader",
            daemon=True,
        )
        self.thread.start()

    @staticmethod
    def _is_live_source(source: int | str) -> bool:
        if isinstance(source, int):
            return True

        value = str(source).lower()
        return value.startswith(("rtsp://", "rtmp://", "http://", "https://"))

    def _open_capture(self) -> cv2.VideoCapture:
        LOG.info("Opening source: %s", self.source)

        cap = cv2.VideoCapture(self.source)

        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.source}")

        return cap

    def _put_latest(self, frame: np.ndarray) -> None:
        try:
            self.queue.put_nowait(frame)
            return
        except queue.Full:
            pass

        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass

    def _reader(self) -> None:
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                if not self.is_live or not self.reconnect:
                    self.finished = True
                    self.connected = False
                    break

                self.connected = False
                LOG.warning("Camera disconnected. Attempting reconnection...")
                if self.stop_event.wait(self.reconnect_seconds):
                    break
                try:
                    self.cap = self._open_capture()
                    self.connected = True
                    LOG.info("Camera reconnected successfully")
                except Exception as exc:
                    LOG.warning("Reconnect attempt failed: %s", exc)
                continue

            ok, frame = self.cap.read()

            if ok:
                self.connected = True
                self._put_latest(frame)
                continue

            # Read returned false
            if not self.is_live or not self.reconnect:
                self.finished = True
                self.connected = False
                break

            self.connected = False
            LOG.warning(
                "Camera read failed. Reconnecting in %.1f seconds...",
                self.reconnect_seconds,
            )

            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

            if self.stop_event.wait(self.reconnect_seconds):
                break

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.stop_event.set()

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


class MQTTPublisher:
    def __init__(
        self,
        broker: str,
        port: int,
        topic: str,
        username: str | None = None,
        password: str | None = None,
    ):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                "MQTT requested but paho-mqtt is not installed."
            ) from exc

        self.topic = topic
        self.client = mqtt.Client()

        if username:
            self.client.username_pw_set(username, password)

        LOG.info("Connecting to MQTT broker %s:%d", broker, port)
        self.client.connect(broker, port, keepalive=30)
        self.client.loop_start()

    def publish(self, payload: dict[str, Any]) -> None:
        message = json.dumps(payload, ensure_ascii=False)
        self.client.publish(self.topic, message, qos=0, retain=False)

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


def load_class_names(model: YOLO, data_file: str | None) -> dict[int, str]:
    """Use names stored in the model first, then data.yaml as fallback."""

    names = getattr(model, "names", None)

    if names:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(name) for i, name in enumerate(names)}

    if data_file:
        path = Path(data_file)
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            yaml_names = data.get("names", {})

            if isinstance(yaml_names, dict):
                return {int(k): str(v) for k, v in yaml_names.items()}
            if isinstance(yaml_names, (list, tuple)):
                return {i: str(name) for i, name in enumerate(yaml_names)}

    raise RuntimeError(
        "Could not find class names in the model or data.yaml."
    )


def parse_source(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def get_color(class_id: int) -> tuple[int, int, int]:
    return COLORS[class_id % len(COLORS)]


def safe_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    top = max(0, y - th - 10)
    bottom = max(th + 10, y)

    cv2.rectangle(
        image,
        (x, top),
        (x + tw + 10, bottom),
        color,
        -1,
    )

    cv2.putText(
        image,
        text,
        (x + 5, bottom - 5),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def extract_detections(
    result: Any,
    class_names: dict[int, str],
    rules: WasteRuleEngine,
) -> list[Detection]:
    detections: list[Detection] = []

    if result.boxes is None:
        return detections

    boxes = result.boxes

    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        confidence = float(boxes.conf[index].item())

        xyxy = boxes.xyxy[index].detach().cpu().numpy().tolist()
        x1, y1, x2, y2 = [max(0, int(round(v))) for v in xyxy]

        class_name = class_names.get(class_id, f"class_{class_id}")

        category, label, requires_review, reason = rules.resolve(class_name)

        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=round(confidence, 4),
                bbox=[x1, y1, x2, y2],
                waste_category=category,
                waste_label=label,
                requires_review=requires_review,
                review_reason=reason,
            )
        )

    return detections


def draw_results(
    frame: np.ndarray,
    result: Any,
    detections: list[Detection],
    show_masks: bool = True,
) -> np.ndarray:
    output = frame.copy()

    # Draw segmentation masks when the model provides them.
    if show_masks and getattr(result, "masks", None) is not None and result.masks is not None:
        overlay = output.copy()
        polygons = result.masks.xy

        for i, polygon in enumerate(polygons):
            if i >= len(detections):
                break

            if len(polygon) < 3:
                continue

            color = get_color(detections[i].class_id)
            points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [points], color)

        output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = get_color(detection.class_id)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        text = f"{detection.class_name} {detection.confidence:.2f}"

        if detection.waste_category:
            text += f" | {detection.waste_category}"

        if detection.requires_review:
            text += " | VERIFY"

        safe_text(
            output,
            text,
            (x1, max(30, y1)),
            color,
        )

    return output


def make_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fps = fps if fps and fps > 1 else 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")

    return writer


def make_payload(
    frame_id: int,
    source: int | str,
    detections: list[Detection],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "frame_id": frame_id,
        "detections": [
            {
                "class_name": item.class_name,
                "confidence": item.confidence,
                "waste_category": item.waste_category,
                "requires_review": item.requires_review,
                "bbox": item.bbox,
            }
            for item in detections
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple medical-waste YOLO detector"
    )

    parser.add_argument(
        "--weights",
        default="models/best.pt",
        help="Path to YOLO .pt model",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera number, video file, or RTSP URL",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="cpu, 0, 1, etc.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO image size",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Maximum detections per frame",
    )
    parser.add_argument(
        "--data",
        default="data/data.yaml",
        help="Fallback class-name YAML",
    )
    parser.add_argument(
        "--mapping",
        default="config/waste_mapping.yaml",
        help="Waste-category mapping YAML",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live OpenCV window",
    )
    parser.add_argument(
        "--output",
        help="Save annotated video",
    )
    parser.add_argument(
        "--save-json",
        help="Save detection results as JSONL",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Do not draw segmentation masks",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 when CUDA is available",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # Web Dashboard / FastAPI options
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start FastAPI web server and UI dashboard",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Web server host interface",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web server port",
    )

    # MQTT configuration options
    parser.add_argument(
        "--mqtt-broker",
        help="MQTT broker hostname/IP",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
    )
    parser.add_argument(
        "--mqtt-topic",
        default="medical_waste/detections",
    )
    parser.add_argument(
        "--mqtt-username",
    )
    parser.add_argument(
        "--mqtt-password",
    )
    parser.add_argument(
        "--publish-empty",
        action="store_true",
        help="Publish frames with no detections too",
    )

    return parser.parse_args()


# Global application state for FastAPI web interface
class DetectorState:
    def __init__(self) -> None:
        self.latest_annotated_frame: np.ndarray | None = None
        self.latest_detections: list[Detection] = []
        self.fps: float = 0.0
        self.model_loaded: bool = False
        self.model_name: str = ""
        self.camera_status: str = "Disconnected"
        self.frame_id: int = 0
        self.lock = threading.Lock()


GLOBAL_STATE = DetectorState()


def create_fastapi_app(
    model: YOLO,
    class_names: dict[int, str],
    rules: WasteRuleEngine,
    args: argparse.Namespace,
) -> FastAPI:
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI / uvicorn not installed.")

    app = FastAPI(title="Medical Waste Detector API", version="1.0.0")

    @app.get("/health")
    def health_check() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": GLOBAL_STATE.model_loaded,
        }

    @app.post("/predict")
    async def predict_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File provided is not an image.")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        results = model.predict(
            source=img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            half=args.half,
            verbose=False,
        )

        detections = []
        if results:
            detections = extract_detections(results[0], class_names, rules)

        return {
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "waste_category": d.waste_category,
                    "requires_review": d.requires_review,
                }
                for d in detections
            ]
        }

    @app.get("/video_feed")
    def video_feed():
        def generate():
            while True:
                with GLOBAL_STATE.lock:
                    frame = GLOBAL_STATE.latest_annotated_frame

                if frame is not None:
                    ok, jpeg = cv2.imencode(".jpg", frame)
                    if ok:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                        )
                time.sleep(0.04)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/state")
    def get_state():
        with GLOBAL_STATE.lock:
            det_list = [
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence * 100, 1),
                    "waste_category": d.waste_category,
                }
                for d in GLOBAL_STATE.latest_detections
            ]
            return {
                "fps": round(GLOBAL_STATE.fps, 1),
                "camera_status": GLOBAL_STATE.camera_status,
                "model_name": GLOBAL_STATE.model_name,
                "frame_id": GLOBAL_STATE.frame_id,
                "detections": det_list,
            }

    @app.get("/", response_class=HTMLResponse)
    def index_page():
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Medical Waste Detector</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                    background-color: #121212;
                    color: #e0e0e0;
                    margin: 0;
                    padding: 20px;
                }
                .container {
                    max-width: 1100px;
                    margin: 0 auto;
                }
                header {
                    border-bottom: 1px solid #333;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }
                h1 {
                    margin: 0;
                    font-size: 24px;
                    font-weight: 600;
                    color: #ffffff;
                }
                .grid {
                    display: grid;
                    grid-template-columns: 2fr 1fr;
                    gap: 20px;
                }
                .card {
                    background: #1e1e1e;
                    border: 1px solid #2d2d2d;
                    border-radius: 8px;
                    padding: 15px;
                }
                .video-container {
                    text-align: center;
                    background: #000;
                    border-radius: 6px;
                    overflow: hidden;
                }
                .video-container img {
                    width: 100%;
                    height: auto;
                    display: block;
                }
                .stat-group {
                    margin-bottom: 15px;
                }
                .stat-label {
                    font-size: 12px;
                    color: #888;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                .stat-value {
                    font-size: 16px;
                    font-weight: 500;
                    color: #fff;
                    margin-top: 4px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }
                th, td {
                    text-align: left;
                    padding: 8px 6px;
                    border-bottom: 1px solid #2d2d2d;
                    font-size: 14px;
                }
                th {
                    color: #888;
                    font-weight: 500;
                }
                .badge {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: 600;
                }
                .badge-WHITE { background: #e0e0e0; color: #121212; }
                .badge-RED { background: #d32f2f; color: #fff; }
                .badge-YELLOW { background: #fbc02d; color: #121212; }
                .badge-BLUE { background: #1976d2; color: #fff; }
                .badge-REVIEW { background: #7b1fa2; color: #fff; }
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>Medical Waste Detector</h1>
                </header>
                <div class="grid">
                    <div class="card">
                        <div class="video-container">
                            <img src="/video_feed" alt="Live Camera Feed">
                        </div>
                    </div>
                    <div class="card">
                        <div class="stat-group">
                            <div class="stat-label">Camera Status</div>
                            <div class="stat-value" id="camera-status">Connected</div>
                        </div>
                        <div class="stat-group">
                            <div class="stat-label">Model</div>
                            <div class="stat-value" id="model-name">best.pt</div>
                        </div>
                        <div class="stat-group">
                            <div class="stat-label">Performance</div>
                            <div class="stat-value"><span id="fps">0.0</span> FPS</div>
                        </div>
                        <div class="stat-group">
                            <div class="stat-label">Detected Objects</div>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Object</th>
                                        <th>Conf</th>
                                        <th>Category</th>
                                    </tr>
                                </thead>
                                <tbody id="detections-list">
                                    <tr><td colspan="3" style="color:#666;">No objects detected</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            <script>
                async function updateStats() {
                    try {
                        const res = await fetch('/api/state');
                        const data = await res.json();
                        document.getElementById('fps').innerText = data.fps;
                        document.getElementById('camera-status').innerText = data.camera_status;
                        document.getElementById('model-name').innerText = data.model_name;
                        
                        const tbody = document.getElementById('detections-list');
                        if (data.detections.length === 0) {
                            tbody.innerHTML = '<tr><td colspan="3" style="color:#666;">No objects detected</td></tr>';
                        } else {
                            tbody.innerHTML = data.detections.map(d => `
                                <tr>
                                    <td>${d.class_name}</td>
                                    <td>${d.confidence}%</td>
                                    <td><span class="badge badge-${d.waste_category}">${d.waste_category}</span></td>
                                </tr>
                            `).join('');
                        }
                    } catch (e) {
                        console.error(e);
                    }
                }
                setInterval(updateStats, 500);
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    return app


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    weights = Path(args.weights)

    if not weights.exists():
        LOG.error("Model file not found: %s", weights)
        return 1

    LOG.info("Loading model: %s", weights)

    try:
        model = YOLO(str(weights))
        GLOBAL_STATE.model_loaded = True
        GLOBAL_STATE.model_name = weights.name
    except Exception as exc:
        LOG.error("Could not load YOLO model: %s", exc)
        return 1

    try:
        class_names = load_class_names(model, args.data)
    except Exception as exc:
        LOG.error("Could not read class names: %s", exc)
        return 1

    LOG.info("Model class count: %d", len(class_names))
    if len(class_names) <= 10:
        LOG.info("Classes: %s", class_names)

    rules = WasteRuleEngine(args.mapping)

    source = parse_source(args.source)

    try:
        capture = LatestFrameCapture(source)
    except Exception as exc:
        LOG.error("Camera initialization error: %s", exc)
        return 1

    mqtt_publisher = None

    if args.mqtt_broker:
        try:
            mqtt_publisher = MQTTPublisher(
                broker=args.mqtt_broker,
                port=args.mqtt_port,
                topic=args.mqtt_topic,
                username=args.mqtt_username,
                password=args.mqtt_password,
            )
        except Exception as exc:
            LOG.warning("MQTT publishing disabled: %s", exc)
    else:
        LOG.info("MQTT disabled")

    json_file = None
    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_file = json_path.open("a", encoding="utf-8")
        LOG.info("JSONL logging enabled: %s", json_path)

    if args.web:
        if not HAS_FASTAPI:
            LOG.error("FastAPI and uvicorn are required for --web mode.")
            return 1

        app = create_fastapi_app(model, class_names, rules, args)

        server_thread = threading.Thread(
            target=lambda: uvicorn.run(
                app, host=args.host, port=args.port, log_level="warning"
            ),
            daemon=True,
        )
        server_thread.start()
        LOG.info("Web dashboard running at http://%s:%d", args.host, args.port)

    writer = None
    frame_id = 0
    processed_frames = 0
    total_detections = 0
    class_counter: Counter[str] = Counter()
    fps_meter = FPSMeter()

    LOG.info("Detector loop started")

    try:
        while not capture.finished or not capture.queue.empty():
            frame = capture.read(timeout=0.5)

            if frame is None:
                with GLOBAL_STATE.lock:
                    GLOBAL_STATE.camera_status = (
                        "Connected" if capture.connected else "Disconnected"
                    )
                if capture.finished:
                    break
                continue

            frame_id += 1
            inference_start = time.perf_counter()

            try:
                results = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=args.device,
                    half=args.half,
                    verbose=False,
                )
            except Exception as exc:
                LOG.error("Inference error on frame %d: %s", frame_id, exc)
                continue

            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            if not results:
                detections = []
                annotated = frame
            else:
                result = results[0]

                detections = extract_detections(
                    result,
                    class_names,
                    rules,
                )

                annotated = draw_results(
                    frame,
                    result,
                    detections,
                    show_masks=not args.no_masks,
                )

            processed_frames += 1
            total_detections += len(detections)

            for item in detections:
                class_counter[item.class_name] += 1

            fps = fps_meter.update()

            with GLOBAL_STATE.lock:
                GLOBAL_STATE.latest_annotated_frame = annotated
                GLOBAL_STATE.latest_detections = detections
                GLOBAL_STATE.fps = fps
                GLOBAL_STATE.frame_id = frame_id
                GLOBAL_STATE.camera_status = (
                    "Connected" if capture.connected else "Disconnected"
                )

            # Draw top overlay panel
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}  Inference: {inference_ms:.1f} ms",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated,
                f"Frame: {frame_id}  Detections: {len(detections)}",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            payload = make_payload(frame_id, source, detections)

            if json_file:
                json_file.write(
                    json.dumps(payload, ensure_ascii=False) + "\n"
                )
                json_file.flush()

            if mqtt_publisher and (detections or args.publish_empty):
                try:
                    mqtt_publisher.publish(payload)
                except Exception as exc:
                    LOG.warning("MQTT publish error: %s", exc)

            if args.output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = make_writer(
                        args.output,
                        fps=max(fps, 25.0),
                        width=w,
                        height=h,
                    )

                writer.write(annotated)

            if args.show:
                cv2.imshow("Medical Waste Detector", annotated)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    LOG.info("User pressed exit key")
                    break

    except KeyboardInterrupt:
        LOG.info("Detection interrupted by user")

    finally:
        capture.stop()

        if writer:
            writer.release()

        if json_file:
            json_file.close()

        if mqtt_publisher:
            mqtt_publisher.close()

        if args.show:
            cv2.destroyAllWindows()

    LOG.info("Processed frames: %d", processed_frames)
    LOG.info("Total detections: %d", total_detections)

    if class_counter:
        LOG.info("Detections by class: %s", dict(class_counter))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
