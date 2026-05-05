"""
One-command wake word + webcam + YOLO + TTS demo.

Run with:
    python main.py

What it does:
1. Listens continuously for a wake word with openWakeWord.
2. Also lets you press Enter as a manual fallback trigger.
3. Captures one webcam frame.
4. Runs YOLO object detection.
5. Prints and speaks the objects it sees.
6. Saves an annotated image as output.jpg.
7. Goes back to listening.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pyttsx3
import sounddevice as sd
from flask import Flask, Response, jsonify, render_template_string, send_file
from openwakeword import utils as wakeword_utils
from openwakeword.model import Model
from ultralytics import YOLO


# -----------------------------
# Easy demo settings to adjust
# -----------------------------

# Default: yolov8s balances accuracy vs speed better than yolov8n.
# Override without editing code: YOLO_MODEL=yolov8m.pt python main.py
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8s.pt")
CONFIDENCE_THRESHOLD = 0.40
CAPTURED_IMAGE_PATH = "captured.jpg"
OUTPUT_IMAGE_PATH = "output.jpg"
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000

MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1
MIC_BLOCK_SIZE = 1280  # 80 ms of audio at 16 kHz; good size for openWakeWord.
WAKE_THRESHOLD = 0.50
WAKE_COOLDOWN_SECONDS = 4.0
LISTENING_RESUME_DELAY_SECONDS = 2.5


# Shared UI state for the local webpage. A lock keeps background threads from
# reading and writing this dictionary at the same time.
UI_LOCK = threading.Lock()
UI_STATE = {
    "status": "Starting up",
    "last_trigger": "Never",
    "last_wake_word": "None yet",
    "detected_objects": [],
    "object_summary": "Nothing detected yet",
    "spoken_sentence": "Nothing spoken yet",
    "captured_image": CAPTURED_IMAGE_PATH,
    "output_image": OUTPUT_IMAGE_PATH,
}
WEB_TRIGGER_QUEUE: queue.Queue[str] | None = None


WEB_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ECE568 Desk Detection Demo</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Arial, Helvetica, sans-serif;
      background: #101418;
      color: #eef2f7;
    }
    body {
      margin: 0;
      padding: 24px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 30px;
    }
    .subtitle {
      margin: 0 0 24px;
      color: #aeb8c5;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }
    .card {
      background: #171d24;
      border: 1px solid #2a3441;
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
    }
    .card h2 {
      margin: 0 0 12px;
      font-size: 18px;
      color: #d6e4ff;
    }
    img {
      width: 100%;
      max-height: 420px;
      object-fit: contain;
      border-radius: 12px;
      background: #05070a;
      border: 1px solid #303a46;
    }
    .status {
      display: inline-block;
      padding: 8px 12px;
      border-radius: 999px;
      background: #153b2f;
      color: #8df7c2;
      font-weight: bold;
    }
    .sentence {
      font-size: 24px;
      line-height: 1.35;
      color: #ffffff;
    }
    .meta {
      color: #aeb8c5;
      line-height: 1.7;
    }
    ul {
      padding-left: 22px;
      line-height: 1.7;
    }
    button {
      cursor: pointer;
      border: 0;
      border-radius: 12px;
      padding: 12px 16px;
      color: white;
      background: #2f6df6;
      font-size: 16px;
      font-weight: bold;
    }
    button:hover {
      background: #2458c8;
    }
  </style>
</head>
<body>
  <h1>Wake Word Desk Object Detection Demo</h1>
  <p class="subtitle">Say the wake word, press Enter in the terminal, or click the manual trigger button.</p>

  <div class="grid">
    <section class="card">
      <h2>Current Camera</h2>
      <img src="/video_feed" alt="Live camera feed">
    </section>

    <section class="card">
      <h2>Latest YOLO Output</h2>
      <img id="output-image" src="/output_image?t=0" alt="YOLO output image">
    </section>

    <section class="card">
      <h2>Captured Image</h2>
      <img id="captured-image" src="/captured_image?t=0" alt="Captured image">
    </section>

    <section class="card">
      <h2>What It Sees</h2>
      <p><span id="status" class="status">Loading...</span></p>
      <p class="meta">
        Last trigger: <span id="last-trigger">Loading...</span><br>
        Wake word: <span id="last-wake-word">Loading...</span><br>
        Summary: <span id="object-summary">Loading...</span>
      </p>
      <ul id="detections"></ul>
      <button onclick="triggerDetection()">Trigger Detection</button>
    </section>

    <section class="card">
      <h2>What It Is Saying</h2>
      <p id="spoken-sentence" class="sentence">Loading...</p>
    </section>
  </div>

  <script>
    async function loadStatus() {
      const response = await fetch('/status');
      const data = await response.json();
      document.getElementById('status').textContent = data.status;
      document.getElementById('last-trigger').textContent = data.last_trigger;
      document.getElementById('last-wake-word').textContent = data.last_wake_word;
      document.getElementById('object-summary').textContent = data.object_summary;
      document.getElementById('spoken-sentence').textContent = data.spoken_sentence;

      const list = document.getElementById('detections');
      list.innerHTML = '';
      if (data.detected_objects.length === 0) {
        const item = document.createElement('li');
        item.textContent = 'No objects detected yet.';
        list.appendChild(item);
      } else {
        for (const detection of data.detected_objects) {
          const item = document.createElement('li');
          item.textContent = `${detection.name}: ${(detection.confidence * 100).toFixed(1)}%`;
          list.appendChild(item);
        }
      }

      const timestamp = Date.now();
      document.getElementById('captured-image').src = `/captured_image?t=${timestamp}`;
      document.getElementById('output-image').src = `/output_image?t=${timestamp}`;
    }

    async function triggerDetection() {
      await fetch('/trigger', { method: 'POST' });
      await loadStatus();
    }

    loadStatus();
    setInterval(loadStatus, 1000);
  </script>
</body>
</html>
"""


def update_ui_state(**updates: object) -> None:
    """Update the webpage state in a thread-safe way."""
    with UI_LOCK:
        UI_STATE.update(updates)


def get_ui_state() -> dict[str, object]:
    """Return a copy of the current webpage state."""
    with UI_LOCK:
        return dict(UI_STATE)


class WebcamManager:
    """Continuously reads the webcam so the UI and YOLO can share one camera."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """Start reading webcam frames in the background."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self) -> None:
        print("[Camera] Starting shared webcam feed...")
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("[Camera] ERROR: Could not open webcam for live UI feed.")
            update_ui_state(status="Camera unavailable")
            self.running = False
            return

        update_ui_state(status="Listening")
        try:
            while self.running:
                ok, frame = cap.read()
                if ok and frame is not None:
                    with self.lock:
                        self.frame = frame.copy()
                time.sleep(0.03)
        finally:
            cap.release()

    def get_frame(self) -> np.ndarray | None:
        """Return the latest camera frame."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()


WEBCAM = WebcamManager()
WEB_APP = Flask(__name__)


def make_placeholder_frame(message: str) -> np.ndarray:
    """Create a simple image for the webpage when no camera/image is ready yet."""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:] = (22, 29, 36)
    cv2.putText(
        frame,
        message,
        (40, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (230, 238, 247),
        2,
        cv2.LINE_AA,
    )
    return frame


def encode_frame_as_jpeg(frame: np.ndarray) -> bytes:
    """Encode an OpenCV frame as JPEG bytes for the local webpage."""
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        placeholder = make_placeholder_frame("Could not encode image")
        ok, buffer = cv2.imencode(".jpg", placeholder)
    return buffer.tobytes()


def generate_camera_stream() -> Iterable[bytes]:
    """Yield MJPEG camera frames for the browser."""
    while True:
        frame = WEBCAM.get_frame()
        if frame is None:
            frame = make_placeholder_frame("Waiting for camera...")

        jpg = encode_frame_as_jpeg(frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        time.sleep(0.08)


def serve_image_or_placeholder(path: str, message: str) -> Response:
    """Serve a saved image, or a generated placeholder if it does not exist yet."""
    image_path = Path(path)
    if image_path.exists():
        return send_file(image_path, mimetype="image/jpeg")

    frame = make_placeholder_frame(message)
    return Response(encode_frame_as_jpeg(frame), mimetype="image/jpeg")


@WEB_APP.route("/")
def index() -> str:
    """Local webpage dashboard."""
    return render_template_string(WEB_PAGE_HTML)


@WEB_APP.route("/video_feed")
def video_feed() -> Response:
    """Live current camera feed."""
    return Response(
        generate_camera_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@WEB_APP.route("/captured_image")
def captured_image() -> Response:
    """Last raw frame captured for YOLO."""
    return serve_image_or_placeholder(CAPTURED_IMAGE_PATH, "No captured image yet")


@WEB_APP.route("/output_image")
def output_image() -> Response:
    """Last image with YOLO bounding boxes."""
    return serve_image_or_placeholder(OUTPUT_IMAGE_PATH, "No YOLO output yet")


@WEB_APP.route("/status")
def status() -> Response:
    """JSON state used by the webpage."""
    return jsonify(get_ui_state())


@WEB_APP.route("/trigger", methods=["POST"])
def trigger_from_webpage() -> Response:
    """Manual trigger button on the webpage."""
    if WEB_TRIGGER_QUEUE is not None:
        WEB_TRIGGER_QUEUE.put("web")
        update_ui_state(status="Manual trigger requested")
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Trigger queue is not ready"}), 503


def start_web_ui(trigger_queue: queue.Queue[str]) -> None:
    """Start the local Flask webpage in a background thread."""
    global WEB_TRIGGER_QUEUE
    WEB_TRIGGER_QUEUE = trigger_queue
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    def run_server() -> None:
        WEB_APP.run(host=WEB_HOST, port=WEB_PORT, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f"[Web UI] Open http://{WEB_HOST}:{WEB_PORT} in your browser.")


def start_enter_key_listener(trigger_queue: queue.Queue[str]) -> None:
    """Start a background thread so pressing Enter can trigger detection."""

    def wait_for_enter() -> None:
        while True:
            try:
                input()
                trigger_queue.put("manual")
            except EOFError:
                # Some terminals do not allow input(). Wake word still works.
                return
            except Exception as exc:
                print(f"[Manual trigger] Enter key listener stopped: {exc}")
                return

    thread = threading.Thread(target=wait_for_enter, daemon=True)
    thread.start()


def initialize_yolo() -> YOLO:
    """Load the YOLO model once at startup."""
    print("[Startup] Loading YOLO model...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"[Startup] YOLO model ready: {YOLO_MODEL_PATH}")
        return model
    except Exception as exc:
        raise RuntimeError(
            "Could not load YOLO. Check your internet connection for the first "
            f"download of {YOLO_MODEL_PATH}, then try again."
        ) from exc


def initialize_tts() -> pyttsx3.Engine | None:
    """Load the offline text-to-speech engine once at startup."""
    print("[Startup] Starting offline TTS engine...")
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        print("[Startup] TTS engine ready")
        return engine
    except Exception as exc:
        print(f"[Startup] TTS failed to initialize: {exc}")
        print("[Startup] The demo will still print results, but it will not speak.")
        return None


def initialize_wake_word_model() -> Model:
    """Load openWakeWord once at startup."""
    print("[Startup] Loading openWakeWord model...")
    try:
        # Downloads the built-in openWakeWord models if they are not already present.
        try:
            wakeword_utils.download_models()
        except Exception as exc:
            print(f"[Startup] Could not download openWakeWord models: {exc}")
            print("[Startup] Continuing in case the models are already cached.")

        model = Model()
        print("[Startup] openWakeWord ready")
        print("[Startup] Using the default wake word models included with openWakeWord.")
        return model
    except Exception as exc:
        raise RuntimeError(
            "Could not load openWakeWord. Try reinstalling requirements or checking "
            "that openWakeWord model files downloaded correctly."
        ) from exc


def speak(engine: pyttsx3.Engine | None, sentence: str) -> None:
    """Speak the sentence out loud if TTS is available."""
    print(f"[TTS] {sentence}")
    if engine is None:
        return

    try:
        engine.say(sentence)
        engine.runAndWait()
    except Exception as exc:
        print(f"[TTS] Could not speak sentence: {exc}")


def capture_webcam_frame() -> np.ndarray | None:
    """Capture the latest frame from the shared webcam feed."""
    print("[Camera] Capturing current webcam frame...")

    # Give the camera a short moment to provide its first frame after startup.
    for _attempt in range(20):
        frame = WEBCAM.get_frame()
        if frame is not None:
            print("[Camera] Captured one frame")
            cv2.imwrite(CAPTURED_IMAGE_PATH, frame)
            return frame
        time.sleep(0.1)

    print("[Camera] ERROR: No webcam frame is available.")
    print("[Camera] Check camera permissions or try changing the camera index.")
    return None


def format_object_phrase(object_counts: Counter[str]) -> str:
    """Turn counts like {'bottle': 2, 'laptop': 1} into readable English."""
    phrases = []
    for name, count in sorted(object_counts.items()):
        if count == 1:
            phrases.append(f"1 {name}")
        elif name == "person":
            phrases.append(f"{count} people")
        else:
            phrases.append(f"{count} {name}s")

    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"


def make_spoken_sentence(object_counts: Counter[str]) -> str:
    """Create the sentence the computer says after detection."""
    if not object_counts:
        return "I don't see any recognizable objects on the desk."

    object_phrase = format_object_phrase(object_counts)
    return f"I see {object_phrase} on the desk."


def print_detection_table(detections: Iterable[tuple[str, float]]) -> None:
    """Print detected names and confidence scores for the demo audience."""
    detections = list(detections)
    if not detections:
        print("[YOLO] No objects above the confidence threshold.")
        return

    print("[YOLO] Detected objects:")
    for name, confidence in detections:
        print(f"  - {name}: {confidence:.2f}")


def run_object_detection(
    yolo_model: YOLO,
    frame: np.ndarray,
) -> tuple[Counter[str], np.ndarray, list[tuple[str, float]]]:
    """Run YOLO on one frame and return grouped object counts plus an annotated image."""
    print("[YOLO] Running object detection...")
    results = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    if not results:
        return Counter(), frame, []

    result = results[0]
    names = result.names
    detections: list[tuple[str, float]] = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        object_name = names[class_id]
        detections.append((object_name, confidence))

    print_detection_table(detections)
    object_counts = Counter(name for name, _confidence in detections)

    # Ultralytics draws bounding boxes and labels on a copy of the image.
    annotated_frame = result.plot()
    return object_counts, annotated_frame, detections


def run_detection_sequence(yolo_model: YOLO, tts_engine: pyttsx3.Engine | None) -> None:
    """Capture the desk, detect objects, speak the result, and save output.jpg."""
    update_ui_state(
        status="Detection running",
        last_trigger=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    print("\n==============================")
    print("Wake word detected")
    print("==============================")

    frame = capture_webcam_frame()
    if frame is None:
        sentence = "I could not access the webcam."
        update_ui_state(status="Camera error", spoken_sentence=sentence)
        speak(tts_engine, sentence)
        print("[Demo] Returning to wake word listening.\n")
        return

    try:
        object_counts, annotated_frame, detections = run_object_detection(yolo_model, frame)
    except Exception as exc:
        print(f"[YOLO] ERROR while running object detection: {exc}")
        sentence = "I could not run object detection."
        update_ui_state(status="YOLO error", spoken_sentence=sentence)
        speak(tts_engine, sentence)
        print("[Demo] Returning to wake word listening.\n")
        return

    if cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame):
        print(f"[Output] Saved image with bounding boxes to {OUTPUT_IMAGE_PATH}")
    else:
        print(f"[Output] ERROR: Could not save {OUTPUT_IMAGE_PATH}")

    sentence = make_spoken_sentence(object_counts)
    detection_rows = [
        {"name": name, "confidence": confidence}
        for name, confidence in detections
    ]
    update_ui_state(
        status="Speaking",
        detected_objects=detection_rows,
        object_summary=format_object_phrase(object_counts) or "No recognizable objects",
        spoken_sentence=sentence,
    )
    speak(tts_engine, sentence)
    print("[Demo] Returning to wake word listening.\n")


def get_detected_wake_word(prediction: dict[str, float]) -> str | None:
    """Return the wake word name if openWakeWord scores pass the threshold."""
    if not prediction:
        return None

    wake_word, score = max(prediction.items(), key=lambda item: item[1])
    if score >= WAKE_THRESHOLD:
        print(f"[WakeWord] Heard '{wake_word}' with score {score:.2f}")
        update_ui_state(last_wake_word=f"{wake_word} ({score:.2f})")
        return wake_word
    return None


def queue_has_manual_trigger(trigger_queue: queue.Queue[str]) -> bool:
    """Check whether Enter was pressed and clear any extra Enter presses."""
    manual_triggered = not trigger_queue.empty()
    if manual_triggered:
        trigger_sources = []
        while not trigger_queue.empty():
            trigger_sources.append(trigger_queue.get_nowait())
        if "web" in trigger_sources:
            print("[Manual] Web button pressed.")
        else:
            print("[Manual] Enter pressed.")
    return manual_triggered


def reset_listening_after_detection(wake_model: Model, stream: sd.InputStream) -> float:
    """Wait for speaker audio to fade, discard stale mic audio, and reset wake state."""
    update_ui_state(status="Pausing so TTS does not retrigger wake word")
    print(
        "[WakeWord] Pausing briefly so the microphone does not hear the "
        "computer's own voice."
    )
    time.sleep(LISTENING_RESUME_DELAY_SECONDS)

    discarded_blocks = 0
    try:
        # During YOLO + TTS, the microphone stream can accumulate old audio.
        # Throw it away so openWakeWord starts from fresh room audio.
        while stream.read_available >= MIC_BLOCK_SIZE:
            stream.read(MIC_BLOCK_SIZE)
            discarded_blocks += 1
    except Exception as exc:
        print(f"[Microphone] Could not flush old audio blocks: {exc}")

    wake_model.reset()
    if discarded_blocks:
        print(f"[Microphone] Discarded {discarded_blocks} old audio block(s).")
    print("[WakeWord] Reset wake word state.")
    update_ui_state(status="Listening")
    return time.time()


def listen_for_enter_only(
    yolo_model: YOLO,
    tts_engine: pyttsx3.Engine | None,
    trigger_queue: queue.Queue[str],
) -> None:
    """Fallback loop for demos when microphone input is unavailable."""
    print("\n[Fallback] Microphone listening is unavailable.")
    print("[Fallback] Press Enter to run detection manually, or Ctrl+C to quit.\n")

    try:
        while True:
            if queue_has_manual_trigger(trigger_queue):
                run_detection_sequence(yolo_model, tts_engine)
                print("[Fallback] Press Enter to run detection again.\n")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Exit] Demo stopped by user.")


def listen_forever(
    wake_model: Model,
    yolo_model: YOLO,
    tts_engine: pyttsx3.Engine | None,
    trigger_queue: queue.Queue[str],
) -> None:
    """Main loop: listen for wake word or Enter, then run one detection."""
    print("\n[Ready] Listening for wake word.")
    print("[Ready] Press Enter at any time to trigger detection manually.")
    print("[Ready] Press Ctrl+C to quit.\n")

    last_trigger_time = 0.0

    try:
        with sd.InputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=MIC_CHANNELS,
            dtype="int16",
            blocksize=MIC_BLOCK_SIZE,
        ) as stream:
            while True:
                manual_triggered = queue_has_manual_trigger(trigger_queue)

                audio_block, overflowed = stream.read(MIC_BLOCK_SIZE)
                if overflowed:
                    print("[Microphone] Warning: audio input overflowed.")

                audio = audio_block.reshape(-1)
                prediction = wake_model.predict(audio)
                wake_word = get_detected_wake_word(prediction)

                now = time.time()
                enough_time_since_last_trigger = now - last_trigger_time > WAKE_COOLDOWN_SECONDS

                if (manual_triggered or wake_word) and enough_time_since_last_trigger:
                    wake_model.reset()
                    run_detection_sequence(yolo_model, tts_engine)
                    last_trigger_time = reset_listening_after_detection(wake_model, stream)
                    print("[Ready] Listening again. Press Enter for manual trigger.\n")

    except KeyboardInterrupt:
        print("\n[Exit] Demo stopped by user.")
    except Exception as exc:
        print(f"[Microphone] ERROR: Could not listen to microphone: {exc}")
        print("[Microphone] Check microphone permissions and your default input device.")
        print("[Microphone] You can also try reinstalling sounddevice or PortAudio.")
        listen_for_enter_only(yolo_model, tts_engine, trigger_queue)


def main() -> None:
    print("==============================================")
    print(" Wake Word Desk Object Detection Demo")
    print("==============================================")
    print("This demo listens for openWakeWord, captures a webcam")
    print("image, detects objects with YOLO, and speaks the result.")
    print()

    trigger_queue: queue.Queue[str] = queue.Queue()
    start_enter_key_listener(trigger_queue)
    WEBCAM.start()
    start_web_ui(trigger_queue)

    try:
        yolo_model = initialize_yolo()
        tts_engine = initialize_tts()
        wake_model = initialize_wake_word_model()
    except RuntimeError as exc:
        print(f"[Startup] ERROR: {exc}")
        return

    listen_forever(wake_model, yolo_model, tts_engine, trigger_queue)


if __name__ == "__main__":
    main()
