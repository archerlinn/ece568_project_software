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

import queue
import threading
import time
from collections import Counter
from typing import Iterable

import cv2
import numpy as np
import pyttsx3
import sounddevice as sd
from openwakeword import utils as wakeword_utils
from openwakeword.model import Model
from ultralytics import YOLO


# -----------------------------
# Easy demo settings to adjust
# -----------------------------

YOLO_MODEL_PATH = "yolov8n.pt"  # Small and fast. Downloads automatically the first time.
CONFIDENCE_THRESHOLD = 0.40
OUTPUT_IMAGE_PATH = "output.jpg"

MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1
MIC_BLOCK_SIZE = 1280  # 80 ms of audio at 16 kHz; good size for openWakeWord.
WAKE_THRESHOLD = 0.50
WAKE_COOLDOWN_SECONDS = 2.0


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
            "download of yolov8n.pt, then try again."
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


def capture_webcam_frame(camera_index: int = 0) -> np.ndarray | None:
    """Open the webcam, capture one frame, and close the webcam."""
    print("[Camera] Opening webcam...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[Camera] ERROR: Could not open webcam.")
        print("[Camera] Check camera permissions or try changing camera_index.")
        return None

    try:
        # Give the camera a short moment to adjust exposure.
        time.sleep(0.5)

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[Camera] ERROR: Webcam opened, but no frame was captured.")
            return None

        print("[Camera] Captured one frame")
        return frame
    except Exception as exc:
        print(f"[Camera] ERROR while capturing frame: {exc}")
        return None
    finally:
        cap.release()
        cv2.destroyAllWindows()


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


def run_object_detection(yolo_model: YOLO, frame: np.ndarray) -> tuple[Counter[str], np.ndarray]:
    """Run YOLO on one frame and return grouped object counts plus an annotated image."""
    print("[YOLO] Running object detection...")
    results = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    if not results:
        return Counter(), frame

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
    return object_counts, annotated_frame


def run_detection_sequence(yolo_model: YOLO, tts_engine: pyttsx3.Engine | None) -> None:
    """Capture the desk, detect objects, speak the result, and save output.jpg."""
    print("\n==============================")
    print("Wake word detected")
    print("==============================")

    frame = capture_webcam_frame()
    if frame is None:
        speak(tts_engine, "I could not access the webcam.")
        print("[Demo] Returning to wake word listening.\n")
        return

    try:
        object_counts, annotated_frame = run_object_detection(yolo_model, frame)
    except Exception as exc:
        print(f"[YOLO] ERROR while running object detection: {exc}")
        speak(tts_engine, "I could not run object detection.")
        print("[Demo] Returning to wake word listening.\n")
        return

    if cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame):
        print(f"[Output] Saved image with bounding boxes to {OUTPUT_IMAGE_PATH}")
    else:
        print(f"[Output] ERROR: Could not save {OUTPUT_IMAGE_PATH}")

    sentence = make_spoken_sentence(object_counts)
    speak(tts_engine, sentence)
    print("[Demo] Returning to wake word listening.\n")


def get_detected_wake_word(prediction: dict[str, float]) -> str | None:
    """Return the wake word name if openWakeWord scores pass the threshold."""
    if not prediction:
        return None

    wake_word, score = max(prediction.items(), key=lambda item: item[1])
    if score >= WAKE_THRESHOLD:
        print(f"[WakeWord] Heard '{wake_word}' with score {score:.2f}")
        return wake_word
    return None


def queue_has_manual_trigger(trigger_queue: queue.Queue[str]) -> bool:
    """Check whether Enter was pressed and clear any extra Enter presses."""
    manual_triggered = not trigger_queue.empty()
    if manual_triggered:
        while not trigger_queue.empty():
            trigger_queue.get_nowait()
        print("[Manual] Enter pressed.")
    return manual_triggered


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
                    last_trigger_time = now
                    run_detection_sequence(yolo_model, tts_engine)
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
