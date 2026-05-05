# ece568_project_software

# Wake Word Desk Object Detection Demo

This is a simple one-script Python demo for a class final project.

Run one command, leave the program listening, and trigger object detection with a wake word or by pressing Enter. When triggered, the program opens the webcam, captures one image of the desk, runs YOLO object detection, saves `output.jpg`, and speaks the detected objects out loud.

## What It Uses

- `openwakeword` for always-on wake word detection
- `sounddevice` for microphone input
- `opencv-python` for webcam capture
- `ultralytics` YOLO with `yolov8s.pt` for object detection (better accuracy than `yolov8n.pt`)
- `pyttsx3` for offline text-to-speech
- `flask` for the local demo webpage

## Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the demo:

```bash
python main.py
```

The first run may take longer because YOLO downloads `yolov8s.pt`.

For even better accuracy (slower), pick a larger checkpoint without editing code:

```bash
YOLO_MODEL=yolov8m.pt python main.py
```

For maximum accuracy on a strong GPU (much slower):

```bash
YOLO_MODEL=yolov8l.pt python main.py
```

Then open the local dashboard in a browser:

```text
http://127.0.0.1:5000
```

## Demo Flow

1. Start the program with `python main.py`.
2. Open `http://127.0.0.1:5000` to see the local dashboard.
3. Wait for the message saying it is listening.
4. Say one of the default openWakeWord wake phrases, press Enter as a manual fallback, or click the dashboard trigger button.
5. The program prints `Wake word detected`.
6. The webcam captures one frame.
7. YOLO detects objects in the image.
8. The program prints object names and confidence scores.
9. The program speaks a sentence such as:

```text
I see 1 laptop, 1 book, and 2 bottles on the desk.
```

10. The dashboard updates with the current camera, captured image, YOLO output, detected objects, and spoken sentence.
11. The annotated detection image is saved as `output.jpg`.
12. The program returns to listening.

## Local Web Dashboard

The dashboard is served locally from the same `main.py` process. It shows:

- the current live webcam feed
- the latest captured raw image
- the latest YOLO output image with bounding boxes
- detected object names and confidence scores
- the grouped object summary
- the sentence being spoken by TTS
- a manual trigger button for demos

The webpage only runs on your own computer at `127.0.0.1:5000`.

## Wake Word Note

This script uses the default wake word models available through `openwakeword`. If the wake word does not trigger reliably during a live demo, press Enter in the terminal to manually trigger the same detection sequence.

The manual Enter fallback is included on purpose so the class demo can continue even if the microphone, room noise, or wake word model causes problems.

## macOS Setup Notes

Install Python 3.10 or newer.

If microphone or camera access fails:

1. Open System Settings.
2. Go to Privacy & Security.
3. Allow microphone access for your terminal app.
4. Allow camera access for your terminal app.
5. Restart the terminal and run the program again.

If `sounddevice` has trouble, install PortAudio with Homebrew:

```bash
brew install portaudio
pip install sounddevice
```

For text-to-speech, `pyttsx3` usually works with built-in macOS speech voices.

## Windows Setup Notes

Install Python 3.10 or newer from the official Python website.

If microphone or camera access fails:

1. Open Settings.
2. Go to Privacy & security.
3. Enable Microphone access.
4. Enable Camera access.
5. Make sure desktop apps are allowed to use the microphone and camera.

If audio input does not work, check that Windows has a default input device selected.

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the virtual environment again.

## Linux Setup Notes

Install Python 3.10 or newer.

On Ubuntu or Debian, these system packages are often helpful:

```bash
sudo apt update
sudo apt install python3-venv portaudio19-dev espeak ffmpeg libsm6 libxext6
```

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

If the webcam fails, make sure your user has permission to access video devices. You may need to log out and back in after adding yourself to the `video` group:

```bash
sudo usermod -aG video "$USER"
```

For text-to-speech on Linux, `pyttsx3` commonly uses `espeak`. If TTS is silent, install it:

```bash
sudo apt install espeak
```

## Troubleshooting

### Microphone Does Not Work

- Make sure the operating system allows microphone access for the terminal.
- Make sure the correct microphone is selected as the default input device.
- Try plugging in a USB microphone or headset.
- On Linux, install PortAudio:

```bash
sudo apt install portaudio19-dev
```

- Reinstall `sounddevice`:

```bash
pip install --upgrade --force-reinstall sounddevice
```

### Wake Word Does Not Trigger

- Speak clearly and close to the microphone.
- Reduce background noise.
- Wait for the program to print that it is listening.
- Press Enter in the terminal to manually trigger detection during the demo.
- Click the `Trigger Detection` button on the local dashboard.

### Webcam Does Not Open

- Check camera permissions in the operating system.
- Close other apps that may be using the webcam.
- Try a different USB port if using an external webcam.
- In `main.py`, change `WEBCAM = WebcamManager()` to `WEBCAM = WebcamManager(camera_index=1)` if the wrong camera is selected.

### YOLO Model Download Fails

- The first run downloads the chosen `.pt` weight file (default `yolov8s.pt`).
- Make sure the computer has internet access for the first run.
- After the model is downloaded, future runs can use the local file.

### TTS Does Not Speak

- The program still prints the sentence even if TTS fails.
- On Linux, install `espeak`.
- On Windows and macOS, make sure system audio output is not muted.

## Files

- `main.py`: the complete one-file demo
- `requirements.txt`: Python dependencies
- `README.md`: setup and troubleshooting instructions
- `captured.jpg`: generated after a detection run
- `output.jpg`: generated after a detection run
