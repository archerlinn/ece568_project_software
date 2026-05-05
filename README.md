# ece568_project_software

# Wake Word Desk Object Detection Demo

This is a simple one-script Python demo for a class final project.

Run one command, leave the program listening, and trigger object detection with a wake word or by pressing Enter. When triggered, the program opens the webcam, captures one image of the desk, runs YOLO object detection, saves `output.jpg`, and speaks the detected objects out loud.

## What It Uses

- `openwakeword` for always-on wake word detection
- `sounddevice` for microphone input
- `opencv-python` for webcam capture
- `ultralytics` YOLO with `yolov8n.pt` for object detection
- `pyttsx3` for offline text-to-speech

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

The first run may take longer because YOLO downloads `yolov8n.pt`.

## Demo Flow

1. Start the program with `python main.py`.
2. Wait for the message saying it is listening.
3. Say one of the default openWakeWord wake phrases, or press Enter as a manual fallback.
4. The program prints `Wake word detected`.
5. The webcam turns on and captures one frame.
6. YOLO detects objects in the image.
7. The program prints object names and confidence scores.
8. The program speaks a sentence such as:

```text
I see 1 laptop, 1 book, and 2 bottles on the desk.
```

9. The annotated detection image is saved as `output.jpg`.
10. The program returns to listening.

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

### Webcam Does Not Open

- Check camera permissions in the operating system.
- Close other apps that may be using the webcam.
- Try a different USB port if using an external webcam.
- In `main.py`, change `capture_webcam_frame(camera_index=0)` to use camera index `1` if the wrong camera is selected.

### YOLO Model Download Fails

- The first run downloads `yolov8n.pt`.
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
- `output.jpg`: generated after a detection run
