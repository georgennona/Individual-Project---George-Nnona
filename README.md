# CM32017: Individual Project - Investigating Acoustic Finger-Tap Classification for TapGazer

This repository contains the code developed for my dissertation: *Investigating Acoustic Sensing as an Alternative Modality for VR Text-Entry Input*.

The project investigates whether acoustic sensing using a standard microphone is sufficient to classify the sounds of finger taps on a desk. This is to assess its viability as a tap-detection modality for TapGazer, a VR text-entry system.

Two machine learning classifiers (Random Forest and CNN) are developed and evaluated against TapXR, a commercial wrist-worn inertial sensing device.

---

## Project Structure

`record_session.py`             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Records continuous audio during a Unity session
<br> `extract_clips.py`         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Extracts labelled tap clips from recorded sessions
<br> `live_tap_recorder.py`     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Initial approach: live tap recording per finger class
<br> `train_model.py`           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Trains the Random Forest classifier
<br> `train_cnn.py`             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Trains the CNN classifier
<br> `live_classifier.py`       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Live RF + CNN inference during evaluation sessions
<br> `tap_test.py`              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# TapXR Bluetooth integration and UDP relay to Unity
<br> `evaluate.py`              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Scores evaluation sessions against Unity trial log CSV

The dataset - `dataset(old)` - from the initial recording approach is included as well.

Unity scripts (C#) for the data collection and evaluation scenes are included in the `Taptest/` folder. Also included is `HandDisplayScene` which was used to test the TapXR systems when running on both hands simultaneously.

---

## Pipeline Overview

1. **Data collection:** A Unity application prompts finger taps at fixed intervals. Audio is recorded simultaneously using `record_session.py`. A synchronisation tap at the start of each session aligns the Unity trial log with the audio recording.

2. **Clip extraction:** `extract_clips.py` detects the sync tap, locates individual tap peaks using a search window, and saves labelled 200ms WAV clips to a structured dataset directory.

3. **Model training:** `train_model.py` trains a Random Forest on hand-crafted spectral features. `train_cnn.py` trains a CNN on log-mel spectrograms using PyTorch.

4. **Live evaluation:** `live_classifier.py` listens to the microphone, detects taps using adaptive thresholding, and runs both models simultaneously, logging timestamped predictions to CSV. `tap_test.py` decodes TapXR finger events via the Tap SDK and relays them to Unity over UDP. `evaluate.py` matches predictions to Unity trial log CSVs using a response window and computes accuracy, detection rate, and latency.

---

## Setup

### Requirements

- Python 3.12
- PyTorch (CUDA recommended):
  ```
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```
- Other dependencies:
  ```
  pip install librosa sounddevice soundfile scipy scikit-learn numpy joblib matplotlib seaborn
  ```
- For TapXR integration: [Tap Python SDK](https://github.com/TapWithUs/tap-python-sdk)
  (install in a separate virtual environment — see note below)

### Configuration

Before running any script, update the file paths at the top of each script to match your local directory structure. Key paths to update:

- `DATASET_PATH` — root folder for processed tap clips (for `evaluate.py`, there is one for each of TapXR and acoustic sensing)
- `MODEL_DIR` — folder for saved model files
- `OUTPUT_DIR` — folder for evaluation session CSVs
- `INPUT_DEVICE` — microphone device index (run `python -m sounddevice` to list available devices)

### Running the pipeline

**Data collection:**
```bash
python record_session.py   # start audio recording first
# then start Unity session and press Space to begin
```

**Clip extraction:**
```bash
python extract_clips.py
```

**Training:**
```bash
python train_model.py   # Random Forest
python train_cnn.py     # CNN
```

**Live evaluation:**
```bash
# Terminal 1 (acoustic):
python live_classifier.py

# Terminal 2 (TapXR, separate venv):
python tap_test.py

# Then start Unity evaluation scene and press Space.
```
**Note:** They were designed to run simultaneously but due to the considerable difference in optimal tapping styles for each system, trials were conducted separately. Make sure to delete the tapxr eval CSV when you run only the live classifier.


**Scoring:**
```bash
python evaluate.py
```

### TapXR note

`tap_test.py` requires the Tap Python SDK and should be run in a separate virtual environment from the rest of the project, as the SDK has its own dependency requirements.

---

## Attribution

- **Tap Python SDK** — [TapWithUs/tap-python-sdk](https://github.com/TapWithUs/tap-python-sdk), used for TapXR Bluetooth integration in `tap_test.py`
- **Hand UI asset** — PNG hand image sourced from [CleanPNG](https://www.cleanpng.com/png-scalable-vector-graphics-computer-icons-encapsulat-7233344/download-png.html), used in the Unity evaluation and data collection interfaces
- **AI assistance** — Claude Sonnet 4.6 (Anthropic, https://claude.ai) was used as an assistive tool to support code development and dissertation writing. I acknowledge that this work is my own.

---

## Author

George Nnona

BSc (Hons) Computer Science, University of Bath, 2025–2026

**Supervisor:** Christof Lutteroth
