import csv
import queue
import time
from collections import deque
from pathlib import Path
from datetime import datetime
import joblib
import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
from scipy.signal import butter, lfilter



SAMPLE_RATE     = 16000
BLOCK_SIZE      = 128
CHANNELS        = 1
INPUT_DEVICE    = None

HP_CUTOFF            = 120.0
THRESHOLD_MULTIPLIER = 5.0 # 3.0
MIN_TRIGGER_GAP_MS   = 300 # 150
NOISE_FLOOR_ALPHA    = 0.995

PRE_TRIGGER_MS        = 50
POST_TRIGGER_MS       = 250
ROLLING_BUFFER_SECONDS = 1.0

MODEL_DIR       = Path("C:/Users/georg/TapDataset/models")
RF_GLOB         = "tap_finger_model_2026-04-24_04-35-28 (all data).joblib"
CNN_GLOB        = "tap_finger_cnn_2026-04-24_04-46-52 (all data).pt"
OUTPUT_DIR      = Path("C:/Users/georg/TapDataset/eval_sessions")


# for CNN
N_MELS      = 64
N_FFT       = 256
HOP_LENGTH  = 64
TARGET_LEN_S  = 0.2
PEAK_OFFSET_S = 0.03

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class TapCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)



def find_newest(model_dir: Path, pattern: str) -> Path:
    candidates = sorted(model_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern} in {model_dir}")
    return candidates[-1]


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
def make_highpass(cutoff_hz: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="highpass")
    return b, a


def align_to_peak(y: np.ndarray, sr: int) -> np.ndarray:
    target_len  = int(TARGET_LEN_S * sr)
    peak_offset = int(PEAK_OFFSET_S * sr)

    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)

    peak_idx = int(np.argmax(np.abs(y)))
    start = max(0, peak_idx - peak_offset)
    end   = start + target_len

    out = y[start:end] if end <= len(y) else np.pad(y[start:], (0, end - len(y)))
    if len(out) < target_len:
        out = np.pad(out, (0, target_len - len(out)))

    return out.astype(np.float32)


def extract_rf_features(y: np.ndarray, sr: int) -> np.ndarray:
    y, _ = librosa.effects.trim(y, top_db=30)

    peak = np.max(np.abs(y)) + 1e-12
    y = y / peak
    y = align_to_peak(y, sr)

    split  = len(y) // 2
    y_early = y[:split]
    y_late  = y[split:]

    n_fft      = 256
    hop_length = 64

    def summarize(feat):
        return np.concatenate([
            np.mean(feat, axis=1),
            np.std(feat,  axis=1),
            np.max(feat,  axis=1),
        ])

    features = np.concatenate([
        summarize(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length, n_mels=26)),
        summarize(librosa.feature.zero_crossing_rate(y, hop_length=hop_length)),
        summarize(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)),
        summarize(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)),
        summarize(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)),
        summarize(librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)),
        summarize(librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)),

        summarize(librosa.feature.mfcc(y=y_early, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length, n_mels=26)),
        summarize(librosa.feature.spectral_centroid(y=y_early, sr=sr, n_fft=n_fft, hop_length=hop_length)),
        summarize(librosa.feature.spectral_bandwidth(y=y_early, sr=sr, n_fft=n_fft, hop_length=hop_length)),
        summarize(librosa.feature.spectral_rolloff(y=y_early, sr=sr, n_fft=n_fft, hop_length=hop_length)),
        summarize(librosa.feature.rms(y=y_early, frame_length=n_fft, hop_length=hop_length)),

        summarize(librosa.feature.mfcc(y=y_late, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length, n_mels=26)),
        summarize(librosa.feature.spectral_centroid(y=y_late, sr=sr, n_fft=n_fft, hop_length=hop_length)),
        summarize(librosa.feature.spectral_bandwidth(y=y_late, sr=sr, n_fft=n_fft, hop_length=hop_length)),
        summarize(librosa.feature.spectral_rolloff(y=y_late, sr=sr, n_fft=n_fft, hop_length=hop_length)),
        summarize(librosa.feature.rms(y=y_late, frame_length=n_fft, hop_length=hop_length)),
    ])

    peak_amp = float(np.max(np.abs(y)))
    peak_idx = float(np.argmax(np.abs(y))) / len(y)

    return np.concatenate([features, [peak_amp, peak_idx]]).astype(np.float32)



def extract_cnn_spectrogram(y: np.ndarray, sr: int) -> torch.Tensor:
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < 128:
        pass

    peak = np.max(np.abs(y)) + 1e-12
    y = y / peak
    y = align_to_peak(y, sr)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mean, std = log_mel.mean(), log_mel.std() + 1e-6
    log_mel = (log_mel - mean) / std

    tensor = torch.tensor(log_mel[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
    return tensor.to(DEVICE)




class LiveClassifier:
    def __init__(self):
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.b, self.a = make_highpass(HP_CUTOFF, SAMPLE_RATE)

        self.noise_floor      = 1e-6
        self.last_trigger_time = 0.0

        self.pre_samples  = int(PRE_TRIGGER_MS * SAMPLE_RATE / 1000)
        self.post_samples = int(POST_TRIGGER_MS * SAMPLE_RATE / 1000)

        self.raw_buffer = deque(maxlen=int(ROLLING_BUFFER_SECONDS * SAMPLE_RATE))

        self.capture_armed     = False
        self.capture_pre_audio = None
        self.capture_post_audio = []

        # load RF
        rf_path = find_newest(MODEL_DIR, RF_GLOB)
        bundle = joblib.load(rf_path)
        self.rf_model  = bundle["model"]
        self.rf_labels = bundle.get("labels", None)
        print(f"RF loaded:  {rf_path.name}")

        # load CNN
        cnn_path = find_newest(MODEL_DIR, CNN_GLOB)
        checkpoint = torch.load(cnn_path, map_location=DEVICE)
        self.cnn_label_to_idx = checkpoint["label_to_idx"]
        self.cnn_idx_to_label = {v: k for k, v in self.cnn_label_to_idx.items()}
        self.cnn_model = TapCNN(num_classes=len(self.cnn_label_to_idx)).to(DEVICE)
        self.cnn_model.load_state_dict(checkpoint["model_state_dict"])
        self.cnn_model.eval()
        print(f"CNN loaded: {cnn_path.name}")

        # CSV output
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        rf_csv_path  = OUTPUT_DIR / f"eval_{session_id}_rf.csv"
        cnn_csv_path = OUTPUT_DIR / f"eval_{session_id}_cnn.csv"

        self.rf_csv  = open(rf_csv_path,  "w", newline="")
        self.cnn_csv = open(cnn_csv_path, "w", newline="")

        self.rf_writer  = csv.writer(self.rf_csv)
        self.cnn_writer = csv.writer(self.cnn_csv)

        self.rf_writer.writerow(["timestamp", "prediction", "confidence"])
        self.cnn_writer.writerow(["timestamp", "prediction", "confidence"])

        print(f"RF  logging to: {rf_csv_path}")
        print(f"CNN logging to: {cnn_csv_path}")

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata[:, 0].copy())

    def classify_and_log(self, clip: np.ndarray, trigger_time: float):
        # RF
        try:
            rf_features = extract_rf_features(clip, SAMPLE_RATE).reshape(1, -1)
            rf_pred = self.rf_model.predict(rf_features)[0]
            rf_probs = self.rf_model.predict_proba(rf_features)[0]
            rf_conf = float(np.max(rf_probs))
            self.rf_writer.writerow([f"{trigger_time:.6f}", rf_pred, f"{rf_conf:.4f}"])
            self.rf_csv.flush()
            print(f"  RF  → {rf_pred} ({rf_conf:.2f})")
        except Exception as e:
            print(f"  RF error: {e}")

        # CNN
        try:
            with torch.no_grad():
                spec   = extract_cnn_spectrogram(clip, SAMPLE_RATE)
                logits = self.cnn_model(spec)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
                idx    = int(np.argmax(probs))
                cnn_pred = self.cnn_idx_to_label[idx]
                cnn_conf = float(probs[idx])
                self.cnn_writer.writerow([f"{trigger_time:.6f}", cnn_pred, f"{cnn_conf:.4f}"])
                self.cnn_csv.flush()
                print(f"  CNN → {cnn_pred} ({cnn_conf:.2f})")
        except Exception as e:
            print(f"  CNN error: {e}")

    def finish_capture_if_ready(self):
        if not self.capture_armed:
            return

        post = np.concatenate(self.capture_post_audio) if self.capture_post_audio else np.array([], dtype=np.float32)
        if len(post) < self.post_samples:
            return

        clip = np.concatenate([self.capture_pre_audio, post[:self.post_samples]]).astype(np.float32)

        print(f"\nTAP detected at {self.capture_trigger_time:.3f}")
        self.classify_and_log(clip, self.capture_trigger_time)

        self.capture_armed      = False
        self.capture_pre_audio  = None
        self.capture_post_audio = []

    def process_block(self, block: np.ndarray):
        for x in block:
            self.raw_buffer.append(float(x))

        if self.capture_armed:
            self.capture_post_audio.append(block.copy())
            self.finish_capture_if_ready()

        filtered   = lfilter(self.b, self.a, block)
        block_rms  = float(np.sqrt(np.mean(filtered ** 2) + 1e-12))
        block_peak = float(np.max(np.abs(filtered))) + 1e-12

        now = time.time()
        ms_since_last = (now - self.last_trigger_time) * 1000.0

        # Refractory period
        if ms_since_last > MIN_TRIGGER_GAP_MS:
            self.noise_floor = (
                NOISE_FLOOR_ALPHA * self.noise_floor
                + (1.0 - NOISE_FLOOR_ALPHA) * block_rms
            )

        # Tap detection trigger
        rms_threshold  = max(self.noise_floor * THRESHOLD_MULTIPLIER, 2e-4) #5e-5
        peak_threshold = max(rms_threshold * 3.0, 8e-4) #2e-4

        is_trigger = (
            not self.capture_armed
            and ms_since_last > MIN_TRIGGER_GAP_MS
            and (block_rms > rms_threshold or block_peak > peak_threshold)
        )

        if not is_trigger:
            return

        self.last_trigger_time = now

        if len(self.raw_buffer) < self.pre_samples:
            return

        pre_audio = np.array(self.raw_buffer, dtype=np.float32)[-self.pre_samples:].copy()

        self.capture_armed        = True
        self.capture_pre_audio    = pre_audio
        self.capture_post_audio   = [block.copy()]
        self.capture_trigger_time = now

    def run(self):
        print(f"\nListening on device {INPUT_DEVICE}... Press Ctrl+C to stop.\n")
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=CHANNELS,
                dtype="float32",
                callback=self.audio_callback,
                device=INPUT_DEVICE,
            ):
                while True:
                    block = self.audio_queue.get()
                    self.process_block(block)
        finally:
            self.rf_csv.close()
            self.cnn_csv.close()
            print("CSVs saved.")


if __name__ == "__main__":
    classifier = LiveClassifier()
    classifier.run()