import os
import csv
from datetime import datetime
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter


DATASET_ROOT = r"C:/Users/georg/TapDataset"
RAW_DIR = os.path.join(DATASET_ROOT, "raw_sessions")
PROCESSED_DIR = os.path.join(DATASET_ROOT, "processed")

SAMPLE_RATE = 16000

PRE_TIME = 0.05
POST_TIME = 0.25
MAX_PAIR_DIFF_SEC = 60

SYNC_SEARCH_START_SEC = 0.5
SYNC_SEARCH_END_SEC = 30.0

REACTION_SEARCH_START_SEC = -0.5
REACTION_SEARCH_END_SEC = 1.0
PEAK_DETECTION_THRESHOLD = 0.003
CLIP_SAVE_THRESHOLD = 0.005

HP_CUTOFF = 120.0

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
def make_highpass(cutoff_hz: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="highpass")
    return b, a


def parse_timestamp_from_name(name: str):
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = stem.replace("_trials", "")
    parts = stem.split("_")
    if len(parts) < 3:
        return None

    # filename format: prefix_YYYYMMDD_HHMMSS
    ts = f"{parts[-2]}_{parts[-1]}"
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def pair_wavs_and_csvs(raw_dir: str):
    files = os.listdir(raw_dir)
    wavs = [f for f in files if f.lower().endswith(".wav")]
    csvs = [f for f in files if f.lower().endswith("_trials.csv")]

    csv_times = []
    for csv_name in csvs:
        dt = parse_timestamp_from_name(csv_name)
        if dt is not None:
            csv_times.append((csv_name, dt))

    pairs = []
    used_csvs = set()

    for wav_name in sorted(wavs):
        wav_dt = parse_timestamp_from_name(wav_name)
        if wav_dt is None:
            print(f"Skipping WAV with unparseable timestamp: {wav_name}")
            continue

        best_csv = None
        best_diff = None

        for csv_name, csv_dt in csv_times:
            if csv_name in used_csvs:
                continue

            diff = abs((csv_dt - wav_dt).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_csv = csv_name

        if best_csv is None or best_diff is None or best_diff > MAX_PAIR_DIFF_SEC:
            print(f"No suitable CSV match for {wav_name}")
            continue

        used_csvs.add(best_csv)
        pairs.append((wav_name, best_csv, best_diff))

    return pairs


def detect_sync_time(audio: np.ndarray, sr: int) -> float:
    if audio.ndim > 1:
        audio = audio[:, 0]

    start_idx = int(SYNC_SEARCH_START_SEC * sr)
    end_idx = min(len(audio), int(SYNC_SEARCH_END_SEC * sr))

    if end_idx <= start_idx:
        raise ValueError("Invalid sync search window.")

    search_audio = audio[start_idx:end_idx]

    b, a = make_highpass(HP_CUTOFF, sr)
    filtered = lfilter(b, a, search_audio)

    sync_rel_idx = int(np.argmax(np.abs(filtered)))
    sync_idx = start_idx + sync_rel_idx
    sync_time_sec = sync_idx / sr

    print(f"  Detected sync tap at {sync_time_sec:.3f}s")
    return sync_time_sec


def detect_tap_peak_time(audio: np.ndarray, sr: int, estimated_time_sec: float) -> float | None:
    if audio.ndim > 1:
        audio = audio[:, 0]

    start_sec = estimated_time_sec + REACTION_SEARCH_START_SEC
    end_sec = estimated_time_sec + REACTION_SEARCH_END_SEC

    start_idx = max(0, int(start_sec * sr))
    end_idx = min(len(audio), int(end_sec * sr))

    if end_idx <= start_idx:
        return None

    search_audio = audio[start_idx:end_idx]

    b, a = make_highpass(HP_CUTOFF, sr)
    filtered = lfilter(b, a, search_audio)

    if len(filtered) == 0:
        return None

    peak_rel_idx = int(np.argmax(np.abs(filtered)))
    peak_value = float(np.max(np.abs(filtered)))

    if peak_value < PEAK_DETECTION_THRESHOLD:
        return None

    peak_idx = start_idx + peak_rel_idx
    peak_time_sec = peak_idx / sr
    return peak_time_sec


def save_clip(clip: np.ndarray, label: str, session_id: str):
    energy = float(np.max(np.abs(clip)))
    if energy < CLIP_SAVE_THRESHOLD:
        return False

    label_dir = os.path.join(PROCESSED_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    # zero-padded index to keep filenames sortable
    existing = len([f for f in os.listdir(label_dir) if f.lower().endswith(".wav")])
    filename = f"{label}_{session_id}_{existing:05d}.wav"  # session ID added
    path = os.path.join(label_dir, filename)

    # normalize to 0.95 (avoid clipping)
    peak = np.max(np.abs(clip)) + 1e-12
    clip = 0.95 * clip / peak if peak > 1e-6 else clip

    sf.write(path, clip.astype(np.float32), SAMPLE_RATE)
    return True


def extract_session(wav_path: str, csv_path: str):
    session_id = os.path.splitext(os.path.basename(wav_path))[0]
    print(f"\nProcessing:")
    print(f"  WAV: {os.path.basename(wav_path)}")
    print(f"  CSV: {os.path.basename(csv_path)}")

    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate {SAMPLE_RATE}, got {sr}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    sync_rows = [r for r in rows if r["target_finger"] == "sync"]
    if not sync_rows:
        raise ValueError(f"No sync row found in {csv_path}")

    sync_time_sec = detect_sync_time(audio, sr)

    saved = 0
    no_peak = 0
    too_short = 0
    low_energy = 0

    for row in rows:
        label = row["target_finger"]
        if label == "sync":
            continue

        prompt_time_sec = float(row["prompt_time_sec"])
        estimated_time_sec = sync_time_sec + prompt_time_sec

        peak_time_sec = detect_tap_peak_time(audio, sr, estimated_time_sec)
        if peak_time_sec is None:
            no_peak += 1
            continue

        center = int(peak_time_sec * sr)
        start = max(0, int(center - PRE_TIME * sr))
        end = min(len(audio), int(center + POST_TIME * sr))

        clip = audio[start:end]

        min_len = int((PRE_TIME + POST_TIME) * sr * 0.8)
        if len(clip) < min_len:
            too_short += 1
            continue

        if save_clip(clip, label, session_id):
            saved += 1
        else:
            low_energy += 1

    print(f"  Saved clips: {saved}")
    print(f"  No peak found: {no_peak}")
    print(f"  Too short: {too_short}")
    print(f"  Low energy: {low_energy}")


def main():
    pairs = pair_wavs_and_csvs(RAW_DIR)

    if not pairs:
        print("No WAV/CSV pairs found.")
        return

    print("Pairs found:")
    for wav_name, csv_name, diff in pairs:
        print(f"  {wav_name}  <->  {csv_name}  (diff={diff:.1f}s)")

    for wav_name, csv_name, _ in pairs:
        extract_session(
            os.path.join(RAW_DIR, wav_name),
            os.path.join(RAW_DIR, csv_name),
        )


if __name__ == "__main__":
    main()