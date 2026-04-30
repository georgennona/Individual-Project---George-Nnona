import os
import json
from pathlib import Path
from datetime import datetime
import uuid
import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 16000
DATASET_PATH = r"C:/Users/georg/TapDataset/processed"
MODEL_DIR = Path("C:/Users/georg/TapDataset/models")
MODEL_BASENAME = "tap_finger_model"
EXCLUDE_DAYS = {} #{'20260415', '20260416', '20260417'} # remove days of choice


def align_to_peak(y: np.ndarray, sr: int, target_len_s: float = 0.2, peak_offset_s: float = 0.03) -> np.ndarray:
    target_len = int(target_len_s * sr)
    peak_offset = int(peak_offset_s * sr)

    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)

    peak_idx = int(np.argmax(np.abs(y)))
    start = max(0, peak_idx - peak_offset)
    end = start + target_len

    if end <= len(y):
        out = y[start:end]
    else:
        out = y[start:]
        out = np.pad(out, (0, end - len(y)))

    if len(out) < target_len:
        out = np.pad(out, (0, target_len - len(out)))

    return out.astype(np.float32)


def extract_features(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    y, _ = librosa.effects.trim(y, top_db=30)

    if len(y) < 128:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    peak = np.max(np.abs(y)) + 1e-12
    y = y / peak

    y = align_to_peak(y, sr, target_len_s=0.2, peak_offset_s=0.03)

    split = len(y) // 2
    y_early = y[:split]
    y_late = y[split:]

    def summarize(feature: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.mean(feature, axis=1),
            np.std(feature, axis=1),
            np.max(feature, axis=1),
        ])

    n_fft = 256
    hop_length = 64

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13,
        n_fft=n_fft, hop_length=hop_length, n_mels=26
    )
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop_length
    )
    rms = librosa.feature.rms(
        y=y, frame_length=n_fft, hop_length=hop_length
    )

    mfcc_early = librosa.feature.mfcc(
        y=y_early, sr=sr, n_mfcc=13,
        n_fft=n_fft, hop_length=hop_length, n_mels=26
    )
    centroid_early = librosa.feature.spectral_centroid(
        y=y_early, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    bandwidth_early = librosa.feature.spectral_bandwidth(
        y=y_early, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    rolloff_early = librosa.feature.spectral_rolloff(
        y=y_early, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    rms_early = librosa.feature.rms(
        y=y_early, frame_length=n_fft, hop_length=hop_length
    )

    mfcc_late = librosa.feature.mfcc(
        y=y_late, sr=sr, n_mfcc=13,
        n_fft=n_fft, hop_length=hop_length, n_mels=26
    )
    centroid_late = librosa.feature.spectral_centroid(
        y=y_late, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    bandwidth_late = librosa.feature.spectral_bandwidth(
        y=y_late, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    rolloff_late = librosa.feature.spectral_rolloff(
        y=y_late, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    rms_late = librosa.feature.rms(
        y=y_late, frame_length=n_fft, hop_length=hop_length
    )

    features = np.concatenate([
        summarize(mfcc),
        summarize(zcr),
        summarize(centroid),
        summarize(bandwidth),
        summarize(rolloff),
        summarize(flatness),
        summarize(rms),

        summarize(mfcc_early),
        summarize(centroid_early),
        summarize(bandwidth_early),
        summarize(rolloff_early),
        summarize(rms_early),

        summarize(mfcc_late),
        summarize(centroid_late),
        summarize(bandwidth_late),
        summarize(rolloff_late),
        summarize(rms_late),
    ])

    peak_amp = np.max(np.abs(y))
    peak_idx = np.argmax(np.abs(y)) / len(y)

    features = np.concatenate([
        features,
        [peak_amp, peak_idx]
    ])

    return features.astype(np.float32)


def load_dataset(dataset_path: str):
    X = []
    y = []
    files = []

    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root.resolve()}")

    labels = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not labels:
        raise ValueError("No label folders found inside dataset/")

    for label in labels:
        label_dir = root / label
        wav_files = sorted(label_dir.glob("*.wav"))

        print(f"{label}: {len(wav_files)} files")

        for wav_path in wav_files:
            try:
                feat = extract_features(str(wav_path))
                X.append(feat)
                y.append(label)
                files.append(str(wav_path))
            except Exception as e:
                print(f"Skipping {wav_path}: {e}")

    if not X:
        raise ValueError("No valid WAV files were loaded.")

    return np.array(X), np.array(y), files


def save_model_bundle(model, accuracy, labels, X_shape, dataset_path, model_dir=MODEL_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"{MODEL_BASENAME}_{timestamp}"

    model_path = model_dir / f"{base_name}.joblib"
    meta_path = model_dir / f"{base_name}.json"

    bundle = {
        "model": model,
        "sample_rate": SAMPLE_RATE,
        "labels": list(labels),
        "feature_shape": X_shape[1],
    }

    joblib.dump(bundle, model_path)

    metadata = {
        "model_file": model_path.name,
        "saved_at": timestamp,
        "accuracy": float(accuracy),
        "labels": list(labels),
        "num_samples": int(X_shape[0]),
        "num_features": int(X_shape[1]),
        "dataset_path": str(Path(dataset_path).resolve()),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path, meta_path

def get_session_id(filepath: str) -> str:
    name = Path(filepath).stem
    parts = name.split("_")
    # filename looks something like thumb_session_20260421_000202_00001
    # session ID is the date_time portion: 20260421_000202
    date_idx = next((i for i, p in enumerate(parts) if len(p) == 8 and p.isdigit()), None)
    if date_idx is None:
        return "unknown"
    return f"{parts[date_idx]}_{parts[date_idx + 1]}"

def main():
    print("Loading dataset...")
    X, y, files = load_dataset(DATASET_PATH)

    print(f"\nLoaded {len(X)} total samples")
    print(f"Feature shape: {X.shape}")

    session_ids = np.array([get_session_id(f) for f in files])
    days = np.array([s[:8] for s in session_ids])

    # date exclusion (if necessary)
    keep_mask = np.array([d not in EXCLUDE_DAYS for d in days])
    X, y, files = X[keep_mask], y[keep_mask], [files[i] for i in range(len(files)) if keep_mask[i]]
    session_ids = session_ids[keep_mask]
    days = days[keep_mask]

    unique_days = sorted(set(days))
    print(f"\nDays found ({len(unique_days)}):")
    for d in unique_days:
        count = np.sum(days == d)
        print(f"  {d}: {count} clips")

    # FOR K-FOLD CROSS VALIDATION 

    # all_accuracies = []
    # all_cms = []
    # best_acc = -1
    # best_model = None

    # for held_out_day in unique_days:
    #     train_mask = days != held_out_day
    #     test_mask = days == held_out_day

    #     X_train, y_train = X[train_mask], y[train_mask]
    #     X_test, y_test = X[test_mask], y[test_mask]

    #     model = RandomForestClassifier(n_estimators=200, random_state=42)
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)

    #     acc = accuracy_score(y_test, y_pred)
    #     all_accuracies.append(acc)
    #     all_cms.append(confusion_matrix(y_test, y_pred))

    #     print(f"\nHeld out {held_out_day} ({np.sum(test_mask)} clips):")
    #     print(f"  Accuracy: {acc:.4f}")
    #     print(classification_report(y_test, y_pred))

    #     if acc > best_acc:
    #         best_acc = acc
    #         best_model = model

    # mean_acc = np.mean(all_accuracies)
    # std_acc = np.std(all_accuracies)

    # print("\n=== Cross-validation summary ===")
    # for d, a in zip(unique_days, all_accuracies):
    #     print(f"  {d}: {a:.4f}")
    # print(f"\nMean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # print("\nMean confusion matrix:")
    # mean_cm = np.round(np.mean(all_cms, axis=0)).astype(int)
    # print(mean_cm)

    # train on all data 
    print("\nTraining on all data...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    model_path, meta_path = save_model_bundle(
        model=model,
        accuracy=0.0,  # no eval accuracy since training on all
        labels=np.unique(y),
        X_shape=X.shape,
        dataset_path=DATASET_PATH,
    )

    print(f"\nSaved versioned model to: {model_path.resolve()}")
    print(f"Saved metadata to: {meta_path.resolve()}")


if __name__ == "__main__":
    main()