import json
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SAMPLE_RATE = 16000
DATASET_PATH = r"C:/Users/georg/TapDataset/processed"
MODEL_DIR = Path("C:/Users/georg/TapDataset/models")
MODEL_BASENAME = "tap_finger_cnn"

EXCLUDE_DAYS = {} #{"20260415", "20260416", "20260417"} # remove days of choice

N_MELS = 64 #80
N_FFT = 256 #512 
HOP_LENGTH = 64 #32
TARGET_LEN_S = 0.2
PEAK_OFFSET_S = 0.03

BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
PATIENCE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### preprocessing

def align_to_peak(y: np.ndarray, sr: int) -> np.ndarray:
    target_len = int(TARGET_LEN_S * sr)
    peak_offset = int(PEAK_OFFSET_S * sr)

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


def compute_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    # Returns a (1, N_MELS, T) float32 array, normalised per clip.
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    mean = log_mel.mean()
    std = log_mel.std() + 1e-6
    log_mel = (log_mel - mean) / std

    return log_mel[np.newaxis, :, :]


def load_clip(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < 128:
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    peak = np.max(np.abs(y)) + 1e-12
    y = y / peak
    y = align_to_peak(y, sr)
    return y


### dataset 

class TapDataset(Dataset):
    def __init__(self, file_paths: list, labels: list, label_to_idx: dict, augment: bool = False):
        self.file_paths = file_paths
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.augment = augment

        print(f"  Preloading {len(file_paths)} clips...", flush=True)
        self.waveforms = []
        for fp in file_paths:
            self.waveforms.append(load_clip(fp))

    def __len__(self):
        return len(self.file_paths)

    def augment_waveform(self, y: np.ndarray) -> np.ndarray:
        # time shift up to 20ms
        shift = np.random.randint(-int(0.02 * SAMPLE_RATE), int(0.02 * SAMPLE_RATE))
        y = np.roll(y, shift)

        # stronger noise
        noise_level = np.random.uniform(0.0, 0.015)
        y = y + noise_level * np.random.randn(*y.shape).astype(np.float32)

        # random amplitude scaling
        y = y * np.random.uniform(0.7, 1.3)

        return y

    def __getitem__(self, idx):
        y = self.waveforms[idx].copy()

        if self.augment:
            y = self.augment_waveform(y)
        spec = compute_spectrogram(y, SAMPLE_RATE)

        label_idx = self.label_to_idx[self.labels[idx]]

        return torch.tensor(spec, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)


### model 

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
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
            nn.Dropout(DROPOUT),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


### training 

def train_epoch(model, loader, criterion, optimiser):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for specs, labels in loader:
        specs, labels = specs.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for specs, labels in loader:
        specs, labels = specs.to(DEVICE), labels.to(DEVICE)
        logits = model(specs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    for specs, labels in loader:
        specs = specs.to(DEVICE)
        preds = model(specs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def train_fold(X_train_paths, y_train, X_test_paths, y_test, label_to_idx, idx_to_label):
    num_classes = len(label_to_idx)

    train_dataset = TapDataset(X_train_paths, y_train, label_to_idx, augment=True)
    test_dataset = TapDataset(X_test_paths, y_test, label_to_idx, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TapCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimiser)
        val_loss, val_acc = eval_epoch(model, test_loader, criterion)
        scheduler.step()

        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, test_loader


### data loading 

def get_session_id(filepath: str) -> str:
    name = Path(filepath).stem
    parts = name.split("_")
    date_idx = next((i for i, p in enumerate(parts) if len(p) == 8 and p.isdigit()), None)
    if date_idx is None:
        return "unknown"
    return f"{parts[date_idx]}_{parts[date_idx + 1]}"


def load_file_list(dataset_path: str):
    """Returns parallel lists of file paths and string labels."""
    files = []
    labels = []
    root = Path(dataset_path)

    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root.resolve()}")

    label_dirs = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not label_dirs:
        raise ValueError("No label folders found inside dataset/")

    for label in label_dirs:
        wav_files = sorted((root / label).glob("*.wav"))
        print(f"{label}: {len(wav_files)} files")
        for wp in wav_files:
            files.append(str(wp))
            labels.append(label)

    return files, labels


### save model

def save_model_bundle(model, accuracy, label_to_idx, dataset_path, model_dir=MODEL_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"{MODEL_BASENAME}_{timestamp}"

    model_path = model_dir / f"{base_name}.pt"
    meta_path = model_dir / f"{base_name}.json"

    torch.save({
        "model_state_dict": model.state_dict(),
        "label_to_idx": label_to_idx,
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
    }, model_path)

    metadata = {
        "model_file": model_path.name,
        "saved_at": timestamp,
        "accuracy": float(accuracy),
        "labels": list(label_to_idx.keys()),
        "dataset_path": str(Path(dataset_path).resolve()),
        "device": str(DEVICE),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path, meta_path


### RUN 

def main():
    print(f"Using device: {DEVICE}\n")
    print("Loading file list...")
    files, labels = load_file_list(DATASET_PATH)

    session_ids = np.array([get_session_id(f) for f in files])
    days = np.array([s[:8] for s in session_ids])

    # filter early sessions
    keep_mask = np.array([d not in EXCLUDE_DAYS for d in days])
    files = [files[i] for i in range(len(files)) if keep_mask[i]]
    labels = [labels[i] for i in range(len(labels)) if keep_mask[i]]
    days = days[keep_mask]

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}

    unique_days = sorted(set(days))
    print(f"\nDays found ({len(unique_days)}):")
    for d in unique_days:
        count = np.sum(days == d)
        print(f"  {d}: {count} clips")

    print(f"\nLabels: {unique_labels}")
    print(f"Total clips: {len(files)}")

    files = np.array(files)
    labels_arr = np.array(labels)

    # TO RUN WITH K-FOLD CROSS VALIDATION 
    
    # all_accuracies = []
    # all_cms = []
    # best_acc = -1.0
    # best_model = None

    # for held_out_day in unique_days:
    #     print(f"\n{'='*60}")
    #     print(f"Fold: held out {held_out_day}")
    #     print(f"{'='*60}")

    #     train_mask = days != held_out_day
    #     test_mask = days == held_out_day

    #     X_train = files[train_mask].tolist()
    #     y_train = labels_arr[train_mask].tolist()
    #     X_test = files[test_mask].tolist()
    #     y_test = labels_arr[test_mask].tolist()

    #     print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    #     model, test_loader = train_fold(X_train, y_train, X_test, y_test, label_to_idx, idx_to_label)

    #     true_labels, pred_labels = predict(model, test_loader)
    #     true_names = [idx_to_label[i] for i in true_labels]
    #     pred_names = [idx_to_label[i] for i in pred_labels]

    #     acc = accuracy_score(true_names, pred_names)
    #     all_accuracies.append(acc)
    #     all_cms.append(confusion_matrix(true_names, pred_names, labels=unique_labels))

    #     print(f"\nHeld out {held_out_day} ({len(X_test)} clips):")
    #     print(f"  Accuracy: {acc:.4f}")
    #     print(classification_report(true_names, pred_names, labels=unique_labels))

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
    dataset = TapDataset(files.tolist(), labels_arr.tolist(), label_to_idx, augment=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model     = TapCNN(num_classes=len(label_to_idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)

    for epoch in range(1, NUM_EPOCHS + 1):
        loss, acc = train_epoch(model, loader, criterion, optimiser)
        scheduler.step()
        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  loss={loss:.4f}  acc={acc:.4f}")

    model_path, meta_path = save_model_bundle(
        model=model,
        accuracy=0.0,  # no eval accuracy since training on all
        label_to_idx=label_to_idx,
        dataset_path=DATASET_PATH,
    )

    print(f"\nSaved model to: {model_path.resolve()}")
    print(f"Saved metadata to: {meta_path.resolve()}")


if __name__ == "__main__":
    main()