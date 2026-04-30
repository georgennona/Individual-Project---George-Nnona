import sys
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

EVAL_MODE = "tapxr" # acoustic or tapxr

ACOUSTIC_DIR = Path("C:/Users/georg/TapDataset/eval_sessions/acoustic")
TAPXR_DIR    = Path("C:/Users/georg/TapDataset/eval_sessions/tapxr")
RESPONSE_START = 0.05   # seconds after prompt to start looking for a prediction
RESPONSE_END   = 1.5   # seconds after prompt to stop looking
MOST_RECENT_ONLY = False

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def load_trials(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "trial_id":        int(row["trial_id"]),
                "target_finger":   row["target_finger"],
                "prompt_unix_time": float(row["prompt_unix_time"]),
            })
    return rows


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "timestamp":  float(row["timestamp"]),
                "prediction": row["finger"] if "finger" in row else row["prediction"],
                "confidence": float(row["confidence"]) if row.get("confidence") else None,
            })
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def match_predictions(trials: list[dict], predictions: list[dict]) -> list[dict]:
    """For each trial, find the first prediction within the response window."""
    results = []

    for trial in trials:
        t0 = trial["prompt_unix_time"] + RESPONSE_START
        t1 = trial["prompt_unix_time"] + RESPONSE_END

        match = None
        for pred in predictions:
            if t0 <= pred["timestamp"] <= t1:
                match = pred
                break

        results.append({
            "trial_id":      trial["trial_id"],
            "true":          trial["target_finger"],
            "pred":          match["prediction"] if match else None,
            "confidence":    match["confidence"] if match else None,
            "latency_ms":    (match["timestamp"] - trial["prompt_unix_time"]) * 1000 if match else None,
        })

    return results


def score(results: list[dict], system_name: str):
    total    = len(results)
    detected = [r for r in results if r["pred"] is not None]
    correct  = [r for r in detected if r["pred"] == r["true"]]
    missed   = [r for r in results if r["pred"] is None]

    overall_acc   = len(correct) / total if total > 0 else 0.0
    detection_rate = len(detected) / total if total > 0 else 0.0
    detected_acc  = len(correct) / len(detected) if detected else 0.0

    latencies = [r["latency_ms"] for r in detected if r["latency_ms"] is not None]
    mean_latency = np.mean(latencies) if latencies else float("nan")

    print(f"\n{'='*60}")
    print(f"  {system_name}")
    print(f"{'='*60}")
    print(f"  Total trials:      {total}")
    print(f"  Detected:          {len(detected)} ({detection_rate*100:.1f}%)")
    print(f"  Missed:            {len(missed)}")
    print(f"  Overall accuracy:  {overall_acc*100:.1f}%  (correct / all trials)")
    print(f"  Detected accuracy: {detected_acc*100:.1f}%  (correct / detected)")
    print(f"  Mean latency:      {mean_latency:.1f} ms")

    if detected:
        true_labels = [r["true"] for r in detected]
        pred_labels = [r["pred"] for r in detected]

        print(f"\n  Classification report (detected trials only):")
        print(classification_report(
            true_labels, pred_labels,
            labels=FINGERS,
            zero_division=0,
        ))

        print(f"  Confusion matrix (rows=true, cols=pred):")
        cm = confusion_matrix(true_labels, pred_labels, labels=FINGERS)
        header = "         " + "  ".join(f"{f[:5]:>5}" for f in FINGERS)
        print(f"  {header}")
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>5}" for v in row)
            print(f"  {FINGERS[i][:5]:>5}  {row_str}")

    return {
        "system":          system_name,
        "total":           total,
        "detected":        len(detected),
        "missed":          len(missed),
        "overall_acc":     overall_acc,
        "detection_rate":  detection_rate,
        "detected_acc":    detected_acc,
        "mean_latency_ms": mean_latency,
    }


def find_matching_files(eval_dir: Path, trials_path: Path):
    """Find rf/cnn/tapxr files closest in time to the trials file."""
    # get timestamp from trials filename: eval_20260422_232528_trials.csv
    parts = trials_path.stem.split("_")
    # date and time are parts[-3] and parts[-2]
    trials_dt = parts[-3] + parts[-2]  # e.g. "20260422232528"

    def time_distance(path: Path, suffix: str) -> int:
        p = path.stem.replace(suffix, "")
        candidate_dt = "".join(p.split("_")[-2:])  # extract YYYYMMDDHHMMSS
        return abs(int(candidate_dt) - int(trials_dt))

    rf_candidates    = list(eval_dir.glob("eval_*_rf.csv"))
    cnn_candidates   = list(eval_dir.glob("eval_*_cnn.csv"))
    tapxr_candidates = list(eval_dir.glob("eval_*_tapxr.csv"))

    rf_path    = min(rf_candidates,    key=lambda p: time_distance(p, "_rf"),    default=None)
    cnn_path   = min(cnn_candidates,   key=lambda p: time_distance(p, "_cnn"),   default=None)
    tapxr_path = min(tapxr_candidates, key=lambda p: time_distance(p, "_tapxr"), default=None)

    return rf_path, cnn_path, tapxr_path


def run_session(trials_path: Path, rf_path: Path, cnn_path: Path, tapxr_path: Path):
    print(f"\nSession: {trials_path.stem}")
    trials = load_trials(trials_path)
    print(f"  Loaded {len(trials)} trials")

    summaries = []

    if rf_path and rf_path.exists():
        rf_preds = load_predictions(rf_path)
        results  = match_predictions(trials, rf_preds)
        summaries.append(score(results, "Random Forest"))
    else:
        print("  RF predictions not found, skipping.")

    if cnn_path and cnn_path.exists():
        cnn_preds = load_predictions(cnn_path)
        results   = match_predictions(trials, cnn_preds)
        summaries.append(score(results, "CNN"))
    else:
        print("  CNN predictions not found, skipping.")

    if tapxr_path and tapxr_path.exists():
        tapxr_preds = load_predictions(tapxr_path)
        results     = match_predictions(trials, tapxr_preds)
        summaries.append(score(results, "TapXR"))
    else:
        print("  TapXR predictions not found, skipping.")

    return summaries


def main():
    # find all/most recent trial files in eval directory
    if EVAL_MODE == "acoustic":
        trial_files = sorted(ACOUSTIC_DIR.glob("eval_*_trials.csv"))
        EVAL_DIR = ACOUSTIC_DIR
    elif EVAL_MODE == "tapxr":
        trial_files = sorted(TAPXR_DIR.glob("eval_*_trials.csv"))
        EVAL_DIR = TAPXR_DIR
    if MOST_RECENT_ONLY:
        trial_files = [trial_files[-1]]

    if not trial_files:
        print(f"No eval trial files found in {EVAL_DIR}")
        sys.exit(1)

    print(f"Found {len(trial_files)} session(s):")
    for f in trial_files:
        print(f"  {f.name}")

    all_summaries = {"Random Forest": [], "CNN": [], "TapXR": []}

    for trials_path in trial_files:
        rf_path, cnn_path, tapxr_path = find_matching_files(EVAL_DIR, trials_path)
        session_summaries = run_session(trials_path, rf_path, cnn_path, tapxr_path)

        for s in session_summaries:
            if s["system"] in all_summaries:
                all_summaries[s["system"]].append(s)

    # Cross-session summary
    if len(trial_files) > 1:
        print(f"\n{'='*60}")
        print(f"  CROSS-SESSION SUMMARY ({len(trial_files)} sessions)")
        print(f"{'='*60}")

        for system, sessions in all_summaries.items():
            if not sessions:
                continue
            overall_accs  = [s["overall_acc"]  for s in sessions]
            detected_accs = [s["detected_acc"] for s in sessions]
            latencies     = [s["mean_latency_ms"] for s in sessions if not np.isnan(s["mean_latency_ms"])]

            print(f"\n  {system}")
            print(f"    Overall accuracy:  {np.mean(overall_accs)*100:.1f}% ± {np.std(overall_accs)*100:.1f}%")
            print(f"    Detected accuracy: {np.mean(detected_accs)*100:.1f}% ± {np.std(detected_accs)*100:.1f}%")
            print(f"    Mean latency:      {np.mean(latencies):.1f} ms" if latencies else "    Mean latency: N/A")


if __name__ == "__main__":
    main()