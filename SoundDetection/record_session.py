import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time
from datetime import datetime

SAMPLE_RATE = 16000
CHANNELS = 1
INPUT_DEVICE = 2
OUTPUT_DIR = r"C:/Users/georg/TapDataset/raw_sessions"

def record_session():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wav_path = os.path.join(OUTPUT_DIR, f"{session_id}.wav")
    meta_path = os.path.join(OUTPUT_DIR, f"{session_id}_meta.txt")

    print(f"\nSession ID: {session_id}")
    print(f"Using input device: {INPUT_DEVICE}")
    print("Press ENTER to start recording...")
    input()

    print("Recording... Press ENTER to stop.")
    print("When Unity shows 'SYNC TAP', perform one strong tap to align the session.")

    start_time = time.time()
    recording = []

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        recording.append(indata.copy())

        level = float(np.max(np.abs(indata)))
        if level > 0.001:
            print(f"level={level:.4f}")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=INPUT_DEVICE,
        callback=callback,
    ):
        input()

    end_time = time.time()

    audio = np.concatenate(recording, axis=0)

    # Normalize to 0.95 to avoid clipping
    peak = np.max(np.abs(audio)) + 1e-12
    audio_out = 0.95 * audio / peak if peak > 1e-6 else audio

    sf.write(wav_path, audio_out, SAMPLE_RATE)

    with open(meta_path, "w") as f:
        f.write(f"session_id: {session_id}\n")
        f.write(f"start_time_unix: {start_time}\n")
        f.write(f"end_time_unix: {end_time}\n")
        f.write(f"duration_sec: {end_time - start_time}\n")
        f.write(f"input_device: {INPUT_DEVICE}\n")

    print(f"Saved WAV: {wav_path}")
    print(f"Saved META: {meta_path}")

if __name__ == "__main__":
    record_session()