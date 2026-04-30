import os
import queue
import time
from collections import deque
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, lfilter


LABEL = "ring"

SAMPLE_RATE = 16000
BLOCK_SIZE = 128
CHANNELS = 1

HP_CUTOFF = 120.0
THRESHOLD_MULTIPLIER = 2.0
MIN_TRIGGER_GAP_MS = 150
NOISE_FLOOR_ALPHA = 0.995

PRE_TRIGGER_MS = 30
POST_TRIGGER_MS = 170
ROLLING_BUFFER_SECONDS = 1.0

OUTPUT_ROOT = "dataset (old)"


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
def make_highpass(cutoff_hz: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="highpass")
    return b, a


class TapRecorder:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        self.b, self.a = make_highpass(HP_CUTOFF, sample_rate)

        self.noise_floor = 1e-6
        self.last_trigger_time = 0.0

        self.pre_trigger_samples = int(PRE_TRIGGER_MS * sample_rate / 1000)
        self.post_trigger_samples = int(POST_TRIGGER_MS * sample_rate / 1000)

        max_samples = int(ROLLING_BUFFER_SECONDS * sample_rate)
        self.raw_buffer = deque(maxlen=max_samples)

        self.label_dir = os.path.join(OUTPUT_ROOT, LABEL)
        os.makedirs(self.label_dir, exist_ok=True)

        self.clip_counter = self._get_next_index()

        # Capture state
        self.capture_armed = False
        self.capture_pre_audio = None
        self.capture_post_audio = []

    def _get_next_index(self) -> int:
        existing = [
            name for name in os.listdir(self.label_dir)
            if name.endswith(".wav")
        ]
        return len(existing) + 1

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        mono = indata[:, 0].copy()
        self.audio_queue.put(mono)

    def save_clip(self, clip: np.ndarray):
        filename = f"{LABEL}_{self.clip_counter:04d}.wav"
        path = os.path.join(self.label_dir, filename)

        peak = np.max(np.abs(clip)) + 1e-12
        clip_out = 0.95 * clip / peak if peak > 1e-6 else clip

        sf.write(path, clip_out.astype(np.float32), self.sample_rate)
        print(f"Saved: {os.path.abspath(path)}")
        self.clip_counter += 1

    def finish_capture_if_ready(self):
        if not self.capture_armed:
            return

        post_audio = np.concatenate(self.capture_post_audio) if self.capture_post_audio else np.array([], dtype=np.float32)

        if len(post_audio) < self.post_trigger_samples:
            return

        post_audio = post_audio[:self.post_trigger_samples]
        clip = np.concatenate([self.capture_pre_audio, post_audio])

        self.save_clip(clip)

        self.capture_armed = False
        self.capture_pre_audio = None
        self.capture_post_audio = []

    def process_block(self, block: np.ndarray):
        # store raw audio
        for x in block:
            self.raw_buffer.append(float(x))

        if self.capture_armed:
            self.capture_post_audio.append(block.copy())
            self.finish_capture_if_ready()

        filtered = lfilter(self.b, self.a, block)

        block_rms = np.sqrt(np.mean(filtered ** 2) + 1e-12)
        block_peak = np.max(np.abs(filtered)) + 1e-12

        now = time.time()
        ms_since_last = (now - self.last_trigger_time) * 1000.0

        if ms_since_last > MIN_TRIGGER_GAP_MS:
            self.noise_floor = (
                NOISE_FLOOR_ALPHA * self.noise_floor
                + (1.0 - NOISE_FLOOR_ALPHA) * block_rms
            )

        rms_threshold = max(self.noise_floor * THRESHOLD_MULTIPLIER, 5e-5)
        peak_threshold = max(rms_threshold * 3.0, 2e-4)

        is_trigger = (
            not self.capture_armed
            and ms_since_last > MIN_TRIGGER_GAP_MS
            and (block_rms > rms_threshold or block_peak > peak_threshold)
        )

        if not is_trigger:
            return

        self.last_trigger_time = now

        if len(self.raw_buffer) < self.pre_trigger_samples:
            return

        recent_raw = np.array(self.raw_buffer, dtype=np.float32)
        pre_audio = recent_raw[-self.pre_trigger_samples:].copy()

        self.capture_armed = True
        self.capture_pre_audio = pre_audio
        self.capture_post_audio = [block.copy()]

        print(
            f"TAP | rms={block_rms:.5f} | peak={block_peak:.5f} "
            f"| noise={self.noise_floor:.5f}"
        )

        self.finish_capture_if_ready()

    def run(self):
        print(f"Recording label: {LABEL}")
        print("Tap repeatedly. Press Ctrl+C to stop.")
        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype="float32",
            callback=self.audio_callback,
        ):
            while True:
                block = self.audio_queue.get()
                self.process_block(block)


if __name__ == "__main__":
    recorder = TapRecorder(SAMPLE_RATE)
    recorder.run()