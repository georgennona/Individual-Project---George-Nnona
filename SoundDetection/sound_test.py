import queue
import time
from collections import deque
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter


SAMPLE_RATE = 16000
BLOCK_SIZE = 128               # small block for low latency
CHANNELS = 1

HP_CUTOFF = 120.0              # high-pass filter cutoff
THRESHOLD_MULTIPLIER = 2.0     # sensitivity; tune this
MIN_TRIGGER_GAP_MS = 70        # refractory period
NOISE_FLOOR_ALPHA = 0.995      # slow background tracker

# short rolling window for future feature extraction
ROLLING_BUFFER_SECONDS = 0.5


# High-pass filter
# Adapted from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
def make_highpass(cutoff_hz: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="highpass")
    return b, a


class TapDetector:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        self.b, self.a = make_highpass(HP_CUTOFF, sample_rate)

        self.noise_floor = 1e-6 # background energy
        self.last_trigger_time = 0.0

        max_samples = int(ROLLING_BUFFER_SECONDS * sample_rate)
        self.rolling_buffer = deque(maxlen=max_samples)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        mono = indata[:, 0].copy()
        self.audio_queue.put(mono)

    def process_block(self, block: np.ndarray) -> bool:
        filtered = lfilter(self.b, self.a, block)

        self.rolling_buffer.extend(filtered.tolist())

        block_rms = np.sqrt(np.mean(filtered ** 2) + 1e-12)
        block_peak = np.max(np.abs(filtered)) + 1e-12

        now = time.time()
        ms_since_last = (now - self.last_trigger_time) * 1000.0

        # only update noise floor outside the refractory period
        if ms_since_last > MIN_TRIGGER_GAP_MS:
            self.noise_floor = (
                NOISE_FLOOR_ALPHA * self.noise_floor
                + (1.0 - NOISE_FLOOR_ALPHA) * block_rms
            )

        rms_threshold = max(self.noise_floor * THRESHOLD_MULTIPLIER, 5e-5)
        peak_threshold = max(rms_threshold * 3.0, 2e-4)

        # trigger if either RMS or peak exceeds thresholds (and outside refractory period)
        is_trigger = (
            ms_since_last > MIN_TRIGGER_GAP_MS
            and (
                block_rms > rms_threshold
                or block_peak > peak_threshold
            )
        )

        if is_trigger:
            self.last_trigger_time = now
            print(
                f"TAP detected | rms={block_rms:.5f} | peak={block_peak:.5f} "
                f"| noise={self.noise_floor:.5f}"
            )
            return True

        return False

    def run(self):
        print("Listening... tap on a desk or surface near the mic. Ctrl+C to stop.")
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
    detector = TapDetector(SAMPLE_RATE)
    detector.run()