# code for generating diagrams

"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_tap_spectrogram(wav_path, title, ax):
    y, sr = librosa.load(wav_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=256, hop_length=64, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(log_mel, sr=sr, hop_length=64, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
plot_tap_spectrogram("C:/Users/georg/TapDataset/processed/index/index_session_20260415_171422_00003.wav", 'Index finger', axes[0])
plot_tap_spectrogram("C:/Users/georg/TapDataset/processed/ring/ring_session_20260416_134954_00157.wav", 'Ring finger', axes[1])
plt.tight_layout()
plt.savefig('spectrogram_comparison.png', dpi=150, bbox_inches='tight')

#################################

import matplotlib.pyplot as plt
import numpy as np

sessions = [f'S{i+1}' for i in range(10)]
rf_acc = [66.0, 76.0, 86.0, 78.0, 74.0, 68.0, 60.0, 66.0, 66.0, 70.0]
cnn_acc = [60.0, 60.0, 64.0, 70.0, 44.0, 54.0, 66.0, 62.0, 52.0, 60.0]

x = np.arange(len(sessions))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, rf_acc, width, label='Random Forest', color='#378ADD')
bars2 = ax.bar(x + width/2, cnn_acc, width, label='CNN', color='#E24B4A')

ax.axhline(y=71.0, color='#378ADD', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=59.2, color='#E24B4A', linestyle='--', linewidth=1, alpha=0.7)

ax.set_xlabel('Session')
ax.set_ylabel('Overall accuracy (%)')
ax.set_xticks(x)
ax.set_xticklabels(sessions)
ax.set_ylim(0, 100)
ax.legend()
ax.text(9.6, 72.5, 'RF mean', fontsize=8, color='#378ADD')
ax.text(9.6, 60.7, 'CNN mean', fontsize=8, color='#E24B4A')

plt.tight_layout()
plt.savefig('per_session_accuracy.png', dpi=150, bbox_inches='tight')


###################################

import matplotlib.pyplot as plt
import numpy as np

systems = ['Random Forest', 'CNN', 'TapXR']
overall_acc = [71.0, 59.2, 94.2]
detection_rate = [99.8, 99.8, 94.2]
detected_acc = [71.2, 59.3, 100.0]

x = np.arange(len(systems))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, overall_acc, width, label='Overall accuracy', color='#378ADD')
ax.bar(x, detection_rate, width, label='Detection rate', color='#1D9E75')
ax.bar(x + width, detected_acc, width, label='Detected accuracy', color='#EF9F27')

ax.set_ylabel('Percentage (%)')
ax.set_xticks(x)
ax.set_xticklabels(systems)
ax.set_ylim(0, 110)
ax.legend()
ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('system_comparison.png', dpi=150, bbox_inches='tight')

"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

rf_cm = np.array([
    [79,  8,  7,  2,  4],
    [ 3, 78,  8,  8,  3],
    [10,  2, 77,  8,  3],
    [ 5,  9,  6, 66, 14],
    [11, 31,  0,  2, 55]
])

cnn_cm = np.array([
    [71, 18,  9,  0,  2],
    [25, 72,  2,  0,  1],
    [ 9, 14, 74,  0,  3],
    [19, 22, 13, 42,  4],
    [27, 26,  2,  7, 37]
])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, cm, title in zip(axes, [rf_cm, cnn_cm], ['Random Forest', 'CNN']):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=fingers, yticklabels=fingers,
                ax=ax, cbar=True)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')