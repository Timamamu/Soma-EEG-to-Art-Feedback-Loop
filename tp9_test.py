import numpy as np
import cv2
from scipy import signal
from pylsl import StreamInlet, resolve_byprop
from pathlib import Path
import time
import threading
import sys

# === Flag to stop the loop ===
stop_flag = False

def wait_for_s_key():
    global stop_flag
    print("Press 's' then Enter to stop.")
    while True:
        user_input = input()
        if user_input.lower() == 's':
            stop_flag = True
            break

# === Start listener thread ===
threading.Thread(target=wait_for_s_key, daemon=True).start()

# === Setup ===
save_dir = Path("live_tp9_spectrograms")
save_dir.mkdir(parents=True, exist_ok=True)

duration = 7  # seconds per chunk
sampling_rate = 256  # Muse default

array_size = int(duration * sampling_rate)

# === Connect to EEG stream ===
print("Searching for EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=5)
if not streams:
    raise RuntimeError("No EEG stream found.")
inlet = StreamInlet(streams[0])

# === Get channel names and find TP9 index ===
info = inlet.info()
ch = info.desc().child('channels').first_child()
channels = [ch.child_value('label')]
for i in range(1, info.channel_count()):
    ch = ch.next_sibling()
    channels.append(ch.child_value('label'))

if "TP9" not in channels:
    raise ValueError("TP9 channel not found in EEG stream.")
tp9_index = channels.index("TP9")

print("Started collecting data...")

samples = []

# === Main Loop ===
while not stop_flag:
    
    
    if stop_flag:
        print("Stopping capture...")
        sys.exit(0)
    chunk, ts = inlet.pull_chunk(timeout=0.1)
    if chunk:
        samples.extend(chunk)

    if len(samples)>array_size:
        remove = len(samples) - array_size
        samples = samples[remove:]

    if  len(samples)< array_size:
        print("No data received.")
        continue

    # Extract TP9 data
    samples_np = np.array(samples)
    tp9 = samples_np[:, tp9_index]

    # Generate spectrogram
    f, t_vals, Sxx = signal.spectrogram(tp9, fs=sampling_rate, nperseg=256, noverlap=192)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    img = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db))
    img = cv2.resize(img, (256, 256))
    img = (img * 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)

    # Save with timestamp
    #filename = save_dir / f"{int(time.time())}.jpeg"
    #cv2.imwrite(str(filename), img_color)
   # print(f"Saved: {filename}")
    cv2.imshow("speccy", img_color)
    cv2.waitKey(1)

print("Stopped.")
