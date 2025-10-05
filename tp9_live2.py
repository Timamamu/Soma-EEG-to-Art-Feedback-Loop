

import torch
from torchvision import transforms
import torch.utils.data

import os
import cv2

from pix2pix_model import *
from pix2pix_dataset import *

import numpy as np
import cv2
from scipy import signal
from pylsl import StreamInlet, resolve_byprop
from pathlib import Path
import time
import threading
import sys

# _______________________________________________________________________________________________________________________
# _______________________________________________________________________________________________________________________
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_FOLDER = os.path.join(THIS_FOLDER, f'models/SHAPE_MODEL')

# ------------------------------------------------------------------ FLICKER-FIX PARAMETERS
MIN_DB, MAX_DB = -100, -20        # colour scale applied to every frame
# ------------------------------------------------------------------

# _______________________________________________________________________________________________________________________
# MODEL
# _______________________________________________________________________________________________________________________
model = Pix2PixModel.createFromFolder(MODEL_FOLDER, DEVICE)
model.eval()
# _______________________________________________________________________________________________________________________

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

with torch.no_grad():
    while not stop_flag:
        if stop_flag:
            print("Stopping capture...")
            sys.exit(0)
        chunk, ts = inlet.pull_chunk(timeout=0.1)
        if chunk:
            samples.extend(chunk)

        if len(samples) > array_size:
            remove = len(samples) - array_size
            samples = samples[remove:]

        if len(samples) < array_size:
            print("No data received.")
            continue

        # Extract TP9 data
        samples_np = np.array(samples)
        tp9 = samples_np[:, tp9_index]

        # Generate spectrogram
        f, t_vals, Sxx = signal.spectrogram(tp9, fs=sampling_rate, nperseg=256, noverlap=192)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # ------------ FLICKER-FREE NORMALISATION ------------
        Sxx_db = np.clip(Sxx_db, MIN_DB, MAX_DB)
        img = (Sxx_db - MIN_DB) / (MAX_DB - MIN_DB)
        # -----------------------------------------------------

        img = cv2.resize(img, (256, 256))
        img = (img * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)

        img = cv2.resize(img_color, (model.info.image_size, model.info.image_size))
        converted = model.applyToImage(img)

        # concatenate img and converted
        img = cv2.resize(img, (1024, 1024))
        converted = cv2.resize(converted, (1024, 1024))

        converted = np.concatenate([img, converted], axis=1)

        cv2.imshow('Converted', converted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
