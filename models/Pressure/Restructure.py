import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import detrend
import matplotlib.pyplot as plt

# === CONFIG ===
BASE_DIR = "./"
TARGET_DIR = "./processed_fft/"
FOLDERS = ["bearing", "misalignment", "normal", "unbalance"]
SAMPLING_RATE = 20000  # Hz

os.makedirs(TARGET_DIR, exist_ok=True)

def process_file(filepath):
    # Load CSV assuming: time, x, y, z
    data = pd.read_csv(filepath, header=None)
    time = data.iloc[:, 0].values
    signal = detrend(data.iloc[:, 1].values)  # Use X-axis, remove DC/drift

    N = len(signal)
    fft_result = fft(signal)
    freq = fftfreq(N, d=1/SAMPLING_RATE)[:N//2]
    
    magnitude = 20 * np.log10(np.abs(fft_result[:N//2]) + 1e-8)  # In dB
    phase = np.angle(fft_result[:N//2], deg=True)

    # Create DataFrame in your format
    df_fft = pd.DataFrame({
        "Frequency": freq,
        "Magnitude": magnitude,
        "Phase": phase
    })

    return df_fft

# === BATCH PROCESS ===
for folder in FOLDERS:
    in_path = os.path.join(BASE_DIR, folder)
    out_path = os.path.join(TARGET_DIR, folder)
    os.makedirs(out_path, exist_ok=True)

    for filename in sorted(os.listdir(in_path))[:100]:  # Limit for testing
        if filename.endswith(".csv"):
            try:
                fft_df = process_file(os.path.join(in_path, filename))
                out_name = filename.replace(".csv", ".txt")
                fft_df.to_csv(os.path.join(out_path, out_name), sep="\t", index=False)
            except Exception as e:
                print(f"Error in {filename}: {e}")
