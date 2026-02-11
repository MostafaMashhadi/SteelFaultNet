"""
Signal preprocessing pipeline for Industrial inference.
Replicates training‑phase filtering (LPF) and scaling (MaxAbs).
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# --- Low‑pass filter ---
def lpf(data, fs=25000, cutoff=1000):
    b, a = butter(4, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data, axis=0)

# --- Max‑Abs normalization ---
def normalize_to_unit(data):
    return data / np.max(np.abs(data), axis=0, keepdims=True)

# --- Full preprocessing pipeline ---
def preprocess_signal(csv_path):
    raw_df = pd.read_csv(csv_path)
    raw_array = raw_df.values.astype(np.float32)

    filtered = lpf(raw_array)
    normalized = normalize_to_unit(filtered)
    X_ready = normalized.reshape(1, 102400, 9)
    return X_ready
