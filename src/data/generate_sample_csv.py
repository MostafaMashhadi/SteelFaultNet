"""
Generate sample_sensor_signals.csv
Author: Dr. M. Mashhadizadeh (Shiraz)
Creates a synthetic 9‑channel sensor dataset compatible with SteelFaultNet Industrial v2.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os

# --- Parameters ---
N = 102400  # samples per channel

# --- Physical ranges (industrial realistic) ---
rng_vib = (-30, 30)       # vibration [m/s^2]
rng_temp = (80, 120)      # temperature [°C]
rng_torque = (3800, 4300) # torque [N·m]
rng_rpm = (1100, 1200)    # rpm

# --- File paths ---
DATA_DIR = os.path.join("..", "..", "outputs", "preprocessed")
os.makedirs(DATA_DIR, exist_ok=True)
main_path = os.path.join(DATA_DIR, "sample_sensor_signals.csv")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = os.path.join(DATA_DIR, f"sample_sensor_signals_{timestamp}.csv")

# --- Generate signals ---
np.random.seed(42)
VibAx_A = np.random.normal(12, 5, N)
VibAx_B = np.random.normal(10, 4, N)
VibTr_A = np.random.normal(9, 4, N)
VibTr_B = np.random.normal(8, 3, N)
Temp_A = np.random.normal(100, 4, N)
Temp_B = np.random.normal(103, 4, N)
Torque = np.random.normal(4100, 50, N)
RPM_in = np.random.normal(1160, 5, N)
RPM_out = np.random.normal(1145, 5, N)

data = np.column_stack([
    VibAx_A, VibAx_B, VibTr_A, VibTr_B,
    Temp_A, Temp_B, Torque, RPM_in, RPM_out
])

# --- Clip to realistic ranges ---
data[:, 0:4] = np.clip(data[:, 0:4], *rng_vib)
data[:, 4:6] = np.clip(data[:, 4:6], *rng_temp)
data[:, 6] = np.clip(data[:, 6], *rng_torque)
data[:, 7:9] = np.clip(data[:, 7:9], *rng_rpm)

# --- Save ---
cols = ["VibAx_A", "VibAx_B", "VibTr_A", "VibTr_B",
        "Temp_A", "Temp_B", "Torque", "RPM_in", "RPM_out"]

df = pd.DataFrame(data, columns=cols)
df.to_csv(main_path, index=False)
df.to_csv(backup_path, index=False)

print(f"[✅] CSV generated successfully.")
print(f" → {main_path}")
print(f" → {backup_path}")
print(f"Shape = {df.shape}")
