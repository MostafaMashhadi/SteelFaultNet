# src/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# ==== Data parameters ====
SAMPLE_RATE = 25600   # Hz
FRAME_DURATION = 4     # seconds
NUM_CHANNELS = 9       # 4 vib + 2 temp + 1 torque + 2 rpm
SEQ_LEN = SAMPLE_RATE * FRAME_DURATION  # 102400

# ==== Model parameters ====
P_FAILURE_THRESHOLD = 0.8

# ==== Training ====
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
VALID_SPLIT = 0.2
SEED = 42

def ensure_dirs():
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_dirs()
    print("Configuration verified.")

