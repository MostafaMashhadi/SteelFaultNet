"""
SteelFaultNet - Subset E Preprocessing
--------------------------------------
Prepares train/valid/test data from NOVIC+ Subset E for model training.
Each sample = 4s signal (102400 points × 9 channels).

Output:
    outputs/preprocessed/train_subsetE.npz
    outputs/preprocessed/valid_subsetE.npz
    outputs/preprocessed/test_subsetE.npz
"""

import os
import numpy as np
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 102400
N_CHANNELS = 9

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------

def load_pair(data_file: Path):
    """Load a pair (data + name) and extract labels from paths."""
    logging.info(f"Loading {data_file.name}")

    names_file = data_file.name.replace("data", "npy_name")
    names_path = data_file.parent / names_file

    X = np.load(data_file)
    names = np.load(names_path)

    y = np.array([1 if "/anomaly/" in n else 0 for n in names], dtype=np.int8)

    logging.info(f"→ Loaded {X.shape[0]} samples; shape={X.shape}; labels={np.bincount(y)}")
    return X, y


def normalize_dataset(X_train, X_valid, X_test):
    """Standardize data per channel using training stats."""
    logging.info("Normalizing per-channel using StandardScaler...")

    scalers = []
    X_train_norm = np.empty_like(X_train, dtype=np.float32)
    X_valid_norm = np.empty_like(X_valid, dtype=np.float32)
    X_test_norm = np.empty_like(X_test, dtype=np.float32)

    for ch in tqdm(range(N_CHANNELS), desc="Channels"):
        scaler = StandardScaler()
        X_train_ch = X_train[:, :, ch]
        X_valid_ch = X_valid[:, :, ch]
        X_test_ch = X_test[:, :, ch]

        X_train_norm[:, :, ch] = scaler.fit_transform(X_train_ch)
        X_valid_norm[:, :, ch] = scaler.transform(X_valid_ch)
        X_test_norm[:, :, ch] = scaler.transform(X_test_ch)
        scalers.append(scaler)

    # Save scalers for reuse
    scaler_path = OUTPUT_DIR / "subsetE_scalers.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers, f)

    logging.info(f"Saved scalers → {scaler_path}")
    return X_train_norm, X_valid_norm, X_test_norm


# -------------------------------------------------------------------
# MAIN PREPROCESSING PIPELINE
# -------------------------------------------------------------------

def main():
    logging.info("Starting preprocessing for Subset E...")

    files = {
        "train": DATA_DIR / "train_data_4s_clf_subsetE.npy",
        "valid": DATA_DIR / "valid_data_4s_clf_subsetE.npy",
        "test":  DATA_DIR / "test_data_4s_clf_subsetE.npy",
    }

    # Load datasets
    X_train, y_train = load_pair(files["train"])
    X_valid, y_valid = load_pair(files["valid"])
    X_test,  y_test  = load_pair(files["test"])

    # Normalize datasets
    X_train, X_valid, X_test = normalize_dataset(X_train, X_valid, X_test)

    # Save compressed outputs
    np.savez_compressed(OUTPUT_DIR / "train_subsetE.npz", X=X_train, y=y_train)
    np.savez_compressed(OUTPUT_DIR / "valid_subsetE.npz", X=X_valid, y=y_valid)
    np.savez_compressed(OUTPUT_DIR / "test_subsetE.npz",  X=X_test,  y=y_test)

    logging.info("✅ Preprocessing completed successfully.")
    logging.info(f"Outputs saved in → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
