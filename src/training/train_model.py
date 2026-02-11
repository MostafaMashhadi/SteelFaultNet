"""
SteelFaultNet - Model Training Pipeline (Final Tuned Version, Fixed Plots + ConfMat)
-----------------------------------------------------------------------------------
Only modifications:
  • Robust plotting to avoid empty figures
  • Added confusion matrix (count-based, non-normalized)
"""

import os
import json
import time
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras import layers, models, callbacks

# Optional converters
try:
    import onnx, tf2onnx
    ONNX_OK = True
except Exception:
    ONNX_OK = False

try:
    import coremltools as ct
    COREML_OK = True
except Exception:
    COREML_OK = False

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "outputs" / "preprocessed"
OUTPUT_BASE = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_BASE / "models"
LOG_DIR = OUTPUT_BASE / "training_logs"
FIG_DIR = OUTPUT_BASE / "figures"
REPORT_DIR = OUTPUT_BASE / "reports"

for d in [MODEL_DIR, LOG_DIR, FIG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def load_npz(split):
    """Load npz splits from preprocessed Subset E directory."""
    path = DATA_DIR / f"{split}_subsetE.npz"
    with np.load(path) as npf:
        X, y = npf["X"], npf["y"]
    return X.astype(np.float32), y.astype(np.float32)


def estimate_remaining(epoch, total_epochs, start_time):
    """Estimate remaining time for training."""
    elapsed = time.time() - start_time
    avg_epoch = elapsed / max(1, epoch)
    remaining = (total_epochs - epoch) * avg_epoch
    return avg_epoch, remaining


# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------
logging.info("Loading preprocessed subsets …")
X_train, y_train = load_npz("train")
X_valid, y_valid = load_npz("valid")
X_test, y_test = load_npz("test")

input_shape = X_train.shape[1:]
logging.info(f"Dataset shapes → Train={X_train.shape}, Valid={X_valid.shape}, Test={X_test.shape}")
logging.info(f"Input shape per sample: {input_shape}")

# -------------------------------------------------------------------
# MODEL DEFINITION
# -------------------------------------------------------------------
def build_model(input_shape):
    """Build hybrid CNN‑GRU architecture."""
    inputs = layers.Input(shape=input_shape, name="signal_input")

    # CNN feature extractor
    x = layers.Conv1D(32, 5, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)

    x = layers.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)

    # Temporal modeling
    x = layers.GRU(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="P_failure_24h")(x)
    model = models.Model(inputs, outputs, name="SteelFaultNet_v1")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="AUC")]
    )
    return model

model = build_model(input_shape)
model_summary_path = REPORT_DIR / "model_summary.txt"
with open(model_summary_path, "w") as f:
    model.summary(print_fn=lambda s: f.write(s + "\n"))
logging.info(f"Model summary saved → {model_summary_path}")

# -------------------------------------------------------------------
# CALLBACKS (Optimized)
# -------------------------------------------------------------------
checkpoint_best = MODEL_DIR / "best_checkpoint.keras"
csv_logger_path = LOG_DIR / "training_metrics.csv"
tensorboard_dir = LOG_DIR / time.strftime("%Y%m%d‑%H%M%S")

cb = [
    callbacks.ModelCheckpoint(checkpoint_best, monitor="val_loss", save_best_only=True),
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    callbacks.CSVLogger(csv_logger_path, append=False),
    callbacks.TensorBoard(log_dir=tensorboard_dir)
]

# -------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------
batch_size = 8     # tuned for 8 GB RAM (M3)
epochs = 30       # reduced for faster convergence

logging.info(f"Training started → batch_size={batch_size}, epochs={epochs}")
start_time = time.time()

for epoch in range(epochs):
    model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        initial_epoch=epoch,
        epochs=epoch + 1,
        batch_size=batch_size,
        verbose=1,
        callbacks=cb
    )

    avg_epoch, remaining = estimate_remaining(epoch + 1, epochs, start_time)
    progress = (epoch + 1) / epochs * 100
    logging.info(
        f"[Progress {progress:.1f}%] Epoch {epoch+1}/{epochs} "
        f"(avg {avg_epoch:.1f}s/epoch, est. {remaining/60:.1f} min left)"
    )

duration = time.time() - start_time
logging.info(f"Training finished in {duration/60:.1f} minutes")

# -------------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------------
logging.info("Evaluating on test set …")
test_results = model.evaluate(X_test, y_test, verbose=0)
metrics_dict = dict(zip(model.metrics_names, [float(r) for r in test_results]))
metrics_dict["training_time_min"] = round(duration / 60, 2)

report_path = REPORT_DIR / "metrics_final.json"
with open(report_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)
logging.info(f"Metrics stored → {report_path}")

# -------------------------------------------------------------------
# PLOTS  (robust & non-empty)
# -------------------------------------------------------------------
def plot_training(csv_path, fig_dir):
    """Plot loss and accuracy curves from CSVLogger output."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Loss ---
    if "loss" in df and "val_loss" in df:
        axes[0].plot(df.index, df["loss"], label="train")
        axes[0].plot(df.index, df["val_loss"], label="val")
    else:
        axes[0].plot([], [])
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # --- Accuracy or AUC ---
    if "accuracy" in df and "val_accuracy" in df:
        axes[1].plot(df.index, df["accuracy"], label="train")
        axes[1].plot(df.index, df["val_accuracy"], label="val")
    elif "AUC" in df and "val_AUC" in df:
        axes[1].plot(df.index, df["AUC"], label="train")
        axes[1].plot(df.index, df["val_AUC"], label="val")
        axes[1].set_title("AUC Curve")
    else:
        axes[1].plot([], [])
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig_path = fig_dir / "training_curves.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logging.info(f"Training curves saved to {fig_path}")

plot_training(csv_logger_path, FIG_DIR)

# -------------------------------------------------------------------
# CONFUSION MATRIX (count‑based)
# -------------------------------------------------------------------
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    logging.info("Generating confusion matrix (counts)…")
    y_pred = (model.predict(X_test, batch_size=batch_size) > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test.astype(int), y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fault"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Counts)")
    cm_path = FIG_DIR / "confusion_matrix_counts.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logging.info(f"Confusion matrix saved → {cm_path}")
except Exception as e:
    logging.warning(f"Confusion matrix generation failed: {e}")

# -------------------------------------------------------------------
# EXPORT MODELS
# -------------------------------------------------------------------
keras_path = MODEL_DIR / "SteelFaultNet_v1.keras"
h5_path = MODEL_DIR / "SteelFaultNet_v1.h5"
onnx_path = MODEL_DIR / "SteelFaultNet_v1.onnx"
mlmodel_path = MODEL_DIR / "SteelFaultNet_v1.mlmodel"

model.save(keras_path)
model.save(h5_path)
logging.info(f"Saved model → {keras_path}, {h5_path}")

# ONNX export
if ONNX_OK:
    try:
        spec = (tf.TensorSpec((None,) + input_shape, tf.float32, name="signal_input"),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        onnx.save_model(onnx_model, onnx_path)
        logging.info(f"ONNX model exported → {onnx_path}")
    except Exception as e:
        logging.warning(f"ONNX export failed: {e}")
else:
    logging.info("ONNX toolchain unavailable → skipping export")

# CoreML export
if COREML_OK:
    try:
        coreml_model = ct.convert(model)
        coreml_model.save(mlmodel_path)
        logging.info(f"CoreML model exported → {mlmodel_path}")
    except Exception as e:
        logging.warning(f"CoreML export failed: {e}")
else:
    logging.info("CoreML toolchain unavailable → skipping export")

# -------------------------------------------------------------------
# COMPLETION LOG
# -------------------------------------------------------------------
logging.info("✅ SteelFaultNet training and documentation completed successfully.")
