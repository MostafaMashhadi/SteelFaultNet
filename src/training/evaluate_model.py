"""
SteelFaultNet Evaluation Script (v2)
Author: Dr. Mostafa Mashhadizadeh
Last update: 2025-11-16

Evaluates the trained model on test data and outputs:
  1. Numeric Confusion Matrix (counts)
  2. Normalized Confusion Matrix (heatmap)
  3. Precision, Recall, F1, AUC, Accuracy
  4. JSON and TXT reports
"""

import json
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score
)

# -----------------------------
# Path settings
# -----------------------------
BASE_DIR   = Path(__file__).resolve().parents[2]
DATA_DIR   = BASE_DIR / "outputs" / "preprocessed"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "SteelFaultNet_v1.keras"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
FIG_DIR    = BASE_DIR / "outputs" / "figures"

REPORT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------------
# Load test data and model
# -----------------------------
print("📥 Loading test data and model ...")
data_path = DATA_DIR / "test_subsetE.npz"
with np.load(data_path) as npf:
    X_test, y_test = npf["X"], npf["y"]

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Prediction
# -----------------------------
print("🔮 Running inference ...")
y_pred_prob = model.predict(X_test, batch_size=8, verbose=1).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

# -----------------------------
# Metrics computation
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy  = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_prob)
report    = classification_report(y_test, y_pred, output_dict=True)

report["accuracy"] = accuracy
report["AUC"] = auc_score
report["confusion_matrix"] = {
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}

# -----------------------------
# Save reports
# -----------------------------
json_path = REPORT_DIR / "classification_report_v2.json"
txt_path  = REPORT_DIR / "classification_summary_v2.txt"

with open(json_path, "w") as f:
    json.dump(report, f, indent=4)

with open(txt_path, "w") as f:
    f.write("STEELFAULTNET EVALUATION REPORT\n")
    f.write("=" * 45 + "\n\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"AUC      : {auc_score:.4f}\n")
    f.write(f"Confusion Matrix (counts): [{tn}  {fp}; {fn}  {tp}]\n\n")
    f.write(json.dumps(report, indent=4))
    f.write("\n")

print(f"✅ Reports saved: {json_path.name} and {txt_path.name}")

# -----------------------------
# Confusion Matrix visualization
# -----------------------------
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Raw counts heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy", "Fault"],
            yticklabels=["Healthy", "Fault"], ax=axes[0])
axes[0].set_title("Confusion Matrix (Counts)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Normalized heatmap
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=["Healthy", "Fault"],
            yticklabels=["Healthy", "Fault"], ax=axes[1])
axes[1].set_title("Confusion Matrix (Normalized)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
fig_out = FIG_DIR / "confusion_matrix_v2.png"
plt.savefig(fig_out, dpi=200)
plt.close()

print(f"✅ Confusion matrix image saved → {fig_out}")
print(cm)
print(f"[TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
print(f"✅ AUC={auc_score:.3f}, Accuracy={accuracy:.3f}")
