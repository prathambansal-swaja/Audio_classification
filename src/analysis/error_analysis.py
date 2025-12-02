# src/analysis/error_analysis.py
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report

from src.config import TARGET_LABELS
from src.data import dataset_tf

PROJECT_ROOT = Path(r"F:\Audio_classification")
EXPERIMENT_DIR = sorted((PROJECT_ROOT / "experiments").glob("run_*"), key=os.path.getmtime)[-1]
MODEL_H5 = EXPERIMENT_DIR / "model.h5"
OUT_DIR = EXPERIMENT_DIR / "analysis"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# 1) load model
model = tf.keras.models.load_model(MODEL_H5)
print("Loaded model:", MODEL_H5)

# 2) build val dataset (no shuffling)
_, val_ds = dataset_tf.build_train_val_datasets(batch_size=32)
# val_ds yields (batch_x, batch_y) where batch_x shape (B, T, n_mels, 1)

# 3) run predictions on entire val set and collect file paths + true labels
# To collect file paths we need to re-create path list from splits
import json
val_json = PROJECT_ROOT / "data" / "splits" / "val_files.json"
with open(val_json, "r") as f:
    items = json.load(f)
val_paths = [it["path"] for it in items]
val_labels = []
for it in items:
    # map to target idx as your pipeline does
    from src.config import map_raw_label, LABEL2INDEX
    val_labels.append(LABEL2INDEX[map_raw_label(it["label"])])

# Predict in the same order as val_paths: build a dataset without shuffle
val_paths_tf = tf.constant(val_paths)
val_labels_tf = tf.constant(val_labels, dtype=tf.int32)
ds = tf.data.Dataset.from_tensor_slices((val_paths_tf, val_labels_tf))
ds = ds.map(dataset_tf.parse_path_label, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(32)
ds = ds.prefetch(tf.data.AUTOTUNE)

y_true = []
y_pred = []
file_list = []

for batch_x, batch_y in ds:
    probs = model.predict(batch_x, verbose=0)
    preds = np.argmax(probs, axis=1).tolist()
    y_pred.extend(preds)
    y_true.extend(batch_y.numpy().tolist())

# 4) classification report + confusion matrix
print("Classification report:")
print(classification_report(y_true, y_pred, target_names=TARGET_LABELS, digits=4))

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_LABELS))))
cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

# save confusion matrix plot
plt.figure(figsize=(10,8))
plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix (val)")
plt.colorbar()
tick_marks = np.arange(len(TARGET_LABELS))
plt.xticks(tick_marks, TARGET_LABELS, rotation=45, ha="right")
plt.yticks(tick_marks, TARGET_LABELS)
fmt = '.2f'
thresh = cm_norm.max() / 2.
for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
    plt.text(j, i, format(cm_norm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm_norm[i, j] > thresh else "black",
             fontsize=8)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_val.png", dpi=150)
print("Saved confusion matrix to:", OUT_DIR / "confusion_matrix_val.png")

# 5) top misclassifications: find top confusion pairs and list examples
cm_flat = cm.copy()
# zero diagonal
np.fill_diagonal(cm_flat, 0)
# find top 5 confusion (true, pred) pairs by count
pairs = []
for i in range(cm_flat.shape[0]):
    for j in range(cm_flat.shape[1]):
        pairs.append(((i,j), cm_flat[i,j]))
pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
top_pairs = [p[0] for p in pairs_sorted[:5]]

# Collect file examples for each pair (scan val set again)
examples = []
# regenerate dataset in order so we can index files (we used val_paths order)
for idx, (p, t, pr) in enumerate(zip(val_paths, y_true, y_pred)):
    for (ti, pi) in top_pairs:
        if t == ti and pr == pi:
            examples.append({"true": TARGET_LABELS[ti], "pred": TARGET_LABELS[pi], "path": p})
# group by pair and keep top 5 each
df = pd.DataFrame(examples)
out_rows = []
for (ti, pi) in top_pairs:
    subset = df[(df["true"]==TARGET_LABELS[ti]) & (df["pred"]==TARGET_LABELS[pi])]
    out_rows.append({
        "true": TARGET_LABELS[ti],
        "pred": TARGET_LABELS[pi],
        "count": len(subset),
        "examples": subset["path"].tolist()[:5]
    })

# save CSV summary
pd.DataFrame(out_rows).to_csv(OUT_DIR / "top_confusions_examples.csv", index=False)
print("Saved top confusion examples to:", OUT_DIR / "top_confusions_examples.csv")
