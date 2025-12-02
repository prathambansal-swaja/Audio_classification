# src/data/compute_class_weights.py
import json
from pathlib import Path
from collections import Counter
import numpy as np

from src.config import map_raw_label, LABEL2INDEX, TARGET_LABELS

PROJECT_ROOT = Path(r"F:\Audio_classification")
TRAIN_JSON = PROJECT_ROOT / "data" / "splits" / "train_files.json"

with open(TRAIN_JSON, "r") as f:
    train_files = json.load(f)

counter = Counter()

for item in train_files:
    raw = item["label"]
    mapped = map_raw_label(raw)
    idx = LABEL2INDEX[mapped]
    counter[idx] += 1

print("Class counts:")
for i, lbl in enumerate(TARGET_LABELS):
    print(f"{i:2d} {lbl:10s} -> {counter[i]}")

# Compute weights inversely proportional to frequency
counts = np.array([counter[i] for i in range(len(TARGET_LABELS))], dtype=np.float32)
inv = 1.0 / counts
weights = inv / inv.sum() * len(TARGET_LABELS)

print("\nCLASS_WEIGHTS (for keras):")
for i, lbl in enumerate(TARGET_LABELS):
    print(f"{i:2d} {lbl:10s}: {weights[i]:.6f}")
