import json
from collections import Counter
from pathlib import Path

from src.config import map_raw_label, TARGET_LABELS

TRAIN_JSON = Path(r"F:/Audio_classification/data/splits/train_files.json")

with open(TRAIN_JSON, "r") as f:
    train_files = json.load(f)

# Count mapped labels
counter = Counter()
for item in train_files:
    raw_label = item["label"]
    mapped = map_raw_label(raw_label)
    counter[mapped] += 1

print("Target labels:", TARGET_LABELS)
print("\nCounts in TRAIN split after mapping:")
for lbl in TARGET_LABELS:
    print(f"{lbl:8s} -> {counter[lbl]}")
