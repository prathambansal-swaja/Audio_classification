import json
from pathlib import Path

SILENCE_DIR = Path(r"F:/Audio_classification/data/processed/silence_clips")
TRAIN_JSON = Path(r"F:/Audio_classification/data/splits/train_files.json")

# Load existing train split
with open(TRAIN_JSON, "r") as f:
    train_files = json.load(f)

existing_count = len(train_files)

# Add each silence file as its own entry
for wav in SILENCE_DIR.glob("*.wav"):
    train_files.append({
        "path": str(wav),
        "label": "_background_noise_",   # raw label
        "speaker": "silence_gen"         # synthetic speaker ID
    })

# Save updated train split
with open(TRAIN_JSON, "w") as f:
    json.dump(train_files, f, indent=2)

print(f"Old train count: {existing_count}")
print(f"New train count: {len(train_files)}")
print(f"Added {len(train_files) - existing_count} silence samples.")
