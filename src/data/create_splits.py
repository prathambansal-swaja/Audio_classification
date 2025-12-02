import os
import glob
import json
from sklearn.model_selection import train_test_split

AUDIO_DIR = r"F:/Audio_classification/data/raw/train/train/audio"

# All label folders
labels = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]

all_files = []
for lbl in labels:
    wavs = glob.glob(os.path.join(AUDIO_DIR, lbl, "*.wav"))
    for w in wavs:
        fname = os.path.basename(w)

        # Speaker ID is before "_nohash_"
        speaker_id = fname.split("_")[0]

        all_files.append({
            "path": w,
            "label": lbl,
            "speaker": speaker_id
        })

print("Total WAV files:", len(all_files))

# Extract speaker ids
all_speakers = list({f["speaker"] for f in all_files})

# Split speakers (80% train, 20% validation)
train_speakers, val_speakers = train_test_split(
    all_speakers,
    test_size=0.20,
    random_state=42
)

train_files = [f for f in all_files if f["speaker"] in train_speakers]
val_files   = [f for f in all_files if f["speaker"] in val_speakers]

print("Train files:", len(train_files))
print("Validation files:", len(val_files))

# Save splits for future loading
os.makedirs("F:\\Audio_classification\\data\\splits", exist_ok=True)

with open(r"F:/Audio_classification/data/splits/train_files.json", "w") as f:
    json.dump(train_files, f, indent=2)

with open(r"F:/Audio_classification/data/splits/val_files.json", "w") as f:
    json.dump(val_files, f, indent=2)

print("Saved splits to data/splits/")
