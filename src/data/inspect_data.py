import os
import glob
import soundfile as sf

DATA_DIR = r"F:/Audio_classification/data/raw/train/train/audio" 

labels = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print("Found label folders:", labels)

for lbl in labels[:10]:
    files = glob.glob(os.path.join(DATA_DIR, lbl, "*.wav"))
    if not files:
        continue
    f = files[0]
    data, sr = sf.read(f)
    print(f"Label: {lbl}, File: {os.path.basename(f)}, SR: {sr}, Duration: {len(data)/sr:.3f}s")
