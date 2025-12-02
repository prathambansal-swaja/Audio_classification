# src/data/generate_silence.py
import os
import glob
import soundfile as sf
from pathlib import Path
from src.features.audio_utils import load_wav, pad_or_trim, SR
import numpy as np

# Adjust paths to your environment
BG_DIR = Path(r"F:/Audio_classification/data/raw/train/train/audio/_background_noise_")
OUT_DIR = Path(r"F:/Audio_classification/data/processed/silence_clips")
CLIP_LEN_SEC = 1.0
OVERLAP_SEC = 0.5  # default overlap; change if you prefer 0.0
SR = SR
CLIP_SAMPLES = int(CLIP_LEN_SEC * SR)
STEP = int((CLIP_LEN_SEC - OVERLAP_SEC) * SR) if OVERLAP_SEC > 0 else CLIP_SAMPLES

os.makedirs(OUT_DIR, exist_ok=True)

bg_wavs = glob.glob(str(BG_DIR / "*.wav"))
print(f"Found background WAVs: {len(bg_wavs)}")

total_created = 0
for bg in bg_wavs:
    data, file_sr = load_wav(bg, sr=SR)
    # If background is shorter than clip, pad
    if len(data) < CLIP_SAMPLES:
        data = pad_or_trim(data, CLIP_SAMPLES)
    # Slide window
    for start in range(0, max(1, len(data) - CLIP_SAMPLES + 1), STEP):
        clip = data[start:start + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES:
            clip = pad_or_trim(clip, CLIP_SAMPLES)
        out_name = OUT_DIR / f"{Path(bg).stem}_clip_{start}.wav"
        # Optionally apply very small random gain here to increase variety (not required)
        sf.write(str(out_name), clip, SR)
        total_created += 1

print(f"Created {total_created} silence clips in {OUT_DIR}")
