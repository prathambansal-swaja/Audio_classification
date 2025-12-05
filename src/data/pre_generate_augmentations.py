#!/usr/bin/env python3
"""
Pre-generate augmented WAVs for the TRAIN split only and produce an updated train JSON.

Usage:
    python scripts/pre_generate_augmentations.py --train_json data/splits/train.json 
        --out_json data/splits/train_augmented_n3000.json --target_n 3000

Notes:
- Does NOT change validation.
- Does NOT augment 'unknown' label.
- Attempts to use project's labels if present; otherwise maps unknown labels to 'unknown'.
- Requires: numpy, soundfile, librosa, tqdm
"""
import argparse
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# ----- CONFIG -----
DEFAULT_TRAIN_JSON = "data/splits/train_files.json"
OUT_TRAIN_JSON = "data/splits/train_augmented_n{N}.json"
AUG_OUT_DIR = "data/augmented/train"  # augmented files written here, under per-label subfolders
TARGET_SR = 16000
TARGET_SECONDS = 1.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_SECONDS)
# Target label set used across your project (from earlier conversation)
TARGET_LABELS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "silence", "unknown"
]
# labels to skip augmentation (unknown)
SKIP_AUGMENT = {"unknown"}
# Background folder (optional)
BACKGROUND_DIR = "data/raw/train/train/audio/_background_noise_"
# Random seed for reproducibility
RANDOM_SEED = 1234

# Augmentation parameter ranges
SPEED_RANGE = (0.90, 1.10)         # time-stretch factor
GAIN_RANGE = (0.7, 1.3)            # amplitude multiplier
TIME_SHIFT_MS = (-100, 100)        # shift in milliseconds
SNR_DB_RANGE = (5.0, 20.0)         # when mixing background
NOISE_SNR_DB = (20.0, 35.0)        # additive noise SNR (higher = quieter noise)
# ------------------


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_wav_mono(path, sr=TARGET_SR):
    data, file_sr = sf.read(path, dtype="float32")
    # If multi-channel, average to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if file_sr != sr:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    return data


def write_wav(path, data, sr=TARGET_SR):
    # clip to [-1,1] safely
    data = np.asarray(data, dtype=np.float32)
    maxv = np.max(np.abs(data))
    if maxv > 1.0:
        data = data / (maxv + 1e-8)
    sf.write(path, data, sr, subtype="PCM_16")


def fix_length(arr, target_len=TARGET_SAMPLES):
    if arr.shape[0] > target_len:
        # center crop
        start = (arr.shape[0] - target_len) // 2
        return arr[start:start + target_len]
    elif arr.shape[0] < target_len:
        pad = target_len - arr.shape[0]
        left = pad // 2
        right = pad - left
        return np.pad(arr, (left, right), mode="constant")
    else:
        return arr


def time_shift(arr, ms_range=TIME_SHIFT_MS, sr=TARGET_SR):
    ms = random.randint(ms_range[0], ms_range[1])
    shift = int(sr * ms / 1000.0)
    return np.roll(arr, shift)


def change_speed(arr, factor):
    # librosa.effects.time_stretch expects float32 and > 1 frame length; use resample approach
    # We'll warp via resampling: resample to new length and then fix length
    if factor == 1.0:
        return arr
    new_len = int(round(len(arr) / factor))
    if new_len < 2:
        return arr
    resampled = librosa.resample(arr, orig_sr=TARGET_SR, target_sr=int(TARGET_SR * factor))
    # resample returned at different sr; resample back to TARGET_SR length
    resampled = librosa.resample(resampled, orig_sr=int(TARGET_SR * factor), target_sr=TARGET_SR)
    return resampled


def apply_gain(arr, gain):
    return arr * gain


def mix_background(clean, bg, snr_db):
    # adjust bg length and level to match clean and SNR
    clean = fix_length(clean, TARGET_SAMPLES)
    bg = fix_length(bg, TARGET_SAMPLES)
    # compute power
    eps = 1e-9
    p_clean = np.mean(clean ** 2) + eps
    p_bg = np.mean(bg ** 2) + eps
    target_p_bg = p_clean / (10 ** (snr_db / 10.0))
    if p_bg <= 0:
        return clean
    scale = np.sqrt(target_p_bg / p_bg)
    mixed = clean + bg * scale
    # normalize to avoid clipping
    maxv = np.max(np.abs(mixed))
    if maxv > 1.0:
        mixed = mixed / (maxv + 1e-8)
    return mixed


def add_noise(clean, snr_db):
    clean = fix_length(clean, TARGET_SAMPLES)
    p_clean = np.mean(clean ** 2) + 1e-9
    noise = np.random.randn(len(clean)).astype(np.float32)
    p_noise = np.mean(noise ** 2) + 1e-9
    target_p_noise = p_clean / (10 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_p_noise / p_noise)
    mixed = clean + noise
    maxv = np.max(np.abs(mixed))
    if maxv > 1.0:
        mixed = mixed / (maxv + 1e-8)
    return mixed


def choose_background(bg_files):
    if not bg_files:
        return None
    p = random.choice(bg_files)
    try:
        return read_wav_mono(p)
    except Exception:
        return None


def single_augment_pipeline(src_wav_path, bg_files):
    """
    Return augmented waveform (float32) of length TARGET_SAMPLES.
    Pipeline:
      - load
      - optionally speed change (30% chance)
      - time shift
      - gain
      - optionally mix background (50% chance if backgrounds exist)
      - optionally additive noise (30% chance)
      - fix length
    """
    try:
        wav = read_wav_mono(src_wav_path)
    except Exception as e:
        raise RuntimeError(f"Error reading {src_wav_path}: {e}")

    # ensure at least tiny length
    wav = fix_length(wav, TARGET_SAMPLES)

    # speed: small prob
    if random.random() < 0.3:
        factor = random.uniform(SPEED_RANGE[0], SPEED_RANGE[1])
        try:
            wav = change_speed(wav, factor)
        except Exception:
            pass

    # time shift
    wav = time_shift(wav)

    # gain
    gain = random.uniform(GAIN_RANGE[0], GAIN_RANGE[1])
    wav = apply_gain(wav, gain)

    # mix background sometimes
    if bg_files and random.random() < 0.6:
        bg = choose_background(bg_files)
        if bg is not None:
            snr = random.uniform(SNR_DB_RANGE[0], SNR_DB_RANGE[1])
            try:
                wav = mix_background(wav, bg, snr)
            except Exception:
                pass

    # additive noise sometimes
    if random.random() < 0.25:
        n_snr = random.uniform(NOISE_SNR_DB[0], NOISE_SNR_DB[1])
        wav = add_noise(wav, n_snr)

    wav = fix_length(wav, TARGET_SAMPLES)
    return wav


def compute_current_counts(train_items):
    counts = {lab: 0 for lab in TARGET_LABELS}
    for it in train_items:
        lab = it.get("label", None)
        if lab not in TARGET_LABELS:
            lab = "unknown"
        counts[lab] = counts.get(lab, 0) + 1
    return counts


def main(args):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_json = args.train_json
    target_n = args.target_n

    if not os.path.exists(train_json):
        raise FileNotFoundError(f"Train JSON not found: {train_json}")

    train_items = load_json(train_json)
    # train_items expected to be list of {"path": "...", "label": "..."} (common)
    counts = compute_current_counts(train_items)
    print("Current train counts (selected labels):")
    for lab in TARGET_LABELS:
        print(f"  {lab:8s} -> {counts.get(lab,0)}")

    # collect per-label source files (only from train)
    per_label_files = {lab: [] for lab in TARGET_LABELS}
    for it in train_items:
        p = it.get("path")
        lab = it.get("label")
        if lab not in TARGET_LABELS:
            lab = "unknown"
        per_label_files[lab].append(p)

    # background pool
    bg_files = []
    if os.path.isdir(BACKGROUND_DIR):
        for ext in ("*.wav", "*.flac", "*.mp3"):
            bg_files += list(map(str, Path(BACKGROUND_DIR).glob(ext)))
    bg_files = [str(x) for x in bg_files]
    if bg_files:
        print(f"Using {len(bg_files)} background files from {BACKGROUND_DIR}")
    else:
        print("No background files found; background mixing will be skipped.")

    # compute how many to create per label (skip unknown)
    to_create = {}
    total_new = 0
    for lab in TARGET_LABELS:
        if lab in SKIP_AUGMENT:
            to_create[lab] = 0
            continue
        cur = counts.get(lab, 0)
        need = max(0, target_n - cur)
        to_create[lab] = need
        total_new += need

    print(f"Target per-class N = {target_n}")
    print("Will create:")
    for lab in TARGET_LABELS:
        print(f"  {lab:8s} -> create {to_create[lab]}")
    print(f"Total new files to create: {total_new}")

    # prepare output dirs
    for lab in TARGET_LABELS:
        d = os.path.join(AUG_OUT_DIR, lab)
        ensure_dir(d)

    new_items = []
    progress = tqdm(total=total_new, desc="Generating augmented WAVs", unit="file")
    for lab, need in to_create.items():
        if need <= 0:
            continue
        src_list = per_label_files.get(lab, [])
        if len(src_list) == 0:
            print(f"Warning: no source files for label {lab}. Skipping.")
            continue
        for i in range(need):
            src = random.choice(src_list)
            try:
                augmented = single_augment_pipeline(src, bg_files)
            except Exception as e:
                print(f"Skipping augmentation for {src} due to error: {e}")
                progress.update(1)
                continue
            fname = f"{Path(src).stem}_aug_{uuid.uuid4().hex[:8]}.wav"
            out_path = os.path.join(AUG_OUT_DIR, lab, fname)
            write_wav(out_path, augmented, sr=TARGET_SR)
            # record item: keep same label name
            new_items.append({"path": out_path.replace("\\", "/"), "label": lab})
            progress.update(1)
    progress.close()

    # create new train JSON combining original train items + new_items
    out_train_json = OUT_TRAIN_JSON.format(N=target_n) if args.out_json is None else args.out_json
    combined = []
    # keep original items unchanged
    for it in train_items:
        combined.append({"path": it.get("path").replace("\\", "/"), "label": (it.get("label") if it.get("label") in TARGET_LABELS else "unknown")})
    for it in new_items:
        combined.append(it)

    write_json(out_train_json, combined)
    print(f"Wrote augmented train JSON to: {out_train_json}")
    # print final counts
    final_counts = compute_current_counts(combined)
    print("Final train counts after augmentation:")
    for lab in TARGET_LABELS:
        print(f"  {lab:8s} -> {final_counts.get(lab,0)}")
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", default=DEFAULT_TRAIN_JSON, help="Path to train JSON list")
    p.add_argument("--out_json", default=None, help="Output train JSON path (if omitted, uses data/splits/train_augmented_n{N}.json)")
    p.add_argument("--target_n", type=int, default=3000, help="Target samples per class (train only)")
    args = p.parse_args()
    if args.out_json is None:
        args.out_json = OUT_TRAIN_JSON.format(N=args.target_n)
    main(args)
