# src/data/dataset_tf.py  -- pure-TF dataset pipeline (no py_function)
from pathlib import Path
import json
import tensorflow as tf
import os
from src.config import map_raw_label, LABEL2INDEX, TARGET_LABELS

PROJECT_ROOT = Path(r"F:\Audio_classification")
TRAIN_JSON = PROJECT_ROOT / "data" / "splits" / "train_augmented_n3000.json"
VAL_JSON   = PROJECT_ROOT / "data" / "splits" / "val_files.json"

SR = 16000
TARGET_LEN = SR  # 1 second

# STFT / mel params (matching previous librosa choices)
N_FFT = 512
FRAME_LENGTH = 400   # 25 ms
FRAME_STEP = 160     # 10 ms
N_MELS = 40
FMIN = 20.0
FMAX = None  # will use sr/2 if None


"""def load_split(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    paths = []
    labels = []
    for it in items:
        raw = it["label"]
        tgt = map_raw_label(raw)
        idx = LABEL2INDEX[tgt]
        paths.append(it["path"])
        labels.append(idx)
    return paths, labels"""
import random
import numpy as np

def load_split(json_path):
    with open(json_path, "r") as f:
        items = json.load(f)

    paths = []
    labels = []
    for it in items:
        raw_label = it["label"]
        mapped = map_raw_label(raw_label)
        idx = LABEL2INDEX[mapped]
        paths.append(it["path"])
        labels.append(idx)

    # ---------- NEW: downsample "unknown" in TRAIN split ----------
    # We only change the training data, NEVER validation.
    json_name = os.path.basename(str(json_path))
    if "train_files" in json_name:
        unknown_idx = LABEL2INDEX["unknown"]

        samples = list(zip(paths, labels))
        unknown_samples = [s for s in samples if s[1] == unknown_idx]
        non_unknown_samples = [s for s in samples if s[1] != unknown_idx]

        MAX_UNKNOWN = 3000  # as decided: option B
        random.seed(42)     # for reproducibility

        if len(unknown_samples) > MAX_UNKNOWN:
            unknown_samples = random.sample(unknown_samples, MAX_UNKNOWN)

        samples = non_unknown_samples + unknown_samples
        random.shuffle(samples)

        paths, labels = zip(*samples)
        paths = list(paths)
        labels = list(labels)

        print(f"[load_split] After downsampling: "
              f"total={len(labels)}, unknown={sum(1 for y in labels if y==unknown_idx)}")
    # ---------------------------------------------------------------

    return paths, labels



def pad_or_trim_tf(audio):
    """audio: 1-D tf.float32 tensor, shape [samples]. Return length TARGET_LEN."""
    length = tf.shape(audio)[0]
    def trim():
        start = (length - TARGET_LEN) // 2
        return audio[start:start + TARGET_LEN]
    def pad():
        pad_len = TARGET_LEN - length
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        return tf.pad(audio, [[pad_left, pad_right]], "CONSTANT")
    return tf.cond(length > TARGET_LEN, trim, pad)


def waveform_to_log_mel(audio):
    """
    Input: audio 1-D tf.float32 shape [TARGET_LEN]
    Output: log-mel float32 tensor shape [T, N_MELS, 1] where T = time frames
    """
    # STFT
    stft = tf.signal.stft(
        signals=audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False
    )  # shape [T, freq_bins] complex64

    spectrogram = tf.abs(stft) ** 2  # power spectrogram, shape [T, freq_bins]
    num_spectrogram_bins = N_FFT // 2 + 1

    # Mel weight matrix
    fmax = FMAX if FMAX is not None else SR / 2.0
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SR,
        lower_edge_hertz=FMIN,
        upper_edge_hertz=fmax
    )  # shape [freq_bins, N_MELS]

    # spectrogram: [T, freq_bins], mel_matrix: [freq_bins, N_MELS]
    mel_spec = tf.matmul(spectrogram, mel_matrix)  # [T, N_MELS]

    # log scaling (use log with small offset)
    log_mel = tf.math.log(mel_spec + 1e-6)  # natural log; relative scale is fine
    # optionally, normalize per-example (zero-mean)
    mean = tf.reduce_mean(log_mel)
    std = tf.math.reduce_std(log_mel)
    log_mel = (log_mel - mean) / (std + 1e-6)

    # final shape -> (T, N_MELS, 1)
    log_mel = tf.expand_dims(log_mel, axis=-1)
    return log_mel


def parse_path_label(path, label):
    """
    path: tf.string path
    label: tf.int32
    returns: (log_mel_image, label)
    """
    # read file bytes and decode wav
    audio_bytes = tf.io.read_file(path)
    # tf.audio.decode_wav returns audio shape [samples, channels], sample_rate scalar
    audio_waveform, sample_rate = tf.audio.decode_wav(audio_bytes, desired_channels=1)
    audio_waveform = tf.squeeze(audio_waveform, axis=-1)  # [samples]

    # If sample_rate != SR, resample (tf.signal.resample not present in older TF).
    # We'll assume dataset is 16k (it is). If not, you can add tfio or custom resample.
    # Pad or trim to TARGET_LEN
    audio_waveform = pad_or_trim_tf(audio_waveform)

    # Normalize RMS to 1.0 (per example)
    rms = tf.sqrt(tf.reduce_mean(tf.square(audio_waveform)))
    audio_waveform = tf.cond(rms > 1e-9, lambda: audio_waveform / (rms + 1e-9), lambda: audio_waveform)

    # waveform -> log-mel
    log_mel = waveform_to_log_mel(audio_waveform)  # shape [T, N_MELS, 1]
    return log_mel, label


def make_dataset(paths, labels, batch_size=64, shuffle=True):
    paths = tf.constant(paths)
    labels = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    ds = ds.map(parse_path_label, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_train_val_datasets(batch_size=64):
    train_paths, train_labels = load_split(TRAIN_JSON)
    val_paths, val_labels = load_split(VAL_JSON)
    print("Train samples:", len(train_paths))
    print("Val   samples:", len(val_paths))
    train_ds = make_dataset(train_paths, train_labels, batch_size=batch_size, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels, batch_size=batch_size, shuffle=False)
    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = build_train_val_datasets(batch_size=32)
    print("TARGET_LABELS:", TARGET_LABELS)
    for batch_x, batch_y in train_ds.take(1):
        print("Batch X dtype:", batch_x.dtype)
        print("Batch X shape:", batch_x.shape)  # (batch, T, N_MELS, 1)
        print("Batch y shape:", batch_y.shape)
        print("Batch y sample:", batch_y[:10].numpy())
