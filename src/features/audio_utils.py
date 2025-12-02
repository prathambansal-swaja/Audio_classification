# src/features/audio_utils.py
import os
import numpy as np
import soundfile as sf
import librosa

SR = 16000  # target sample rate for this dataset
TARGET_LEN = SR * 1  # 1 second -> 16000 samples


def load_wav(path, sr=SR):
    """Load audio file as float32 mono. Return waveform (1D numpy) and sample rate."""
    data, file_sr = sf.read(path, dtype='float32')
    # If stereo, convert to mono by averaging channels
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if file_sr != sr:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    return data.astype('float32'), file_sr


def pad_or_trim(wave, target_len=TARGET_LEN):
    """Pad with zeros or trim center to fixed length (target_len samples)."""
    if len(wave) > target_len:
        # trim center region (keeps signal in middle)
        start = (len(wave) - target_len) // 2
        return wave[start:start + target_len]
    elif len(wave) < target_len:
        pad_left = (target_len - len(wave)) // 2
        pad_right = target_len - len(wave) - pad_left
        return np.pad(wave, (pad_left, pad_right), mode='constant')
    else:
        return wave


def normalize_wave(wave, eps=1e-9):
    """Simple per-clip RMS normalization to unit RMS (helps model invariance to loudness)."""
    rms = np.sqrt(np.mean(wave ** 2))
    if rms < eps:
        return wave
    return wave / (rms + eps)


def compute_log_mel(wave, sr=SR, n_mels=40, n_fft=512, hop_length=160, win_length=400, fmin=20, fmax=None):
    """
    Compute log-mel spectrogram (shape: n_mels x time_frames).
    Defaults chosen for 1s @ 16k: frame hop 10ms (160), win 25ms (400).
    """
    # power spectrogram
    mel = librosa.feature.melspectrogram(
        y=wave,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    # convert to log scale (dB). Use top_db to limit dynamic range if desired.
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype('float32')


def preprocess_file_to_log_mel(path):
    """
    Full pipeline: load -> pad/trim -> normalize -> compute log-mel -> return array.
    Output shape: (n_mels, time_frames)
    """
    w, sr = load_wav(path)
    w = pad_or_trim(w, TARGET_LEN)
    w = normalize_wave(w)
    log_mel = compute_log_mel(w)
    return log_mel
