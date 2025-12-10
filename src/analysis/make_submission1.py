import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.dataset_tf import parse_path_label, TARGET_LABELS

# Adjust these three if needed
MODEL_H5 = "experiments/run_1765179265/checkpoints/best_model.h5"
TEST_DIR = "data/raw/test/test/audio"
OUTPUT_CSV = "submission1.csv"
BATCH_SIZE = 256


def build_test_dataset(wav_paths, batch_size=BATCH_SIZE):
    """
    Build a tf.data.Dataset of test files using the SAME preprocessing
    as training (parse_path_label).
    """
    # Dummy labels (parse_path_label expects (path, label))
    dummy_labels = np.zeros(len(wav_paths), dtype=np.int64)

    ds = tf.data.Dataset.from_tensor_slices(
        (np.array([str(p) for p in wav_paths]), dummy_labels)
    )
    ds = ds.map(parse_path_label, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    # 1. Load model
    print(f"Loading model: {MODEL_H5}")
    model = tf.keras.models.load_model(MODEL_H5)

    # 2. Collect test wav files
    test_dir = Path(TEST_DIR)
    wav_paths = sorted(test_dir.glob("*.wav"))
    print(f"Found {len(wav_paths)} wav files in {test_dir}")

    if not wav_paths:
        raise SystemExit("No .wav files found in TEST_DIR")

    # 3. Build dataset with same pipeline as training
    ds = build_test_dataset(wav_paths)

    # 4. Run batched prediction
    print("Running batched prediction...")
    probs = model.predict(ds, verbose=1)  # shape: (N, 12)
    pred_indices = probs.argmax(axis=1)   # shape: (N,)
    pred_labels = [TARGET_LABELS[i] for i in pred_indices]

    # 5. Write submission.csv (fname, label)
    fnames = [p.name for p in wav_paths]  # like 'clip_000044442.wav'

    print(f"Writing predictions to {OUTPUT_CSV}")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fname", "label"])
        writer.writerows(zip(fnames, pred_labels))

    print(f"Done. Saved {len(fnames)} predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
