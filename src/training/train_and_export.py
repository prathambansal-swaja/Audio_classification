# src/training/train_and_export.py
import os
from pathlib import Path
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks, optimizers

# Import your dataset builder (pure-TF pipeline)
from src.data import dataset_tf

# Config / paths
PROJECT_ROOT = Path(r"F:\Audio_classification")
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / f"run_{int(time.time())}"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
LOG_DIR = EXPERIMENT_DIR / "logs"
MODEL_H5 = EXPERIMENT_DIR / "model.h5"
MODEL_TFLITE = EXPERIMENT_DIR / "model.tflite"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Training hyperparams
BATCH_SIZE = 32
EPOCHS = 10
INITIAL_LR = 1e-3
INPUT_SHAPE = (98, 40, 1)  # (T, n_mels, 1) — matches dataset_tf STFT params

# Load class weights (you computed them earlier). We'll read from the JSON output if you saved it;
# otherwise you can compute them inline. Here we'll compute again from splits to be sure they match.
from src.config import TARGET_LABELS, LABEL2INDEX, map_raw_label

"""def compute_class_weights_from_split():
    import json
    from collections import Counter
    train_json = PROJECT_ROOT / "data" / "splits" / "train_files.json"
    with open(train_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    c = Counter()
    for it in items:
        mapped = map_raw_label(it["label"])
        c[LABEL2INDEX[mapped]] += 1
    counts = np.array([c[i] for i in range(len(TARGET_LABELS))], dtype=np.float32)
    inv = 1.0 / counts
    weights = inv / inv.sum() * len(TARGET_LABELS)
    # Keras expects dict {class_index: weight}
    return {i: float(weights[i]) for i in range(len(TARGET_LABELS))}

CLASS_WEIGHTS = compute_class_weights_from_split()
# --- adjust silence weight (index 10) to avoid over-prediction ---
# ensure index exists
silence_idx = 10
if silence_idx in CLASS_WEIGHTS:
    CLASS_WEIGHTS[silence_idx] = min(float(CLASS_WEIGHTS[silence_idx]), 1.0)
else:
    CLASS_WEIGHTS[silence_idx] = 1.0
print("CLASS_WEIGHTS:", CLASS_WEIGHTS)"""
def compute_class_weights_from_split(tolerance_ratio=0.01):
    """
    Compute class weights from TRAIN split JSON.
    - If all class counts are within `tolerance_ratio` of mean count -> return None (balanced).
    - Protects against zero counts (replace zeros with 1 when computing).
    - Returns either None or dict {class_index: float(weight)} suitable for Keras.
    """
    import json
    from collections import Counter

    train_json = PROJECT_ROOT / "data" / "splits" / "train_augmented_n3000.json"
    with open(train_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    c = Counter()
    for it in items:
        mapped = map_raw_label(it["label"])
        c[LABEL2INDEX[mapped]] += 1

    counts = np.array([c[i] for i in range(len(TARGET_LABELS))], dtype=np.float32)

    # If any class has zero samples, replace with 1 to avoid division by zero (will produce very large weight,
    # but better than runtime error). We also handle balanced-case below.
    counts_safe = np.where(counts <= 0.0, 1.0, counts)

    # Detect near-balanced dataset: if every count is within tolerance_ratio (e.g. 1%) of mean -> treat balanced
    mean = np.mean(counts_safe)
    max_dev_ratio = np.max(np.abs(counts_safe - mean) / (mean + 1e-9))
    if max_dev_ratio <= tolerance_ratio:
        # Balanced enough — do not use class_weight (let dataset be uniformly sampled)
        print(f"[compute_class_weights_from_split] dataset appears balanced (max_dev_ratio={max_dev_ratio:.4f}), "
              "skipping class_weight (will pass None).")
        return None

    # Otherwise compute inverse-frequency weights (normalized to sum = n_classes)
    inv = 1.0 / counts_safe
    weights = inv / inv.sum() * len(TARGET_LABELS)

    # Make sure weights are finite and reasonable
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)

    return {i: float(weights[i]) for i in range(len(TARGET_LABELS))}

# --- call it once during import ---
CLASS_WEIGHTS = compute_class_weights_from_split(tolerance_ratio=0.01)
# If CLASS_WEIGHTS is a dict we can optionally cap silence weight:
if isinstance(CLASS_WEIGHTS, dict):
    silence_idx = 10
    if silence_idx in CLASS_WEIGHTS:
        CLASS_WEIGHTS[silence_idx] = min(float(CLASS_WEIGHTS[silence_idx]), 1.0)
print("CLASS_WEIGHTS:", CLASS_WEIGHTS)


# ---- Model definition (compact, effective 2D-CNN) ----
def build_model(input_shape=INPUT_SHAPE, n_classes=len(TARGET_LABELS), dropout=0.3):
    inp = layers.Input(shape=input_shape, name="log_mel_input")  # (98,40,1)
    x = inp

    def conv_block(x, filters, kernel=(3,3), pool=(2,2)):
        x = layers.Conv2D(filters, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(pool)(x)
        return x

    x = conv_block(x, 32, kernel=(3,3), pool=(2,2))   # -> (49,20)
    x = conv_block(x, 64, kernel=(3,3), pool=(2,2))   # -> (24,10)
    x = conv_block(x, 128, kernel=(3,3), pool=(2,1))  # -> (12,10)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(256, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inp, out)
    return model


# ---- Training loop ----
def train_and_save():
    # Build datasets
    train_ds, val_ds = dataset_tf.build_train_val_datasets(batch_size=BATCH_SIZE)

    # Build model
    model = build_model()
    model.summary()

    # Compile
    opt = optimizers.Adam(learning_rate=INITIAL_LR)
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_accuracy"])

    # Callbacks
    ckpt_path = str(CHECKPOINT_DIR / "best_model.h5")
    cb_checkpoint = callbacks.ModelCheckpoint(ckpt_path, monitor="val_sparse_categorical_accuracy",
                                              save_best_only=True, save_weights_only=False)
    cb_early = callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy", patience=6, mode="max",
                                       restore_best_weights=True)
    cb_reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    cb_tb = callbacks.TensorBoard(log_dir=str(LOG_DIR))

    # Fit
    """model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[cb_checkpoint, cb_early, cb_reduce, cb_tb],
        class_weight=CLASS_WEIGHTS
    )"""
    fit_kwargs = dict(
    x=train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[cb_checkpoint, cb_early, cb_reduce, cb_tb],
     )
    if isinstance(CLASS_WEIGHTS, dict):
        fit_kwargs["class_weight"] = CLASS_WEIGHTS

    model.fit(**fit_kwargs)



    # Save final model (best weights already restored by EarlyStopping if used)
    model.save(MODEL_H5)
    print("Saved Keras model to:", MODEL_H5)

    # Also copy best checkpoint if different
    if os.path.exists(ckpt_path) and ckpt_path != str(MODEL_H5):
        try:
            tf.keras.models.load_model(ckpt_path)  # quick check
            # overwrite MODEL_H5 with best
            tf.keras.models.load_model(ckpt_path).save(MODEL_H5)
            print("Saved best checkpoint to MODEL_H5.")
        except Exception as e:
            print("Could not reload checkpoint:", e)

    return MODEL_H5


# ---- TFLite export ----
def export_tflite(h5_path, tflite_path):
    # Load Keras model
    model = tf.keras.models.load_model(h5_path)
    # Create a concrete function with a fixed input shape
    # Input shape: [1, T, n_mels, 1]
    @tf.function(input_signature=[tf.TensorSpec([1, *INPUT_SHAPE], tf.float32)])
    def serve(x):
        return model(x)

    # Convert
    converter = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()])
    # Optional optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    try:
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print("Saved TFLite model to:", tflite_path)
    except Exception as e:
        print("TFLite conversion failed:", e)


# ---- Inference helpers (offline & realtime) ----
def predict_from_wav(h5_path, wav_path):
    """
    Loads saved Keras model and predicts label for a single wav file.
    This reuses dataset_tf parsing operations for consistency.
    """
    # Load model
    model = tf.keras.models.load_model(h5_path)
    # Build a one-element TF dataset for the wav path (so preprocessing is identical)
    ds = tf.data.Dataset.from_tensors((wav_path, 0)).map(dataset_tf.parse_path_label)
    ds = ds.batch(1)
    # Get model prediction
    for x, _ in ds.take(1):
        probs = model.predict(x)
        idx = int(np.argmax(probs[0]))
        label = TARGET_LABELS[idx]
        return label, float(np.max(probs[0]))


def predict_realtime(h5_path, record_seconds=1.0):
    """
    Simple real-time capture loop using sounddevice. Installs required package:
      pip install sounddevice numpy
    After capturing 1 second, it runs prediction and prints label.
    """
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError("sounddevice not installed. Run: pip install sounddevice") from e

    model = tf.keras.models.load_model(h5_path)
    sr = dataset_tf.SR
    print("Starting real-time prediction. Press Ctrl+C to stop.")
    try:
        while True:
            rec = sd.rec(int(record_seconds * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            # Save to a temporary WAV-like in-memory tensor and run same TF preprocessing
            # Convert numpy -> tf tensor and run dataset_tf.waveform_to_log_mel pipeline
            audio = tf.convert_to_tensor(rec.squeeze(), dtype=tf.float32)
            audio = dataset_tf.pad_or_trim_tf(audio)
            rms = tf.sqrt(tf.reduce_mean(tf.square(audio)))
            audio = tf.cond(rms > 1e-9, lambda: audio / (rms + 1e-9), lambda: audio)
            log_mel = dataset_tf.waveform_to_log_mel(audio)  # (T, n_mels, 1)
            x = tf.expand_dims(log_mel, axis=0)  # batch dim
            probs = model.predict(x)
            idx = int(np.argmax(probs[0]))
            label = TARGET_LABELS[idx]
            conf = float(np.max(probs[0]))
            print(f"Predicted: {label} ({conf:.3f})")
    except KeyboardInterrupt:
        print("Realtime loop stopped by user.")


if __name__ == "__main__":
    # Run training, export, and demo prediction
    h5 = train_and_save()
    export_tflite(h5, MODEL_TFLITE)

    # Example offline predict (uncomment and provide a wav path)
    # wav_example = r"F:\Audio_classification\data\raw\train\train\audio\yes\00176480_nohash_0.wav"
    # print("Predict:", predict_from_wav(h5, wav_example))

    # For realtime prediction, ensure sounddevice installed and microphone available:
    # pip install sounddevice
    # predict_realtime(h5)
