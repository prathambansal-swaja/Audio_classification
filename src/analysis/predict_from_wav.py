from src.training import train_and_export as tae
import numpy as np
import tensorflow as tf
import os
if __name__ == "__main__":
    # Run training, export, and demo prediction
    h5='experiments/run_1765179265/checkpoints/best_model.h5'
    wav_example = r"F:/Audio_classification/data/raw/test/test/audio/clip_00a6a875c.wav"
    print("Predict:", tae.predict_from_wav(h5, wav_example))
