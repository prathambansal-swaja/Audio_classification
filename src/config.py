# F:\Audio_classification\src\config.py

# Final target labels for the Kaggle competition
KEYWORDS = [
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go"
]

SPECIAL_LABELS = [
    "silence",
    "unknown"
]

TARGET_LABELS = KEYWORDS + SPECIAL_LABELS

# For convenience: map from label name to integer index for the model
LABEL2INDEX = {label: idx for idx, label in enumerate(TARGET_LABELS)}
INDEX2LABEL = {idx: label for label, idx in LABEL2INDEX.items()}

# Raw folder -> target label mapping
def map_raw_label(raw_label: str) -> str:
    """
    Map raw folder name to one of the 12 target labels.

    - If it's one of the 10 keywords, keep it.
    - If it's background noise, call it 'silence'.
    - Otherwise, call it 'unknown'.
    """
    if raw_label in KEYWORDS:
        return raw_label
    if raw_label == "_background_noise_":
        return "silence"
    return "unknown"
