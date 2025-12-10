#!/usr/bin/env python3
"""
downsample_unknowns.py

Usage:
    python downsample_unknowns.py \
        --input train_augmented_n3000.json \
        --output train_augmented_n3000_downsampled.json \
        --max-unknown 3000
"""

import json
import argparse
import random
from pathlib import Path

# keyword list used by your project
KEYWORDS = ["yes","no","up","down","left","right","on","off","stop","go"]

def is_unknown_label(raw_label: str) -> bool:
    # treat both folder names and explicit "unknown" label
    if raw_label is None:
        return True
    rl = raw_label.strip()
    if rl in KEYWORDS:
        return False
    if rl == "_background_noise_" or rl == "silence":
        return False
    if rl.lower() == "unknown":
        return True
    # anything not in keywords and not background/silence => unknown
    return True

def main(input_path: str, output_path: str, max_unknown: int, seed: int):
    p = Path(input_path)
    items = json.loads(p.read_text(encoding="utf-8"))

    unknown_items = []
    non_unknown_items = []

    for it in items:
        raw_label = it.get("label", "")
        if is_unknown_label(raw_label):
            unknown_items.append(it)
        else:
            non_unknown_items.append(it)

    print(f"Before: total={len(items)}, non_unknown={len(non_unknown_items)}, unknown={len(unknown_items)}")

    random.seed(seed)
    if len(unknown_items) > max_unknown:
        unknown_items = random.sample(unknown_items, max_unknown)
        print(f"Downsampled unknowns to {max_unknown}")
    else:
        print("No downsampling required for unknowns")

    # combine and shuffle to avoid grouped ordering
    combined = non_unknown_items + unknown_items
    random.shuffle(combined)

    out_p = Path(output_path)
    out_p.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"After: total={len(combined)}, non_unknown={len(non_unknown_items)}, unknown={sum(1 for it in combined if is_unknown_label(it.get('label','')))}")
    print(f"Wrote: {out_p.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample 'unknown' entries in a dataset JSON file.")
    parser.add_argument("--input", "-i", required=True, help="Input JSON path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    parser.add_argument("--max-unknown", "-n", type=int, default=3000, help="Max unknown samples to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()
    main(args.input, args.output, args.max_unknown, args.seed)
