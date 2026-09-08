#!/usr/bin/env python3
"""Small two-blob classification task; no downloads or scikit-learn required."""
import argparse
import json
from pathlib import Path
import numpy as np

try:
    from .data import content_hash
except ImportError:
    from data import content_hash


def dataset(seed=7):
    rng = np.random.default_rng(seed)
    data = {}
    for split, count in (("train", 20), ("test", 50)):
        labels = np.repeat(np.arange(2), count)
        x = np.full((2 * count, 16), 0.5)
        centers = np.array([[0.25, 0.75], [0.75, 0.25]])
        x[:, :2] = np.clip(centers[labels] + rng.normal(0, 0.07, (len(labels), 2)), 0, 1)
        order = rng.permutation(len(labels))
        data[f"{split}_x"], data[f"{split}_y"] = x[order], labels[order]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.suffix != ".npz" or args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("choose a new .npz path (including an unused .json sidecar)")
    try:
        arrays = dataset()
        with args.output.open("xb") as output:
            np.savez_compressed(output, **arrays)
        with args.output.with_suffix(".json").open("x") as output:
            json.dump({"dataset_sha256": content_hash(arrays), "dataset_hash_format": "spicenn2-dataset-content-v1",
                       "generator": "two-blobs-v1", "seed": 7, "train_per_class": 20, "test_per_class": 50}, output, indent=2)
            output.write("\n")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(args.output)
