#!/usr/bin/env python3
"""Small two-blob classification task; no downloads or scikit-learn required."""
import argparse
from pathlib import Path
import numpy as np


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
    with args.output.open("xb") as output:
        np.savez_compressed(output, **dataset())
