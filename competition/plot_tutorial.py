#!/usr/bin/env python3
"""Regenerate the tutorial figures (requires matplotlib, separate from runner)."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prepare_toy import dataset
from prepare_mnist import read_idx, preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("idx_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 12})
    toy = dataset()
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for c, color in enumerate(("#0969da", "#bc4c00")):
        points = toy["test_x"][toy["test_y"] == c]
        ax.scatter(points[:, 0], points[:, 1], c=color, label=f"Class {c}", s=28, alpha=0.8)
    ax.set(xlabel="Input 0", ylabel="Input 1", xlim=(0, 1), ylim=(0, 1),
           title="First task: classify two clouds of points")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output_directory / "tutorial_blobs.svg")
    plt.close(fig)
    images = read_idx(args.idx_directory / "t10k-images-idx3-ubyte.gz", 3)
    labels = read_idx(args.idx_directory / "t10k-labels-idx1-ubyte.gz", 1)
    chosen = np.array([images[np.flatnonzero(labels == c)[0]] for c in (0, 1, 7)])
    small = preprocess(chosen)
    normalized = np.clip((small - small.mean(1, keepdims=True)) / (small.std(1, keepdims=True) + 1e-6) * 0.25 + 0.5, 0, 1)
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4), layout="constrained")
    for i, label in enumerate((0, 1, 7)):
        axes[0, i].imshow(chosen[i], cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(f"Digit {label}")
        axes[1, i].imshow(normalized[i].reshape(4, 4), cmap="gray", vmin=0, vmax=1)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("MNIST: 28×28")
    axes[1, 0].set_ylabel("Circuit inputs: 4×4")
    fig.savefig(args.output_directory / "tutorial_mnist.svg")
    plt.close(fig)
    for name in ("tutorial_blobs.svg", "tutorial_mnist.svg"):
        path = args.output_directory / name
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
