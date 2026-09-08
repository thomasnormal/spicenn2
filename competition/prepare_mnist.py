#!/usr/bin/env python3
"""Prepare the draft benchmark from the four official MNIST IDX gzip files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np

# Standard MNIST archive checksums, also published by torchvision's MNIST loader:
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py
ARCHIVES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, checksum in ARCHIVES.items():
        path = directory / name
        if path.exists():
            raw = path.read_bytes()
        else:
            print(f"Downloading {name}", flush=True)
            with urllib.request.urlopen("https://ossci-datasets.s3.amazonaws.com/mnist/" + name, timeout=60) as response:
                raw = response.read()
        if hashlib.md5(raw).hexdigest() != checksum:
            raise ValueError(f"MNIST archive checksum mismatch: {path}")
        if not path.exists():
            with path.open("xb") as output:
                output.write(raw)


def read_idx(path, dimensions):
    raw = gzip.decompress(path.read_bytes())
    if len(raw) < 4 or raw[:3] != b"\x00\x00\x08" or raw[3] != dimensions:
        raise ValueError(f"invalid unsigned-byte IDX header: {path}")
    end = 4 + 4 * dimensions
    shape = struct.unpack(">" + "I" * dimensions, raw[4:end])
    if not all(shape) or len(raw) - end != int(np.prod(shape)):
        raise ValueError(f"invalid IDX payload: {path}")
    return np.frombuffer(raw, dtype=np.uint8, offset=end).reshape(shape)


def preprocess(images):
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("expected 28x28 MNIST images")
    # Nonoverlapping 7x7 block averages: deliberately no per-image z-score.
    return (images.reshape(-1, 4, 7, 4, 7).mean(axis=(2, 4)) / 255).reshape(-1, 16)


def balanced(images, labels, count, rng, classes=range(10), normalize="raw"):
    if labels.shape != (len(images),) or np.any(labels > 9):
        raise ValueError("invalid MNIST labels")
    parts = []
    for label in classes:
        indices = np.flatnonzero(labels == label)
        if len(indices) < count:
            raise ValueError(f"not enough examples for class {label}")
        parts.append(rng.permutation(indices)[:count])
    selection = rng.permutation(np.concatenate(parts))
    x = preprocess(images[selection])
    if normalize == "zscore":
        x = np.clip((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6) * 0.25 + 0.5, 0, 1)
    return x, labels[selection].astype(np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("idx_directory", type=Path)
    parser.add_argument("output", type=Path, help="new .npz file")
    parser.add_argument("--train-per-class", type=int, default=24)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--normalize", choices=("raw", "zscore"), default="raw")
    parser.add_argument("--download", action="store_true", help="download and verify the four official archives")
    args = parser.parse_args()
    if min(args.train_per_class, args.test_per_class) <= 0 or args.seed < 0:
        parser.error("sample counts must be positive and seed nonnegative")
    if args.output.exists() or args.output.with_suffix(".json").exists() or args.output.suffix != ".npz":
        parser.error("choose a new .npz path (including an unused .json sidecar)")
    classes = [int(x) for x in args.labels.split(",")]
    if len(classes) < 2 or len(set(classes)) != len(classes) or any(x not in range(10) for x in classes):
        parser.error("choose at least two distinct labels from 0..9")
    if args.download:
        download(args.idx_directory)
    arrays, hashes = {}, {}
    rng = np.random.default_rng(args.seed)
    for split, prefix, expected, count in (("train", "train", 60000, args.train_per_class),
                                           ("test", "t10k", 10000, args.test_per_class)):
        images_path = args.idx_directory / f"{prefix}-images-idx3-ubyte.gz"
        labels_path = args.idx_directory / f"{prefix}-labels-idx1-ubyte.gz"
        images, labels = read_idx(images_path, 3), read_idx(labels_path, 1)
        if len(images) != expected:
            raise ValueError(f"expected official {split} split with {expected} images")
        arrays[f"{split}_x"], arrays[f"{split}_y"] = balanced(images, labels, count, rng, classes, args.normalize)
        for path in (images_path, labels_path):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with args.output.open("xb") as output:
        np.savez_compressed(output, **arrays)
    metadata = {"preprocessing": "28x28 uint8 -> 4x4 block mean / 255; row major",
                "normalization": args.normalize, "classes": classes,
                "seed": args.seed, "train_per_class": args.train_per_class,
                "test_per_class": args.test_per_class, "source_sha256": hashes,
                "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "note": "Public development data, not a secret final test set. Source authenticity must be verified by organizer."}
    with args.output.with_suffix(".json").open("x") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
