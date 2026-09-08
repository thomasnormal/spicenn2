"""Validated benchmark arrays and a version-independent content identifier."""
import hashlib
import json

import numpy as np


KEYS = ("train_x", "train_y", "test_x", "test_y")


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        missing = set(KEYS) - set(archive.files)
        if missing:
            raise ValueError("dataset is missing arrays: " + ", ".join(sorted(missing)))
        arrays = {key: archive[key].copy() for key in KEYS}
    for split in ("train", "test"):
        x, y = arrays[f"{split}_x"], arrays[f"{split}_y"]
        if (x.ndim != 2 or x.shape[1] != 16 or len(x) == 0
                or y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer)
                or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x)
                or not np.isfinite(x).all() or np.any((x < 0) | (x > 1))
                or np.any((y < 0) | (y > 9))):
            raise ValueError(f"invalid {split} data; expected X Nx16 in [0,1], integer Y in 0..9")
    return arrays


def content_hash(arrays):
    """SHA-256 of ordered names, shapes, and little-endian f64/i64 array values.

    Archive compression, ZIP metadata, byte order, memory layout, and the original
    label integer width do not affect this identifier. Different floating-point
    values *do* change it: this is not an approximate equivalence test.
    """
    digest = hashlib.sha256(b"spicenn2-dataset-content-v1\n")
    for key in KEYS:
        dtype = "<f8" if key.endswith("_x") else "<i8"
        array = np.array(arrays[key], dtype=dtype, order="C", copy=True)
        if key.endswith("_x"):
            array[array == 0] = 0  # Canonicalize signed zero.
        header = json.dumps([key, dtype, list(array.shape)], separators=(",", ":"))
        digest.update(header.encode("ascii") + b"\n")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
