#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.2.6",
#   "scipy==1.15.3",
#   "scikit-learn==1.7.2",
#   "joblib==1.5.2",
#   "threadpoolctl==3.6.0",
# ]
# ///
"""NON-CIRCUIT software reference; ineligible for the analog leaderboard.

Run with ``uv run --python 3.11 competition/software_reference.py --help``.
No circuit is generated or simulated, and energy is not measured.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import warnings

import numpy as np

try:
    from .data import content_hash, load_data
except ImportError:
    from data import content_hash, load_data


TASK_DIRECTORY = Path(__file__).resolve().parent / "tasks"
TASKS = ("mnist10-v0", "mnist017-v0")
# Fixed before evaluation; there is deliberately no hyperparameter CLI or search.
MODEL_PARAMETERS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "fit_intercept": True,
    "class_weight": None,
    "tol": 1e-8,
    "max_iter": 10000,
    "random_state": 0,
    "warm_start": False,
}


def verified_dataset(task_id, data_path):
    """Check the actual array content against the frozen task manifest."""
    if task_id not in TASKS:
        raise ValueError("expected one of the frozen tasks: " + ", ".join(TASKS))
    manifest = json.loads((TASK_DIRECTORY / (task_id + ".json")).read_text())
    arrays = load_data(data_path)
    digest = content_hash(arrays)
    if digest != manifest["dataset_sha256"]:
        raise ValueError(
            f"dataset content hash does not match {task_id}: "
            f"expected {manifest['dataset_sha256']}, got {digest}"
        )
    return arrays, manifest, digest


def evaluate(task_id, data_path):
    # Verify before fitting; sidecar metadata and archive filenames are not trusted.
    arrays, manifest, digest = verified_dataset(task_id, data_path)
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from threadpoolctl import threadpool_limits

    train_x = np.asarray(arrays["train_x"], dtype=np.float64)
    test_x = np.asarray(arrays["test_x"], dtype=np.float64)
    model = LogisticRegression(**MODEL_PARAMETERS)
    # lbfgs uses multinomial loss for these multiclass tasks in sklearn 1.7.2.
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train_x, arrays["train_y"])
        predictions = model.predict(test_x)
    labels = manifest["preparation"]["labels"]
    correct = int(np.count_nonzero(predictions == arrays["test_y"]))
    return {
        "kind": "non-circuit-software-reference",
        "eligible_analog_submission": False,
        "energy": {"measured": False, "comparable_to_analog_energy": False},
        "task": task_id,
        "dataset_sha256": digest,
        "dataset_hash_format": "spicenn2-dataset-content-v1",
        "dataset_preparation": manifest["preparation"],
        "input_scaling": "Frozen train_x/test_x values in [0,1], float64; no additional scaling",
        "training": "Only train_x/train_y; no validation search or test-label tuning",
        "model": "sklearn.linear_model.LogisticRegression",
        "loss": "multinomial logistic loss with L2 regularization",
        "parameters": model.get_params(),
        "numerical_threads": 1,
        "iterations": model.n_iter_.tolist(),
        "versions": {
            "python": platform.python_version(),
            **{package: version(package) for package in
               ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        },
        "platform": platform.platform(),
        "train_samples": len(train_x),
        "test_samples": len(test_x),
        "feature_count": train_x.shape[1],
        "labels": labels,
        "correct": correct,
        "accuracy": correct / len(test_x),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            arrays["test_y"], predictions, labels=labels
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(args.task, args.data)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
