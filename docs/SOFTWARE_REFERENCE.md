# Software reference on the frozen MNIST tasks

This is a **NON-CIRCUIT software reference**, not an eligible analog submission.
It trains and predicts in Python with scikit-learn; it does not generate or
simulate a circuit. Energy is **unmeasured** and cannot be compared with the
analog energy reports. These accuracy results do not belong in the analog
leaderboard and are not analog learning results.

Measured on September 8, 2026:

| Frozen task | Training examples | Test examples | Correct | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| `mnist10-v0` | 240 | 500 | 328 | **65.6%** |
| `mnist017-v0` | 72 | 150 | 135 | **90.0%** |

This verifies approximately 66% ten-digit accuracy for the configuration below.
It establishes software classification performance on these inputs, not an
upper bound or a promise that a circuit can attain it. Regularization and input
scaling affect logistic regression; no regularization sweep was performed to
match that approximate figure. Both tasks used the same configuration, fixed
before evaluation, with no validation search or test-label tuning. These are
the public benchmark test splits, not a secret final test set.

## Data and input scaling

[software_reference.py](../competition/software_reference.py) loads the four
arrays `train_x`, `train_y`, `test_x`, and `test_y` and verifies their actual
content against the chosen frozen task manifest before fitting. It does not
trust an archive filename or its JSON sidecar. The content hash uses
`spicenn2-dataset-content-v1`, as implemented in
[data.py](../competition/data.py), rather than the compressed NPZ file hash.

| Task | Verified dataset content SHA-256 |
| --- | --- |
| `mnist10-v0` | `9c7fd4db02f655862f8f50a94bf331da47a77986e747a201d82f46d7dd4a07b0` |
| `mnist017-v0` | `7e615774a3a79e96bd5f50bf16ae2eda0b4b7521a47cb154d759ba7ac8f04a3e` |

Each example contains the frozen 16 features in `[0,1]`, cast to float64 with
**no additional scaling or centering**. Dataset preparation uses 7×7 block
averages of the original 28×28 MNIST image, divided by 255 and flattened in row
order. It then normalizes each image independently:

```text
x = clip((x - mean(x)) / (std(x) + 1e-6) * 0.25 + 0.5, 0, 1)
```

The standard deviation uses NumPy's default `ddof=0`. This normalization is
already present in the frozen arrays and is not repeated by the reference.
Both tasks use preparation seed 0, 24 training examples and 50 test examples
per class. The classifier fits only `train_x` and `train_y`; `test_y` is used
for the dataset integrity check and the final accuracy/confusion calculation.

## Fixed model and environment

The estimator is `sklearn.linear_model.LogisticRegression`, with multinomial
logistic loss, L2 regularization, and the following explicit configuration:

```python
LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    fit_intercept=True,
    class_weight=None,
    tol=1e-8,
    max_iter=10000,
    random_state=0,
    warm_start=False,
)
```

There are no sample weights, augmentation, feature selection, or fitted
preprocessing steps. `C=1.0` is the estimator's default inverse regularization
strength. Remaining scikit-learn 1.7.2 defaults are `dual=False`,
`intercept_scaling=1`, `l1_ratio=None`, `multi_class="deprecated"`, `n_jobs=None`,
and `verbose=0`. In this version, `lbfgs` uses multinomial loss for both tasks.
`random_state` is unused by this solver but recorded explicitly. Numerical
libraries are limited to one thread. A convergence warning fails the run;
these runs converged in 101 and 76 iterations, respectively.

Measured environment: CPython 3.11.13; NumPy 2.2.6; SciPy 1.15.3;
scikit-learn 1.7.2; joblib 1.5.2; threadpoolctl 3.6.0; Linux x86_64,
kernel `5.14.0-687.20.1.el9_8.x86_64`, glibc 2.34. The script pins all five
Python packages in its inline dependency metadata. These dependencies are
isolated from the public circuit runner's requirements.

## Reproduce

The optional reference uses `uv` to isolate its pinned dependencies from the
runner's virtual environment. If needed, install it with `python -m pip install uv`.
From the repository root, prepare new archives with the frozen settings:

```sh
reference_work=$(mktemp -d)
uv run --python 3.11 --with numpy==2.2.6 competition/prepare_mnist.py \
  "$reference_work/idx" "$reference_work/mnist10.npz" --download \
  --labels 0,1,2,3,4,5,6,7,8,9 --normalize zscore \
  --train-per-class 24 --test-per-class 50 --seed 0
uv run --python 3.11 --with numpy==2.2.6 competition/prepare_mnist.py \
  "$reference_work/idx" "$reference_work/mnist017.npz" --download \
  --labels 0,1,7 --normalize zscore \
  --train-per-class 24 --test-per-class 50 --seed 0

uv run --python 3.11 competition/software_reference.py \
  --task mnist10-v0 --data "$reference_work/mnist10.npz"
uv run --python 3.11 competition/software_reference.py \
  --task mnist017-v0 --data "$reference_work/mnist017.npz"
```

Existing archives with matching content hashes can be used directly.
Each command prints a JSON report containing the
hash, configuration, runtime versions, accuracy, and confusion matrix. The
script writes no submission, leaderboard entry, or circuit report.

The dataset guard tests need only NumPy:

```sh
uv run --python 3.11 --with numpy==2.2.6 \
  python -m unittest competition.test_software_reference
```

## Confusion matrices

Rows are true labels; columns are predicted labels. Every row contains 50
test examples. For `mnist10-v0`, both axes are ordered `0,1,2,3,4,5,6,7,8,9`:

```text
25  2  2  0  2  4  3  0 11  1
 0 37  1  7  0  1  0  0  2  2
 1  3 34  2  5  0  5  0  0  0
 0  1  5 37  0  1  0  3  1  2
 0  2  1  0 34  0  1  0  1 11
 1  1  0  7  7 11  4 10  8  1
 0  0  1  0  2  0 46  0  1  0
 0  0  1  2  0  0  0 43  0  4
 0  3  2  0  0  2  2  1 33  7
 0  0  0  0  7  1  1 11  2 28
```

For `mnist017-v0`, both axes are ordered `0,1,7`:

```text
47  2  1
 2 40  8
 0  2 48
```
