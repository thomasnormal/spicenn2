# Historical experiments

This directory contains the research trainers, deck generators, scoring programs,
and launch scripts. The current main learner is [pc_deep.py](../pc_deep.py).

The canonical multiclass helpers are `gen_mc.py`, `gen_mc_infer.py`, `score_mc.py`,
and `spectre_mc.py`; their former root-level duplicates have been removed from Git.
Invoke a helper by its path, for example `python3 experiments/gen_mc.py`, from the
repository root. It writes generated decks and intermediate data into the working
directory. Shell launchers resolve helper paths from their own location.

For the historical MNIST recipe, first prepare the `mnist_raw.npz` cache expected
by `prep_mnist.py`, then run `bash experiments/run_mnist.sh` from the root.
See [MNIST results](../docs/MNIST_RESULTS.md) for the recipe's scope and limitations.
The dataset cache is not bundled. These scripts are preserved research workflows;
some depend on a previous training run, separately prepared datasets, or Spectre.

`fixtures/cal.npy` is the saved gain/clamp calibration used by several older
experiments. Those readers now resolve the fixture relative to the source file,
so moving the fixture does not require changing the caller's working directory.
Running `surrogate.py` still writes a new local `cal.npy`; replacing the saved
fixture with new calibration is a separate, deliberate update.

Longer batch launchers live in [scripts/](../scripts/README.md), circuit snapshots
in [circuits/](../circuits/README.md), and published plots and summaries in
[docs/](../docs/README.md).
