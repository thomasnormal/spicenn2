# What belongs in Git

Keep code, experiment launchers, documentation, paper sources and figures, a small
set of reference circuits, and calibration fixtures needed by the checked-in code.
Generated simulations and downloaded datasets belong in the local working directory,
not in commits.

The cleanup removes generated files from Git's index while preserving local copies.
It does not rewrite history. Old artifacts remain recoverable from earlier commits;
normal full clones will still download that history. For a small initial checkout:

```bash
git clone --depth 1 https://github.com/thomasnormal/spicenn2.git
```

## Retained assets

- Reference `.cir` decks live in `circuits/`. These
  include cell-characterization examples, the documented XOR trainer and inference
  decks, and the Xyce inference example. Many other decks are generated parameter
  sweeps or individual runs, so a `.cir` suffix alone does not identify source code.
- `fastsim/` retains its characterization decks and numerical fixtures. In particular,
  `devmodel.py` reads `char.npz`, and `devnet.py` reads `devnet_tables.npz`,
  `cmld_tables.npz`, `gprod_grid.npz` and `gprod_gridCM.npz`. These are small reference
  inputs rather than downloaded training datasets. `experiments/fixtures/cal.npy` is also retained for
  the historical deep-network experiment scripts.
- Documentation figures in `docs/figures/`, `figs/*.pdf`, compact published result summaries in
  `docs/results/`, and paper
  `.tex` sources remain tracked. Compiled `paper/paper.pdf` and `position.pdf` do not.
- Competition circuits belong in `competition/examples/` or `submissions/<name>/`.
  Only `entry.json`, `report.json`, and `verification.json` under
  `competition/results/<task>/<entry>/` are retained for verified leaderboard
  records. Full traces, harness output directories, and downloaded arrays stay local.

## Regenerating local artifacts

`pc_deep.py` generates its simulation deck and traces when run with the configuration
in the main README. Historical scripts under `experiments/` generate their own decks,
weight snapshots and inference outputs. Some expect files from an earlier training
step or separately prepared data; they are research scripts, not a uniform command-line
interface. Shell launchers resolve their source scripts relative to the checkout;
the former root-level `gen_mc.py`, `gen_mc_infer.py`, `score_mc.py` and `spectre_mc.py`
were exact duplicates of the canonical files in `experiments/`. References in old findings documents to specific run files describe those
experiments and do not imply that every generated file is distributed in a fresh clone.

Download MNIST/CIFAR datasets separately into `data/` or the paths expected by the
relevant script. They are no longer bundled in Git. Build the paper from its `.tex`
source, for example with `tectonic paper/paper.tex`.

## Before committing

```bash
git status --short
# This should print nothing: tracked paths that are now classified as ignored.
git ls-files --cached --ignored --exclude-standard
```

Python caches, build directories, assistant runtime state, simulation traces and
solver state files, downloaded datasets, generated root decks, and weight snapshots
are ignored. Avoid `git add -f` for generated artifacts. To retain a new reference
fixture, document its purpose and add a narrow exception to `.gitignore`, or place
authored circuit sources in a dedicated source/example directory.

Do not use `git clean -fdx` to tidy a research checkout: ignored datasets and valuable
uncommitted simulation results may exist there. Ignoring a file is not permission to
delete it from disk.
