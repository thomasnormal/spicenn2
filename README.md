# spicenn2 — learning entirely in circuit dynamics

Analog CMOS networks whose **inference *and* training are physical circuit dynamics**, validated by
transistor-level Cadence Spectre transient simulation of complete training runs. Weights are capacitor
charge; neurons, synapses, error subtractors, and the weight-update multiply are small transistor cells;
the external controller only cycles data and bias/clock voltages. No behavioral elements anywhere in the
training or inference path.

## Headline results (all real Spectre, in-circuit training)

| Result | Number |
|---|---|
| Nonlinear 2D benchmarks (circles / rings / spirals), every seed | **> 0.95** (L-finals 0.99) |
| 8×8 digits, 4-class, sparse fan-in-4 net | **0.890** (backprop ideal 0.91–0.94) |
| 8×8 digits, 10-class — best (keystone: zero-sum + tuned PERAZ, **ngspice**) | **0.747** (all 10 classes alive, ideal 0.85) |
| 8×8 digits, 10-class (random features, no PERAZ) | 0.587 (deep-conv path underperforms this) |
| CIFAR-10 (color 8×8, in-circuit) | **0.164** (ceiling 0.41; gap is simulation-cost-bound) |
| Depth-4 pyramid via **amplitude-restoring error transport** (BKSIGN) | 0.55 → **0.84** |
| Component economy (fixed random hidden) | 0.74 @ −42% FETs / −80% caps |
| Label-free (masked-pixel PC, no labels in hardware) | 0.596 (chance 0.5) |
| VT-mismatch dose-response (σ = 0/2/5/10 mV) | 0.84 / 0.46 / 0.31 / chance |
| VT-mismatch **rescue** (bias + sign-update + freeze, 5-chip MC @2mV) | dead → **0.78 mean**, yield 1/3 → **5/5** |

The full worked/refuted experiment record is **[the experiment ledger](docs/IDEAS_LEDGER.md)** — every idea tried,
every number, including the failures.

## Repo map

| Start here | Contents |
| --- | --- |
| [pc_deep.py](pc_deep.py) | Main circuit learner: generate a deck, simulate, and score |
| [circuits/](circuits/README.md) | Reference cell circuits and worked XOR examples |
| [experiments/](experiments/README.md) | Historical trainers, generators, and calibration fixtures |
| [scripts/](scripts/README.md) | Experiment launchers and batch runs |
| [docs/](docs/README.md) | Findings, experiment ledger, figures, and saved result summaries |
| [fastsim/](fastsim/) | Device-level numerical surrogate and characterization tables |
| [paper/](paper/) | LaTeX manuscript sources; publication figures are in [figs/](figs/) |

The main learner is configured through environment variables; working configurations
are recorded in the ledger. The numerical surrogate under-predicts some real Spectre
training outcomes; see the paper's limitations.

## Reproducing a run

```bash
# depth-4 BKSIGN digits C=4 (the 0.84 result); needs Cadence Spectre on PATH
env TASK=digits C=4 LAYERS=64,32,16,8 BKSIGN=1 SOFTC=1 FANIN=4 NEP=64 NTR=40 NTE=25 \
    SEED=1 WSEED=1 SIM=spectre RUNTAG=demo python3 pc_deep.py
```

Each run writes `pd_<RUNTAG>.scs` (the full transistor deck), runs one transient covering the whole
training schedule, and prints the test-accuracy curve. Simulation artifacts (`raw_*/`, decks, logs) are
git-ignored — they are GBs per run.

Generated run files and downloaded datasets are not bundled in the current tree.
Reference circuits, figures and required calibration fixtures remain versioned;
see [repository hygiene](docs/REPOSITORY_HYGIENE.md) for details and regeneration notes.

## Provenance discipline

Numbers in the paper and ledger are labeled **real Spectre** vs **surrogate/ideal** throughout. PyTorch
"backprop ideal" numbers are topology calibrations, not circuit results.
