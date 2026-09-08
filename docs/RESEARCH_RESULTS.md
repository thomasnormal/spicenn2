# Earlier research results

These are historical results from the research scripts, using different tasks and
configurations from the competition tutorial. Consult the [ledger](IDEAS_LEDGER.md)
and individual findings documents for the experimental details.

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
