# Real-data check: sklearn digits

Dataset: sklearn load_digits (1797 x 8x8, 10 classes). Run through the circuit-faithful
surrogate (physical-ish forward, transpose-read backward, divisive output competition).

## Key results
- **Logistic-regression reference (PCA-16): 93.9%** - the data/split is clean and easily separable.
- **64-dim raw -> surrogate caps ~56% and UNDERFITS.** With 64 inputs all injecting current into
  each column, the virtual ground rises and the per-synapse multiply compresses: **high fan-in
  squashes signal dynamic range** (a real circuit effect, and outside the XOR-era calibration of
  fan-in ~3). A real chip would need a front-end / limited fan-in.
- **PCA-16 -> surrogate ~69-72%** (divisive, lightly tuned, full-batch GD; still underfitting,
  more epochs / a better optimizer would close some of the ~25-pt gap to logreg).
- **One-vs-all (no competition) = chance (~10%) on 10 classes; divisive = ~69%.** This is the
  strongest confirmation of the multi-class finding: on a real 10-class task the normalization
  stage is not optional - independent outputs fail outright.

## Honest limits surfaced
- The analog forward costs ~25 points vs an ideal linear classifier (neuron nonidealities:
  dead-zone, smooth ReLU^2, clamp, fan-in compression, divisive approximation) - the same
  forward-fidelity ceiling that limits depth.
- High fan-in compression means wide input layers need a front-end (PCA / pooling / limited fan-in).
- The col-solve forward is expensive at 64-dim+; full-batch GD underfits in the epochs affordable here.
- Not yet run on digits (budget/speed): softmax norm, and the depth/isometry comparison.
