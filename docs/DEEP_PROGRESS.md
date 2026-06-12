# Toward 90% MNIST in-circuit (sparse multi-layer) — progress log

Goal: a sparse multi-layer net that LEARNS BY ITSELF in-circuit (SGD + noise) to 90% on 10-class
MNIST, proven in ngspice/Xyce/Spectre.

## Done
1. **Topology found & acc/MOSFET-optimized** (PyTorch, `build_sparse.py`): 2x2 stride-2 PYRAMID,
   locally-connected (non-weight-shared), NO conv, NO pooling, every neuron fan-in<=4 & fan-out<=4.
   - 8x8 in, ch 1->4->16->64: 808 w / 202 neurons / **91.2%** / ~13k MOS (MOSFET-optimal >90%)
   - 16x16, ch 2,4,12,36: 1144 w / **90.2% under plain SGD+noise+rails** / ~18k MOS
   - 16x16, ch 2,8,32,128: 93.2% / ~33k MOS.  (4x growth wasteful; ~2x is the sweet spot.)
   Constraint math: fan-out<=4 => Nout<=Nin per layer => channels grow <=4x per 2x2 step.
   Readout = sparse (each of 10 classes reads 4 of the top features); the pyramid ends 1x1xK.

2. **Multi-layer in-circuit backprop TRAINS in ngspice** (`gen_deep.py`): N-layer generalization of
   the proven gen_mc cells — sparse synapses + tanh keystone neurons, transpose-read backprop with
   per-hidden-layer common-mode subtraction (CMSUB) + tanh' gate, OTA cap updates, OCMSUB output.
   - **Depth-2** (4x4 -> 2x2x4 -> 1x1x16 -> 3), 3-class MNIST, real ngspice (2128 MOS):
     trains to **~59% train acc (chance 33%), all 3 classes alive**, stable. Proves multi-layer
     sparse backprop works in silicon (extends the depth-1 result [[backprop-hidden-layer-works]]).
   - Accuracy is modest: the 2-stage backward is less faithful than depth-1 (the negative-delta
     buffer `dnb` uses a crude resistor inverting-mirror around vrefd -- a known tuning target),
     and the 16-feature 4x4 net is tiny.

## Simulator situation (benchmarked on 1827-MOS C=10 deck, 400 slots)
- **ngspice: 16s** -- fastest at this scale; the workhorse.
- Xyce: blocked by UIC-from-.ic convergence ("max failures at time 0"); no-UIC does a DCOP that is
  slow AND destroys the stored cap weights. A background agent is fixing this (KLU could scale to the
  full net far better than ngspice if UIC converges).
- Spectre (+mt): ~30x slower than ngspice here, then segfaults. Not useful at this size.

## Open / next
- Improve the depth-2 backward fidelity (proper differential negative-delta buffer, per-layer LR).
- Scale to the full 8x8 pyramid (depth-4) -- gated by compute; needs the Xyce fix or layer-wise
  training. Full-net training to 90% in transient SPICE is hours-days; plan = prove the mechanism at
  each depth + cross-check vs the analog-faithful model (`build_sparse.py` SGD+noise = 90.2%), and
  report the 90% as model-projected with the ngspice multi-layer-training result as silicon evidence.
- Label everything ngspice-proven vs model-projected [[spice-vs-surrogate-honesty]].
