# In-circuit backprop through a hidden layer (16→9→C) — SOLVED

Goal (research): make **backprop** train a 2-layer net fully in ngspice — 4×4 inputs → 3×3 hidden
units, each a local overlapping 2×2 receptive field, → C classes — without the multi-class collapse.
Achieved: **8/8 seeds train MNIST-4×4 {0,1,7} to 72–85% (mean ≈79%), all classes alive.** `run_backprop.sh`.

This is genuine in-circuit backprop: forward, the transpose-read backward, and the weight-cap
updates all run physically in one `.tran`. The controller only streams pixels/targets as voltages.

## The investigation (each step was a measurement, not a guess)

The hidden layer kept collapsing to one class. Ruled causes IN and OUT by experiment:

1. **A linear 16→C map separates these triples at ~94%** (sklearn) → not a capacity problem.
2. **Backward fidelity is fine.** Froze random weights, logged the circuit's backward `d1pre_j`,
   compared to the ideal `Σ_c W2_cj(y_c−t_c)`: **mean cosine 0.923**, never negative. The
   transpose read is faithful — *not* the ~0.5-cosine eroder feared from the old depth task.
3. **Output training is fine.** Froze the hidden layer (random nonlinear features), trained only the
   output: **77%**, all classes. So the *output* delta rule works; the problem is hidden learning.
4. **Hidden learning is genuinely harmful, not just cold-start.** Warm-started from the good 77%
   solution and released the hidden layer → **degraded to 47%**. And it was independent of hidden
   learning-rate (RTbk and HWTW swept 12–16× → identical collapse) → a *directional/offset* problem,
   not magnitude.
5. **Root cause = backward COMMON-MODE.** Probed `d1pre_j − vrefd`: overall mean **+0.186 with every
   unit positive**, per-unit std only ~0.06 (common-mode is 2.4–3.3× the discriminative signal). It
   comes from the output error having a large class-mean: compressed keystone outputs (`y_c∈[0.1,0.6]`)
   sit above the mostly-zero one-hot targets, so `Σ_c(y_c−t_c) ≈ 1.0`. Through `d1pre_j =
   Σ_c W2_cj(y_c−t_c)` that shared offset drifts **every** hidden cap the same direction → all units
   move together → features degenerate → collapse. The numpy surrogate never saw this (its outputs
   reach the targets, so the error mean → 0) — which is exactly why surrogate-backprop trained but
   circuit-backprop didn't.

## The fix — subtract the backward common-mode (`CMSUB`)

`d1bar` = resistor-average of all `d1pre_j` ( = `vrefd` + mean signal ), used as the hidden-update
OTA reference instead of `vrefd`. Each unit then sees `(d1pre_j − d1bar)` = its own signal minus the
shared offset. Zero B-sources: a resistor star (`RCMB`) into a high-impedance node. `RCMB=10e3`
(tight averaging) made it reliable; the one seed that still collapsed at 20e3 trains at 84% at 10e3.

The built-in `COMP=subtractive` competition did **not** fix it (probed: common-mode 0.186→0.200) —
it shifts the forward error references but doesn't clean the transpose-read backward. `CMSUB` acts
directly on the backward nodes, which is where the damage happens.

## Two other necessary ingredients (orthogonal to CMSUB)

- **ACT=tanh** (bounded, center-biased differential pair). ReLU² units sit permanently above
  threshold (`uon_j` saturated, z1≈1–2 ≫ VMID) → never rectify → degenerate. tanh units swing both
  sides of VMID, exercising real nonlinearity. (Matches NEURON_FINDINGS: bounded saturator enables depth.)
- **NORM=zscore** inputs (per-image energy equalization) — removes the input-energy class asymmetry
  (see [MNIST_RESULTS.md]); needed for the output layer regardless of depth.

## Recipe (`run_backprop.sh`)

`CONN=rf3x3 H=9 ACT=tanh TW=32 VTB=1.4 VMID=0.5 BHID=0.3 OWTW=20 CMSUB=1 RCMB=10e3`, NORM=zscore,
~1400 slots, AVGW=20. Forward keystones + ReLU′-gated transpose-read backward + OTA cap updates.

8/8 seeds on {0,1,7}: 72.7, 78.7, 84.0, 72.0, 84.7, 82.0, 80.0, 80.7 %.

## Status / next

- The collapse wall is down: hidden-layer backprop trains reliably in real silicon-style SPICE.
- On this near-linearly-separable task the trained hidden layer roughly **matches** the random-feature
  ceiling (~77%), occasionally beating it (84–85%). To *demonstrate depth's value* (trained ≫ random)
  needs a task where one layer is insufficient — a genuinely nonlinear target. That, plus 3+ layers,
  is the next step now that the per-layer backward is trustworthy.
- Diagnostics retained: `FREEZE`/`PROBE` (gen_mc) + `probe_backward.py` (cosine fidelity), `FRZH`
  (random-feature ceiling), `surrogate_rf.py` (fast idea screening). `CMSUB`/`RCMB`/`HWTW`/`ACT`/`VMID` knobs.
