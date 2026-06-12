# Circuit-faithful surrogate — scaling & normalization study

A fast NumPy surrogate of the analog trainer, built to answer two questions that
ngspice is too slow to sweep: **which analog output-normalization works for many
classes**, and **does adding hidden neurons buy robustness**. The surrogate is a
study tool, not the learner — every conclusion here is meant to be re-checked in SPICE.

## What the surrogate models (and what it doesn't)

Faithful, device-level **forward**:
- LEVEL-1 NMOS conventional current (signed, handles reverse, GAMMA=0), constants taken
  straight from the SPICE `.model` cards.
- The synapse signed multiply with its **dead-zone** (Mp cuts off when the weight goes
  low → unit dies), exactly the failure we fought in the transistor deck.
- The current-difference **keystone** with the column virtual grounds **solved
  self-consistently** (the diode-NMOS sinks the injected current; col self-regulates).
  A fixed col is wrong — solving it dropped forward error 7.9 → 0.04.
- The quadratic **ReLU** mirror with activation clamp.
- One lumped fit constant: the current-mirror ratio `kgain = 0.64`.

**Backward** = the circuit's actual transpose-read rule: δ·Wᵀ·ReLU′ with the class sum
done by KCL at the transpose column. This is **not** the exact device Jacobian — it is
what the silicon computes, so the surrogate inherits the same approximate-gradient learning.

Not modeled (second-order, or deferred to SPICE): OTA update saturation, settling
latency, per-device mismatch beyond injected weight noise, temperature.

**Calibration check vs ngspice (XOR, H=6):** surrogate y = [0.26, 0.89, 0.60, 0.40] vs
ngspice [0.24, 1.04, 0.62, 0.36] — 4/4 correct, the right hidden units fire per pattern,
activations graded. Trustworthy at the level needed for these sweeps.

## Finding 1 — output normalization (the softmax replacement)

Error signal is always δ_c = p_c − t_c on the normalized outputs (no Jacobian through
the normalizer — that's what's buildable; it's exact only for softmax+CE).
Hard problem: C=8 classes, D=8, hidden=20, overlapping blobs. Best test acc over an lr sweep:

| scheme | best test | lr-robustness | analog realization |
|---|---|---|---|
| one-vs-all (none) | 35% | fragile (collapses above lr 0.2) | none — independent outputs |
| **divisive** (yᵢ/Σy) | **70%** | **robust across all lr** | shared-node current divide |
| **subtractive** (y−mean+1/C) | **78%** | fragile (low lr only) | **one shared common-mode amp** (cheapest) |
| softmax (exp/Σexp) | 80% | robust to lr 1.8 | subthreshold exp (regime + drift cost) |

- **Global normalization roughly doubles accuracy** on many classes (35 → 70–80%).
  One-vs-all's independent outputs are poorly calibrated for argmax and unstable.
- **You do not need softmax.** Cheap above-threshold competition captures nearly all of it:
  subtractive common-mode feedback (the single cheapest circuit, one shared amplifier)
  matches softmax to within ~2 points; divisive is a few points lower but the most
  lr-robust — attractive when lr can't be tuned precisely on-chip.
- **shiftmax (shared hard-max reference) fails at chance** with δ=p−t and was dropped.
  The hard max pins the winner's output to exactly 1, so a correct-and-winning class gets
  zero error and the margin collapses. It's the one option that *needs* a Jacobian through
  the max — exactly what we can't build cheaply.
- Implementation gotcha that bit us: a subtractive/feedback normalizer must target the
  right total. Offset 1/C (sum-to-one) works; a fixed 0.5 offset injects a common-mode
  error that grows with C (Σδ = C/2 − 1) and collapses the net for C>2.

## Finding 2 — width vs robustness

Divisive norm (lr-robust → clean comparison), vary hidden width, then perturb **all**
trained weights with Gaussian gate-voltage noise (σ in represented-weight units;
σ_gate = Gs·σ). Accuracy vs σ:

```
σ:        0    0.2   0.4   0.8   1.6   3.2
 8 hid:   60    57    54    47    32    20
24 hid:   65    66    65    57    38    20
48 hid:   69    69    68    61    45    24
```

- **More neurons → more accurate** (60 → 69%) **and lower seed variance** (8-unit nets are
  a lottery, 48–67% across seeds; 48-unit nets reliably ~70%).
- **More neurons → more graceful degradation.** At σ=1.6 the 48-unit net holds 45% where
  the 8-unit net has dropped to 32%. Dead/saturated hidden fractions stay flat (~15%/~35%)
  across width, so this is genuine **redundancy** (each weight matters less), not a change
  in the operating regime. This is the "more neurons → more robust" hypothesis, quantified.
- The decision is insensitive up to σ≈0.4 (median |w|≈1) — partly margins, partly the
  dead-zone/saturation acting like coarse quantization. The knee is ~σ 0.4–0.8.

## Finding 3 - depth: the transpose-read backward explodes

Same C=8 problem, divisive norm, add hidden layers. Per-layer gradient RMS at convergence:

```
1 hidden [20]      : grad-rms/layer [1.9, 0.04]                 -> test 68%
2 hidden [20,20]   : grad-rms/layer [1.5e4, 2.9, 0.02]          -> test 30%
3 hidden [20,20,20]: grad-rms/layer [1.2e11, 1.5e4, 2.0, 0.02]  -> test 29%
```

- The back-propagated error grows by a large, compounding factor every layer, because
  d.W^T has ||W^T||>1 (weights are O(1)) and nothing renormalizes the backward path. The
  first-layer gradient reaches 1e11 at three layers; its weights blow up and training
  diverges. **The naive transpose-read cannot train deep nets as-is** - the analog
  exploding-gradient problem made concrete (it explodes, weights>1; it does not vanish).
- **Cheap fixes do not rescue it** (tested on 2 layers): lower lr makes it WORSE
  (lr 0.01 -> 16%, not simple divergence), per-layer error normalization -> 27%, physical
  weight clamping [-1.5,1.0] -> 35%. All far below the 1-layer 68%.
- **Caveat:** blobs are near-linearly-separable, so this task gains nothing from depth -
  "depth hurts here" is not "depth is impossible." A task that needs hierarchy plus real
  backward gain control would be the fair test. But the optimization hazard is real.
- **Design takeaway:** a single hidden layer is the safe silicon bet. Spend transistors on
  WIDTH, not depth, unless you commit to backward-path gain control.

## Finding 4 - more output classes: competition's edge widens

Vary C with fixed net (D=10, hidden=24). Test accuracy (mean of 2 seeds):

| C | chance | one-vs-all | divisive | softmax |
|---|---|---|---|---|
| 4 | 25% | 91% | 96% | 99% |
| 8 | 12% | 37% | 80% | 91% |
| 12 | 8% | 19% | 74% | 86% |
| 16 | 6% | 12% | 66% | 83% |

- **One-vs-all collapses toward chance as C grows** (91->37->19->12%): each independent
  output sees only 1/C positives, learns to predict "not me," and argmax becomes noise.
- **Competition degrades gracefully**: softmax 99->83%, divisive 96->66%. The advantage over
  one-vs-all grows from ~8 points at C=4 to ~55-70 points at C=12-16.
- The more classes the chip must distinguish, the more the shared-budget competition is
  worth. Softmax holds best; divisive (cheap, lr-robust) still gives 66% at 16 classes.

## Finding 5 - depth, resolved with REAL backprop

The "deep nets explode (1e11)" result was chased to ground; the resolution has three layers.

**1. The explosion was largely a surrogate artifact.** The naive surrogate let weights and
updates run unbounded. The chip cannot: cap voltages are bounded (gp in [0,3] V, so w in
[-1.5,1]; the ngspice-trained XOR weights sit at gp in [0.57,2.84] V, i.e. w in [-1,0.9]) and
the update OTA is current-limited. Enforcing both, the 2-layer gradient profile drops from
[12250, 2.9, 0.02] to [0.46, 0.12, 0.02]. **Deep nets do not blow up on real hardware.**
(Honest cost: the physical clamp also lowers shallow accuracy from an inflated 69% to ~52-59%
- earlier studies used some unphysical gate-voltage headroom.)

**2. The residual gradient growth comes from ||W^T|| > 1, and the real-backprop cure is
dynamical isometry.** Even bounded, the true transpose gradient still grows with depth
(first-layer RMS 0.06 -> 0.51 -> 6.6 across 1-3 layers) because each layer multiplies by
||W^T|| > 1 (fan-in ~16, weights O(1)). A residual skip helps accuracy but does NOT flatten
this - the identity path adds the gradient without removing the multiplicative term. The fix
that keeps the *true* gradient norm-preserving is constraining each layer to ||W^T|| ~ 1:
**orthogonal cap initialization** (free at reset) **+ per-neuron fan-in normalization**
(a divisive feedback on each neuron's weight vector - the same primitive validated for output
competition) **+ residual skips**. All real backprop (the actual transposed weights), all
buildable. Result on blobs (C=8):

```
depth   naive transpose        real backprop + isometry
 1      72%  g0=0.055           64%  g0=0.070
 2      42%  g0=0.512           68%  g0=0.112
 3      25%  g0=6.6             67%  g0=0.183
 4      12%  g0=0.111           40%  g0=0.482
```

The gradient is tamed (g0 creeps 0.07 -> 0.48 over 1-4 layers, not exploding) and accuracy is
flat-to-rising through depth 3 (64 -> 68 -> 67%), with **2-3 hidden layers BEATING both
depth-1 and the shallow reference**. The cheap buildable proxy (per-neuron L2 fan-in norm
instead of exact spectral norm) gives 66%/64% at depths 2/3 - essentially the same. The
trainable-depth frontier moved from 1 to 3 layers using the real gradient.

(DFA - a fixed random feedback projection - also flattens the gradient and removes weight
transport, but it is NOT real backprop: the signal it sends is not the true gradient. It is
excluded here. The isometry result above uses the actual transposed weights.)

**3. The last limit at depth is the forward neuron, not the backward.** Even with the gradient
controlled, depth 4 still fades (67% -> 40%): stacking lossy analog neurons (dead-zone +
smooth ReLU^2 + activation clamp) destroys signal faster than depth adds value, and
depth-favoring tasks (high-frequency stripes) are exactly the ones the limited-dynamic-range
forward represents worst (a wide *shallow* net already struggles on them). **The lever for
going beyond ~3 layers is neuron fidelity** (a self-normalizing, higher-dynamic-range
activation), because the backprop machinery is no longer the bottleneck.

**Design takeaway:** depth through ~3 hidden layers is unblocked with REAL backprop, via
orthogonal init + per-neuron fan-in normalization + residual skips + physical bounds + leaky
comparator + gain control (all buildable, no weight-transport tricks). 2-3 layers beat shallow.
Going deeper is a forward-neuron design problem.

## Buildable recommendation → what to validate in SPICE

- **Normalization:** start with **subtractive common-mode feedback** (one shared amp, ~free,
  ~softmax accuracy) or **divisive** (most lr-robust). Skip softmax/subthreshold. Use a hard
  WTA only for the inference argmax.
- **Width:** scaling hidden width is worth the transistors — it buys accuracy, training
  reliability, *and* perturbation tolerance simultaneously.
- **Next (path B):** generalize the SPICE generator to a layer-based builder with the
  shared-negative-reference cell, the chosen competition stage, and multi-class one-hot
  targets; run a larger net in ngspice and spot-check that the surrogate's accuracy and
  the σ-knee hold up on the real device.

## Files
- `surrogate.py` — device model, self-consistent col solve, calibrated forward (XOR validation).
- `mc_experiment.py` — batched general net, transpose-read backward, normalization options, blob A/B.
- `run_sweep.py`, `run_width.py` — timeout-robust sweep runners (incremental disk writes).
- `results_hard.txt`, `results_width.txt`, `surrogate_results.json` — raw measurements.
- `surrogate_findings.png` — the two figures above.
