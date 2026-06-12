# Getting depth to work — investigation log (honest, with a retraction)

Goal: genuine **depth** (compositional credit assignment), not width, not shortcuts.

## Retraction: "local deep supervision" is NOT depth
First pass I trained each hidden layer with its own auxiliary head → the final target, and
reported it "made depth work" (2×2 checkerboard 90%). **That is not depth** — training every
layer to be directly target-decodable collapses the stack into *parallel 1-layer classifiers*;
no layer is learned to *serve the next layer's* computation. The tell: it dies the instant a
task actually requires composition (3×3 checkerboard → back to chance). Retracted.

## The benchmark problem (the real blocker right now)
Before claiming "analog depth works" we need a task where depth genuinely, *learnably* helps.
Tested with an **ideal** MLP (standard ReLU, exact backprop) so we measure the task, not the
circuit:

| task | ideal shallow | ideal wide-shallow | ideal deep | verdict |
|---|---|---|---|---|
| 2×2 checkerboard | 81% | – | 90% | it's just XOR — 1 hidden layer suffices, **not a depth task** |
| 3×3 / 4×4 checkerboard (w=8) | ~53% | – | ~55% (chance) | **no net solves it** at this width — bad benchmark |
| K-stripe 1D (capped width) | ~55% | ~55% | ~58% | spectral bias — hard for **all** nets, weak signal |
| degree-K staircase | 100% | 100% | 100% | sign ≈ low-degree term — **near-linear, not depth** |

**Conclusion:** the obvious small-scale tasks either don't need depth (shallow solves them) or
are SGD-hard for *every* architecture (deep included). This is the classic tension — provable
depth separations (parity-like) are optimization-hard; SGD-easy tasks are shallow-solvable. It
also means earlier depth results here (surrogate Finding 5 used linearly-separable blobs) were
never on depth-necessary tasks.

## What's actually tamed vs not
- **Backward explosion: tamed.** With the physical cap bounds, deep gradients stay 0.006–0.19
  (no blow-up). Not the wall.
- **Real depth (inter-layer credit assignment): the open problem.** Deep backprop on the one
  mild compositional task (2×2) gets *stuck* at 60% with healthy gradients — the approximate
  transpose-read direction and/or the lossy forward neuron, not explosion.

## Genuine-depth directions to try (no shortcuts)
1. **Measure gradient fidelity** — cosine between the analog transpose-read gradient and the
   exact gradient of the same analog forward, per layer. Tells us if the blocker is backward
   *direction error* (→ fix the backward) or forward representation (→ fix the neuron).
2. **Difference Target Propagation** — per-layer *learned inverses* propagate real targets that
   encode composition (unlike the naive heads above). Buildable, no weight transport, no explosion.
3. **Equilibrium Propagation** — energy-based net relaxes to equilibrium; free/nudged difference
   IS the true deep gradient, computed locally. The principled analog answer (this is HANDOFF_B).
4. **More accurate analog backward** — calibrate per-layer keystone gain/offset so the deep
   gradient direction survives stacking; use the true ReLU² derivative not a step mask.
5. **Parameter/transistor-efficiency framing** — depth "works" if a deep *narrow* net matches a
   wide shallow net at fewer transistors on a compositional task. Fits the chip's economics.

## DIAGNOSIS RESULT (decisive): the backward is fine — the forward neuron is the wall

"Diagnose first" was run to the end. Three independent measurements all point the same way:

1. **Gradient-direction fidelity** (`grad_fidelity2.py`, validated: pure-linear control gives
   cosine 1.000, finite-diff converged across eps). The analog transpose-read gradient vs the
   exact gradient of the same forward: **output layer cosine ≈ 0.89, hidden layers ≈ 0.4–0.6,
   and roughly depth-INDEPENDENT** (depth-3 layers 0.31–0.65, no worse than depth-1's 0.40). So
   the gradient does **not** explode, vanish, or go orthogonal with depth. (My first pass that
   showed ~0.0 cosines was finite-diff quantization noise from a 12-iteration column solve —
   caught and fixed.)
2. **Graded vs step ReLU′** made no difference — the residual direction error is the keystone
   transconductance (small-signal gain ≠ represented weight), not the activation derivative.
3. **Exact-gradient training test** (`deep_exact.py`): train deep [6,6,6,2] on 2×2 with the
   approximate gradient to convergence, then switch to the **exact** (finite-diff) gradient.
   Across 3 seeds: **80→79, 66→70, 50→50** — the exact gradient buys 0–4 points and never
   approaches the ideal-MLP's 90%.

**Conclusion:** the analog backward is *good enough*; replacing it with a perfect gradient does
not unlock depth. The bottleneck is the **forward neuron** — dead-zone (synapse cutoff) +
quadratic ReLU + rail clamp give limited dynamic range, and cascading lossy neurons destroys the
signal faster than depth adds value. This *proves* (not just hypothesizes) SURROGATE_STUDY
Finding 5 part 3, and shows it bites even at shallow depth.

**Therefore: pursuing a fancier backward (EqProp / DTP / gain-calibrated transpose) would be
wasted effort.** The lever for depth is **neuron fidelity** — a self-normalizing / higher-
dynamic-range analog activation (e.g. companding / log-domain, or a normalized activation that
holds unit scale through layers) so cascaded neurons stop compressing signal. That is the next
thing to design and test.

## Open question for the project
"Get depth to work" needs a concrete **anchor task where depth is provably necessary AND
learnable**. The candidates above don't cleanly separate at small scale. The right next move is
to agree on that task (or accept the parameter-efficiency framing) before building deep in SPICE.
