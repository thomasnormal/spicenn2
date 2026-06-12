# MNIST 4x4, ≥3 labels — fully-in-ngspice multi-class trainer (SOLVED)

Goal: train a ≥3-label MNIST-4x4 classifier in-circuit, above chance, reliably. **Achieved**, and
the fix was a root-cause insight, not a lottery.

## Result (real ngspice, `run_mnist.sh`)

| labels | classes | test acc | chance | per-class recall | reliability |
|---|---|---|---|---|---|
| 0,1,7 | 3 | **77.3%** | 33.3 | [56,98,78] | identical on seeds 1–4 (deterministic) |
| 0,3,6 | 3 | **83.3%** | 33.3 | [68,94,88] | deterministic |
| 1,4,8 | 3 | **65.3%** | 33.3 | [70,100,26] | deterministic |
| 0,1,3,7 | **4** | **72.0%** | 25.0 | [56,96,48,88] | deterministic |

All classes stay alive; no noise, no restart, seed-independent. In-circuit accuracy lands a few
points under the ideal-linear ceiling (e.g. 0,1,7: 77% vs 86.7% sklearn logistic on the same 4x4
z-scored data).

## What was actually wrong (the long-standing "multi-class collapse")

For weeks the trainer collapsed to a 2-vs-1 degenerate basin: one class got ~0% recall, which one
varying by seed/noise. We blamed (and tried) capacity, dense-vs-sparse receptive fields, output-bias
freeze (NOBIAS), subtractive competition (COMP), and a noise lottery — all only patched it to ~50%
on a lucky draw. Two clean experiments found the real cause:

1. **A purely linear 16→3 map already separates these triples at ~94%** (sklearn). So it was never
   capacity or the hidden layer being too small — a linear classifier suffices.
2. **A DIRECT (hidden-free) linear classifier with an EXACT in-circuit gradient still collapsed**
   ([82,0,16], margin 0.03). So it was never the approximate transpose-read backward either.

Inspecting the deck: output `y_c` maxed at ~0.6 while one-hot targets are 1.0 (weights railed,
gate caps saturated at 2.99 V), and **digit "1" — sparse, few bright pixels — had a structurally
lower summed input current than dense "0", so its keystone output `y_1` was permanently lower
regardless of weights.** The collapse was an **input-energy asymmetry**: the keystone sums weighted
inputs, sparse-digit classes sum less, and they lose every tie. Per-image min-max normalization does
NOT fix this (a sparse image still sums lower after filling [0,1]).

## The fix — equalize input energy (z-score per image)

`prep_mnist.py NORM=zscore`: per image subtract mean, divide by std (→ every image has equal
energy, mean 0 / var 1), then map to the [0,1] input band. With energy equalized no class is
structurally disadvantaged, and the **exact linear delta rule trains cleanly and deterministically**
to 77–83%. This single data change flipped 33% (one class dead) → 77% (all alive), no noise.

## Recipe (`run_mnist.sh`)

`DIRECT=1` (hidden-free linear keystones read inputs directly), `NORM=zscore`, `OWTW=20`,
~1400 slots, `AVGW=20`. Exact gradient = error × input, so no backward, no collapse, no restart.

## Note on depth / the hidden layer

Adding the trained ReLU² hidden layer back — even on z-scored inputs — **re-collapses** ([100,0,0]).
Its approximate transpose-read backward (~0.5 cosine, see `NEURON_FINDINGS.md`) reintroduces the
2-vs-1 basin. The hidden layer is not needed to meet this goal; the linear in-circuit classifier is
the robust solution. Making the *hidden* path equally reliable is the open follow-on (it needs the
backward fixed, consistent with the earlier neuron findings), separate from this multi-class goal.
