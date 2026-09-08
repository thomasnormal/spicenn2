# Fully-transistor analog XOR trainer (in-circuit backprop)

A 2-input XOR MLP (6 hidden ReLU units, 1 output) that **trains itself on-chip**: the
forward pass, the gradient (backprop), and the weight update are all physical circuit
behavior. The controller only presents inputs and targets as voltages. The entire deck is
transistor-level: **357 MOSFETs, zero behavioral/dependent (B-)sources.**

## Result

Frozen-weight inference (weights extracted after training, scored on a separate
forward-only deck against the *real* output y):

| pattern | 00 | 01 | 10 | 11 |
|---|---|---|---|---|
| output y | 0.23 | 1.01 | 0.60 | 0.36 |
| target | 0 | 1 | 1 | 0 |

**4/4 corners correct**, MSE 0.086 (averaged readout). See [xor_trans2_convergence.png](figures/xor_trans2_convergence.png).

### Honest caveats
- The **(1,0) margin is soft** (~0.60, vs the other three corners which are crisp). It is
  correctly on the right side of the 0.5 threshold but with a thin margin.
- The learning curve is **non-monotonic**: the network finds a *crisp* solution early
  (MSE 0.064, (1,0)=0.80 at ~19% of training) and then **drifts into the soft fixed
  point** (MSE ~0.10) and stays there. Early stopping or averaging the late weights
  recovers the better solution; the figure marks both.
- These are real characteristics of the deterministic dynamics, not tuning artifacts:
  the soft corner is invariant to backward gain, comparator bias, and training length.

## How it works

- **Weights** are gate voltages on capacitors (volatile; leakage acts as weight decay).
- **Synapse** = a triode transistor pair; signed multiply via the difference of a
  live-weight transistor and a fixed-reference transistor.
- **Keystone** (current-difference -> voltage): a diode-NMOS virtual ground + current
  mirrors + a resistor to a reference. Used for both activations and the backward path.
- **Forward fix** (what makes static XOR solvable): low synapse threshold (Vt=0.2) widens
  the synapse conduction window so hidden units don't die when a weight goes negative;
  high activation gain (RH=8k) sharpens the margins.
- **Backward = exact transpose-read gradient**, fully in transistors:
  - delta1_j = delta2 * w2_j * ReLU'(z1_j).
  - `w2_j * delta2` is a 4-transistor differential cell that *shares the output weight's
    gate* (the transpose) and is driven by a **differential error** (d2e_p = 1.2+delta2,
    d2e_n = 1.2-delta2). The differential encoding is essential: a single-ended error
    drain cuts off exactly when y << target, which deadlocks the (1,0) corner.
  - The error node is **buffered** by matched low-Vt source followers so the transpose
    read doesn't load (and destabilize) it.
  - `ReLU'` is a per-unit **comparator** (z1 vs the 0.5 threshold) whose near-rail output
    gates a widened OTA tail. A softer z1-gated tail starves near-threshold units.
- **Update** = a 5-transistor OTA per weight, push-pull into the weight cap.

The three backward fixes (buffer, differential error, comparator + wide tail) were each
forced by a diagnosed failure (runaway, (1,0) deadlock, soft margin), not guessed.

## Stochastic weights / robustness (Bayesian tangent)

Treating the weights as noisy (which they are on real silicon) was tested:
- A leaky cap under noisy updates is an Ornstein-Uhlenbeck / SGLD sampler (leak = Gaussian
  prior, gradient = likelihood, device noise = diffusion); the late-weight average is its
  posterior mean.
- **Averaging the late weights** is a free, strict win (lifts deterministic (1,0)
  0.55 -> 0.62); on a real chip it's just "stop updating and time-average the output."
- **Injected weight noise + averaging** appeared to improve robustness in the runs tested:
  a good noise-trained draw stayed 100% correct out to +/-0.2 V gate jitter where the clean
  net dropped to 80% at +/-0.15 V. Two honest caveats keep this from being a clean claim:
  (1) noise training is **high-variance** -- it solves on most seeds but fails outright on
  some, so the benefit only applies to draws that converge; and (2) the more-robust noisy
  draws also simply had **more margin** at baseline, and this experiment does not separate
  "flatter minimum" from "more margin" (both push the same way, and both come from the noise
  training). A clean claim would need matched-margin, multi-solution comparisons. So: a
  promising robustness signal, consistent with theory, not a settled result.
- **Noise annealing does NOT help** (it's worse): the crisp region is a posterior centroid,
  not a deterministic basin floor, so settling collapses back to the soft attractor.

## Files

Reproducible toolchain (ngspice 42, numpy/matplotlib):
- `gen_xor_full.py` - parameterized fully-transistor trainer (env-configurable;
  `BWD=trans2` is the fully-transistor exact backward). Emits [xor_full.cir](../circuits/xor_full.cir).
- `gen_infer.py` - env-aware forward-only frozen-weight deck (supports late-weight
  averaging `AVGW` and an inference-time perturbation `PERTURB` for robustness tests).
- `score.py` - matches the 4 corners, samples settled y, reports MSE / corners.
- `checkpoint_curve.py` - builds the honest checkpointed validation learning curve.
- `plot_convergence.py` - renders [xor_trans2_convergence.png](figures/xor_trans2_convergence.png).
- [xor_full.cir](../circuits/xor_full.cir) - the working deck (deterministic trans2, the config above).
- [xor_trans2_convergence.png](figures/xor_trans2_convergence.png) - honest learning curve + frozen truth table.

To reproduce:
```
export TS=0.3 DTFRAC=0.34 VTOSYN=0.2 RH=8e3 H=6 VC=1.8 GS=1.2 RTH=8e3 RTO=7e3 \
       VG0=0.7 LO=0.7 HI=1.3 VREFH=0.5 VREFO=0.5 BWD=trans2 RTBK=8e3 VCBIAS=1.0
python3 gen_xor_full.py 600 0.3 5e-4 && ngspice -b xor_full.cir
AVGW=30 python3 gen_infer.py && ngspice -b xor_infer.cir && python3 score.py result
```

Component-characterization figures from earlier validation are retained: [real_synapse.png](figures/real_synapse.png)
(synapse forward multiply), [real_cells.png](figures/real_cells.png) (cell behaviors). The chasing-era convergence
figures have been removed; they did not measure the real output and were superseded by the
checkpointed curve here.
