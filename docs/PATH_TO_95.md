# Path to >0.95 on MNIST (10-class) — 10 big ideas

## The gap, located (token-free ceiling analysis, 8×8 sklearn digits)

Current in-circuit best (keystone, ngspice): **0.747**. Decomposition:

| Ceiling (ideal classifier) | acc |
|---|---|
| 96 random tanh feats, 40/cls (current keystone setup) | 0.865 |
| 96 random feats, 400/cls | 0.898 |
| 300 random feats | 0.915 |
| logistic on raw 64 px (full data) | 0.920 |
| trained 1-hidden MLP | 0.926 |
| **RBF-SVM (kernel)** | **0.948** |

`0.747 = 0.865 ceiling × 0.86 efficiency`. Two independent gaps: **representation** (raise the ceiling) and **efficiency** (close the analog/ideal gap).

**Decisive fact: 8×8 digits cap at ~0.93–0.95 even with the best classifier.** >0.95 on 10 classes is an *input information limit* at 8×8 — it is NOT reachable by any circuit improvement on 8×8. **Real MNIST (28×28, ideal ~0.99) is a prerequisite.** Every idea below assumes we move to higher-resolution MNIST.

---

## A. Raise the representation ceiling (the dominant gap)

**1. Move to real MNIST 28×28 (or 16×16).** The single most important change. 8×8 is data-capped ~0.94; 28×28 logistic ≈ 0.92, conv ≈ 0.99. Without this, >0.95 is impossible. Cost: 784 inputs → must pair with local receptive fields / conv (idea 2) to keep the circuit tractable.

**2. Convolutional weight-sharing front-end.** The standard MNIST→0.99 architecture. Weight sharing collapses the 28×28 param explosion and adds the translation-invariance prior. Analog realizations: (a) *scanning conv* — one small synapse bank time-multiplexed across patches (cheap silicon, slow); (b) *replicated cells sharing one weight-cap* (a single charge node fans out to all positions, updated by the summed local gradients). Builds on the RFGRID local-window wiring already in pc_deep.

**3. Trained features, not random.** Random features cap the ideal (0.885 @ N=96); trained features reach 0.926 (MLP) and ~0.99 (conv). We already have the working depth-2 *amplitude-restoring* backward (BKSIGN, 0.55→0.84) to train hidden layers. Combine trained conv features + the keystone readout (zero-sum + PERAZ).

**4. More features + more data (cheap, stackable).** N=96→300 lifts ideal 0.885→0.915; 40→400 samples/cls lifts 0.865→0.898. Both are free ceiling points; the cost is simulation slots. Worth banking once on the real-MNIST substrate.

## B. Close the efficiency gap (86% → ~98%)

**5. Offset-canceling cells everywhere (generalize PERAZ).** The in-circuit/ideal gap is analog imperfection (mismatch, finite gain, output-conductance compression). PERAZ (per-class auto-zero) already closed much of the *readout* gap (C=10 0.30→0.747). Extend the same slow-tracker offset cancellation to the synapse array and neurons — the paper's stated #1 agenda. This is the lever that turns a 0.97 ceiling into 0.95 in-circuit instead of 0.83.

**6. Differential rails / better-matched cells.** Fully-differential signal paths reject common-mode offsets and even-order distortion; larger / common-centroid devices cut σ_VT. Directly attacks the residual efficiency loss after PERAZ.

## C. Better learning dynamics

**7. A working analog optimizer (per-layer AGC or momentum).** Plain SGD underperforms; momentum projects +2.4%, and per-layer adaptive gain rebalances deep layers' weak gradients. The cascaded momentum cell collapsed before, but per-layer AGC (O(layers), one v-cap/layer) is simpler and was the proposed cheap variant — worth a real build now that depth is trainable.

**8. Two-phase equilibrium propagation with a proper nudge.** The soft β-nudge was decisive for 2D (1/4→4/4 seeds). A clean free-phase/nudged-phase EP gives less-biased gradients than simultaneous PC; the contrastive (CHL) update path already exists. Could close part of the efficiency gap *and* improve depth.

## D. Leverage / cheap wins

**9. Ensemble of readout heads.** Several independent readouts (different feature subsets / seeds) voting. Analog noise + init variance are partly independent, so averaging reliably adds +1–2% for near-linear silicon cost. Committee infrastructure (SAVEM margins) already exists.

**10. Train-offline, deploy-in-circuit (separate hard from easy).** Key finding: in-circuit *inference* capacity is high ("deploy ridge → 1.0") while continuous *learning* lands at a worse equilibrium. So train the weights with a high-fidelity method (the device-faithful numpy model, or 2-phase EP) and *deploy* the learned capacitor charges for in-circuit inference. Decouples the 0.99-capable forward pass from the lossy learning loop; an honest "inference-in-circuit, training-assisted" result could clear 0.95 well before fully-in-circuit learning does.

---

## Recommended sequence (impact × tractability)

1. **#1 + #2 + #3** are the prerequisite trio: real MNIST + conv weight-sharing + trained features → raises the ceiling from 0.86 to ~0.98. Nothing else matters until the ceiling moves.
2. **#10** is the fastest route to a *number* >0.95 (deploy-trained-weights), honestly labeled as inference-in-circuit.
3. **#5** (generalized offset cancellation) is what makes fully-in-circuit learning competitive at the high ceiling.
4. **#4, #9** are cheap stackable points; **#7, #8, #6** are the harder learning/efficiency frontier.

**Biggest risk / honest caveat:** simulation cost. 28×28 conv nets are far larger than anything simulated here (the 17 MB keystone deck already stalls Spectre; ngspice is single-threaded). Realistically this needs either the device-faithful fast surrogate for exploration (with the known caveat that it under-predicts), batched/scanning architectures to shrink the deck, or a faster simulator — itself a prerequisite for the whole program.

---

## Update (grounded ceiling re-analysis on REAL MNIST)

Two corrections to the analysis above, from measuring on *real* MNIST (not sklearn digits):

- **Resolution is NOT the bottleneck.** Real MNIST at **8×8** already reaches **0.956 with a kernel** (RBF-SVM); 12×12 ≈ 0.955. So 8×8 is sufficient resolution to clear 0.95 — the earlier "need 28×28" was an artifact of the smaller/noisier *sklearn digits* set (capped 0.948). 8×8 stays circuit-tractable (64 inputs). [Idea #1 ⇒ use *real* MNIST, not necessarily higher res.]
- **The lever is nonlinearity, and trained ≫ random for it.** Linear (logistic) caps ~0.90; the kernel/nonlinear ceiling is ~0.956. Random features approach it only slowly (real MNIST 8×8: N=96→0.869, 600→0.910, 1200→0.929; ~0.95 needs N≈2400 = impractical cell count). **Trained features reach the ceiling with far fewer units** ⇒ Idea #3 (trained features) dominates Idea #4 (more random features).
- **The binding constraint for >0.95 *in-circuit* is efficiency, not the ceiling.** At our ~88% in-circuit/ideal efficiency, even a 0.95 ideal yields ~0.84 in-circuit. So Idea #5 (offset-canceling cells → push efficiency toward ~0.98) is *required*, not optional, for a >0.95 in-circuit number.

**Refined recommended path:** real MNIST 8×8 (done: `data/mnist_real_8x8.npz`) + trained nonlinear features (#3, via the amplitude-restoring backward) to reach a ~0.95 ideal at modest cell count, + offset cancellation (#5) to convert that into >0.95 in-circuit. Idea #10 (deploy trained weights) remains the fastest route to a labeled >0.95 inference number meanwhile.

## BREAKTHROUGH: >0.95 is deployable on 8×8 (idea #10 is concrete)

Real MNIST 8×8, trained MLP (exactly pc_deep's input→tanh-hidden→readout architecture):
`64→64→10 = 0.953`, `64→128→10 = 0.960`, `64→128→64→10 = 0.964` — **>0.95 on a 64-input circuit**.
So idea #10 is now a concrete target: train 64→64→10 offline (0.953), deploy weights via WLOAD, run
in-circuit INFERENCE (training-assisted). If inference is high-fidelity (cf. "deploy ridge→1.0"), this is
a >0.95 in-circuit-inference number — no 28×28, no exotic cells. Building the deploy pipeline now.

## Deploy-pipeline feasibility (#10), measured

pc_deep's neuron is a *sharp* tanh ($0.537\tanh(15.7u)$, near-binary), not standard tanh. Deploying
standard-MLP weights would mismatch, so the MLP must be trained with the device transfer. Measured cost:
a 64→64→10 MLP with sharp activation (gain 6–12) reaches ~0.91–0.92 (quick training) vs ~0.93 at gain 1 —
**the sharp neuron costs ~1–2 points.** So device-matched deploy on 8×8 lands ~0.93–0.95: **borderline** for
>0.95. Clean margin needs either (a) 128 hidden + thorough device-matched training, or (b) 28×28 input.

**Concrete #10 build (the executable recipe):**
1. Replicate pc_deep's exact forward in numpy (sharp neuron + degenerated-synapse gain + wgv rail map).
2. Train 64→128→10 on real MNIST 8×8 with that forward (target ~0.95 device-matched).
3. Export weights → WLOAD JSON (keys `L_j_p`, biases `wbp_b_L_j`), dense (FANIN=64, KOUT=128/64).
4. Run pc_deep TASK=mnist with WLOAD + learning-gate OFF (deploy + inference-only) → read eval accuracy.
5. Gap to verify: does in-circuit inference match the device-matched offline number (cf. "deploy ridge→1.0")?

**Status:** path fully analyzed and grounded; the deploy pipeline is a defined multi-step build (steps 1–5),
borderline-0.95 on 8×8 / clean on 28×28. This is the recommended next focused effort for a >0.95 number.
