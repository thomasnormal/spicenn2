# Path to 0.90 on C=10 (deep, full-PC, Spectre)

State: the C=10 *collapse* is solved (symmetric ±β nudge + weight decay → 0.41±0.04, robust, deep, full PC, real Spectre). The remaining gap to 0.90 is NOT collapse — it's accuracy ceiling. Decomposed: (A) feature/representation quality, (B) analog readout/synapse efficiency, (C) optimization within the sim horizon, (D) data. The project's own evidence says 0.90 @8×8 is reachable (pyramid ideal 91%, 2×2-patch features 88–93% vs random 85%, device-faithful model already 87%). 10 big ideas, ranked by expected impact, all respecting deep / full-PC / Spectre / no-wide.

## 1. Convolutional weight-shared local receptive fields (BIGGEST lever)
Features are the bottleneck. Replace the random-sparse FANIN=4 wiring with a **deep convolutional PC net**: local 2×2 receptive fields, **weight-shared across spatial positions** (same weight cap reused at every position → few params, high precision, translation invariance). Evidence: 2×2-patch features → 88–93% ideal vs random-tanh 85%; build_sparse.py pyramid = 91% ideal @8×8. This is the single highest-impact change. Stays deep (conv layers ARE depth), full PC (each conv layer has its local target), and few caps (sharing). Implement: a conv-layer generator in pc_deep that emits shared `wp_/wn_` keys per filter, stride-2 pyramid 8×8→4×4→2×2→C.

## 2. Higher input resolution (16×16, then 28×28)
Ideal ceiling: 8×8 ≈ 85–88%, 16×16 ≈ 93%, 28×28 higher. More pixels = more separable. Costs more input nodes, but with conv weight-sharing (idea 1) the param count stays bounded. Combine 1+2: deep conv on 16×16 → ideal ≥93% → 0.90 reachable at realistic efficiency. Data already present (mnist_16x16.npz).

## 3. Linearize the synapse multiply (degenerated Gilbert cell)
The localized ~6-pt analog efficiency loss is the **square-law single-ended synapse** (gsyn). The project's pc_batch.py `gsyn` is a 4-quadrant **degenerated Gilbert multiplier** (differential, source-degeneration RDEG) with multiply linearity R²≈0.99 vs the single-ended cell's ≈0.92. Swap pc_deep's synapse for it → recover most of the readout-efficiency gap directly. Cost: differential routing + ~3× synapse MOS. Known-good in-project.

## 4. Deep supervision / per-layer auxiliary heads (PC-native)
PC already computes a local target at every layer. Attach a small **auxiliary classifier head to each hidden layer** with its own zero-sum/symmetric-nudge target, so every layer is pushed to be class-discriminative (not just the final readout). This sharpens deep features and shortens the effective credit-assignment path — exactly what PC is built for. Sum the per-layer update pressures. Cheap, very on-architecture, should lift feature quality across depth.

## 5. Residual / skip connections (deep gradient flow)
User-requested earlier; now worth it since collapse is fixed. Add identity skips (a_{l-1} → x_{l+1}) on same-size blocks (RES knob exists in pc_deep). Residuals stabilize deep PC equilibria and let error propagate cleanly through many layers → enables genuinely deep nets (6–10 layers) without signal fade. Pairs with idea 1 (residual conv blocks).

## 6. In-circuit hinge / margin readout (proven multi-class normalizer)
Project finding: the HINGE-GATE was the real multi-class fix (rings C=5: 56→92% vs softmax 92, delta 79). Add a per-output **margin comparator** that only drives the update when (y_correct − max_other < margin) — kills confidently-correct over-drive, focuses capacity on confusable classes (the 2/5/9 weak-class problem we saw). Add on top of symmetric nudge. Targets multi-class accuracy directly.

## 7. Ensemble of independent analog nets (cheap robust points)
Seed variance is high (0.34–0.46). Train K nets with different feature seeds and **vote** (average output rails — trivial in analog: sum the current rails). Ensembling independent learners reliably adds several points AND collapses the variance. Hardware-cheap (parallel, no new cell). K=5 deep nets → expect +5–8 pts and σ↓. Run as K parallel Spectre decks, combine at readout.

## 8. Analog momentum / Adam-like optimizer
Current update is plain delta-on-cap. The project has "analog momentum ≈ Adam" cells (neuron-and-optimizer-cells). A momentum/adaptive optimizer converges faster and to a better minimum **within the sim horizon** — the deep+decay runs were often still climbing or early-stopped. Better optimization closes part of the efficiency gap and reaches higher before time runs out. Implement as a 2nd "velocity" cap per weight integrating past updates.

## 9. More data + light augmentation
Ideal scales strongly with data: 85% @25/cls → 92.7% @500/cls. We train at NTR=20/cls — data-starved. Use the full MNIST train set (mnist_full.npz present). Add only structure-preserving augmentation (small shifts — natural fit with conv idea 1), NOT the input noise that previously hurt. More data raises the ceiling the deep net trains toward.

## 10. Batch-norm / activity homeostasis for deep stability (+ coarse-to-fine curriculum)
Deep PC equilibria can drift in scale across layers (the over-growth we saw). An analog **per-layer activity normalizer** (homeostatic gain that holds each layer's mean |activation| to a setpoint) keeps deep nets in their operating region — enabling more layers and bigger gains without retuning. Combine with a **coarse-to-fine resolution curriculum** (train 8×8 → warm-start 16×16) so the deep conv net learns low-freq structure first. Stabilizes the deeper/bigger nets that ideas 1–2 require.

---
RECOMMENDED FIRST STRIKE (compounding, highest-impact): **idea 1 (conv weight-shared RFs) + idea 2 (16×16) + idea 3 (Gilbert synapse)** — these three together move the *ideal* to ≥93% and the *efficiency* toward ~0.96, which is the arithmetic of 0.90. Then idea 7 (ensemble) + idea 6 (hinge) to bank robust points. Everything stays deep, full-PC, Spectre.
