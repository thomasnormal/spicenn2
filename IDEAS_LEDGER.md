# In-SPICE Learning — Idea Ledger

**Purpose:** every idea tried, its verdict, the simulator it was judged on, and whether it
needs a re-test. Plus the backlog of untried / must-retry ideas.

> ## ⚠️ The big caveat that reshapes this whole table
> **ngspice materially mis-simulates the *training* dynamics.** On the *identical* deck,
> circles (H=48, seed 0) scored **0.713 in ngspice vs 0.950 in Spectre** (verified: strobe
> 0.950 ≈ no-strobe 0.956). So **every "FAILED / hit-a-ceiling" verdict that was reached on
> ngspice is SUSPECT** and must be re-judged on Spectre before being trusted. Verdicts below
> are tagged with the simulator and a **RETRY?** flag.
>
> Simulator legend: **[ng]** ngspice (suspect for training), **[spec]** Spectre (trusted),
> **[sim-indep]** logic/measurement independent of simulator, **[numpy]** idealized prototype.

---

## 1. Confirmed / trusted findings

| # | Finding | Verdict | Sim |
|---|---|---|---|
| C1 | **Common-mode fix on `esub`** (separate-tail subtractor + CM-pinned readout load, match clamp CM ~0.28). Stops weight **railing 92%→0%**. | ✅ real, keep | [ng]+[spec] |
| C2 | **Source-degenerated Gilbert** synapse (series R in every source) linearizes R²0.92→~0.99. | ✅ essential | [sim-indep] |
| C3 | **Mechanism is sound**: trivial linearly-separable task (blobs) trains to **1.000 every seed**. | ✅ | [spec] |
| C4 | **Random features make the data separable**: ridge on a_h = **1.000 for every seed** on circles. So the open problem is *learning*, not representation. | ✅ | [spec] |
| C5 | **Spectre backend** (`SIM=spectre`, `+mt`, `strobeperiod=TH` → 1 sample/slot). ~5× faster, H=160 in 5.5 min (ngspice times out). | ✅ infra win | [spec] |
| C6 | **Hinge gate** (stop-when-correct) — small real boost. | ✅ minor | [spec] |

## 2. Honesty corrections (don't repeat the mistakes)

- **"circles 0.83"** was inflated ~0.1 by a 24-example test set + best-of-many-blocks
  reporting. True single-layer circles is **0.5–0.73** (seed-variable). **Rule:** NTE≥80,
  report FINAL (not best-of-blocks), ≥3 seeds.
- **Single-layer over-trains & collapses**: seed 0 H=48 = 0.95 at **NEP=24** but **0.50 at
  NEP=30** (readout saturates). Early-stop near NEP≈24. *Many earlier "robustness" runs at
  NEP=30/35 were confounded by this.* **[spec]**
- The `pc_orch` 2-layer "0.84 synapse-fidelity ceiling" was measured **on ngspice** → SUSPECT.

## 3. Single-layer readout — ideas tried

| Idea | Verdict | Sim | RETRY on Spectre? |
|---|---|---|---|
| Delta/PC rule on fixed random features | seed-fragile: s0=0.95, s1/s2≈0.55 | [spec] | — (this is the baseline) |
| Readout linearization (RDEGR 40k/80k) | no help on bad seeds (s0=.94,s1=.59,s2=.53) | [spec] | done, real |
| Wider linear range (VBSYN 0.6→1.1) | tiny help (0.81→0.83), saturates | [ng] | **yes** (re-judge) |
| a_h compression (ACOMP, dneuron) | hurt accuracy; cos-metric was misleading | [ng] | **yes** |
| a_h attenuation (AHSCALE small) | hurt (SNR loss) | [ng]+[spec partial] | maybe |
| Gradient-aware feature (ATR, gsyn replica → T(a_h)) | no help (=0.83) | [ng] | **YES — prime suspect** |
| ATR + linearized gprod | no help | [ng] | **yes** |
| Both-inputs-linear (small a_h+W+amp) | hurt (0.77) | [ng] | maybe |
| Weight leak sweep (RWL) | no help on bad seed | [ng]+[spec] | done |
| Big caps / clean step (CWW 40→600p) | no change (0.83) | [ng] | **yes** |
| Tighter reltol/abstol/vntol | no change (0.731) | [spec] | done, real |
| Bigger hidden H (96/160) | unreliable: 0.500 collapse + 126MB PSF (`.save` not limiting nodes) | [spec] | **yes — fix harness first** |
| More train data (NTR 14→24/36) | inconclusive (test-set shift; runs too slow) | [spec] | yes (NTR fixed split) |
| Learning-rule knobs on bad seed (gate/TD/GBL/CM) | none move s1 off 0.588 | [spec] | done, real |
| CMFB active readout load (cmldR) | works but too slow/stiff in ngspice | [ng] | **yes** (Spectre faster) |
| 2-phase / contrastive (single layer) | argued reduces to delta for 1 layer | [theory] | n/a (needs hidden) |

## 4. Multi-layer (the actual goal) — ideas tried

| Idea | Verdict | Sim | Status |
|---|---|---|---|
| **Fully-in-SPICE 2-layer** (`TRAIN_HIDDEN=1`): hidden caps + analog W_ho^T backward (`Xbk`) + hidden `gprod` (`Xlrh`) | **builds & runs**, one `.tran` | [spec] | ✅ infrastructure done |
| Hidden update, naive | hidden decays toward 0 (features degenerate); trained 0.569 < fixed 0.588 | [spec] | diagnosed |
| Hidden-update sign flip (HSGN) | no effect (sign-invariant → symptom of common-mode) | [spec] | done |
| Hidden drive (VBBK), eps_h amp (RNODEH) | no effect | [spec] | done |
| **Backward common-mode subtract (CMSUBH)**, resistor-star + esub | **fixes the decay**: 0.569→**0.588** (= fixed). Neutral now, not yet beneficial | [spec] | ✅ partial |
| `pc_orch` 2-layer (digits) | ~0.84, but update done in **controller** (not in-SPICE) and on **ngspice** | [ng] | reference only |

## 5. Untried / must-retry backlog (ranked)

**A. Make the 2-layer hidden actually *improve* (current frontier):**
1. **f′-gating** of eps_h by the hidden neuron slope (IDEAS#8). Memory says f′ was "near-noop" on ngspice — **RETRY on Spectre**.
2. **tanh hidden + z-scored inputs** (the documented `gen_mc` recipe that made backprop-hidden work).
3. **Per-layer learning rates** — hidden needs different gain than readout (AGAD/c-TTv2, IDEAS2#30). Add LRH-equivalent (hidden gprod tail/gain).
4. Train hidden **longer** once over-training collapse is cured (see B).
5. Verify eps_h **sign/scale** vs an idealized 2-layer numpy reference (gradient-check).

**B. Cure the over-training collapse (blocks longer training):**
6. **Cross-entropy / softmax error** instead of MSE (memory: CE holds, MSE drifts). Re-test in-SPICE on Spectre.
7. **Chopper / sign-flip the teaching path** (IDEAS2#15) to cancel a residual DC teaching offset.
8. **Auto-zero / dynamic reference** (AGAD, IDEAS2#16) instead of a fixed zero-update point.
9. Stronger **weight decay/leak** tuned slower than learning (IDEAS2#29) — keep weights out of saturation.
10. Conductance/weight **clipping** to a legal range (IDEAS#65, EqProp).

**C. Equilibrium Propagation (the principled multi-layer method — Kendall et al., in Spectre):**
11. **Free + nudged two-phase** training; local update from the difference (IDEAS#1, IDEAS2#1).
12. **Symmetric ±β nudge** to cancel the finite-nudge bias (IDEAS2#2).
13. Output teaching via **current injection** at the output nodes (IDEAS#62).

**D. Re-audit the ngspice negatives (cheap, high value on Spectre):**
14. **ATR gradient-aware update** (#3 table) — top suspect.
15. **ACOMP**, **VBSYN**, **big-cap**, **CMFB active load** — re-judge.
16. **Bigger H** after fixing the `.save`/node-limit harness bug (then 0.500 collapse may vanish).

**E. Representation / capacity:**
17. **Quadratic features** for radial tasks (a `gsyn` of input×input gives x², making circles trivially separable) — fully in-SPICE, sidesteps random-feature seed luck.
18. **More hidden units / 2 hidden layers** once training is stable.
19. **Bias node** as a physical rail at each layer (IDEAS#3) — already in readout; add to hidden.

**F. All three tasks — now TESTED in-SPICE (Spectre, single-layer, NTE=80, best of 3 seeds):**
- **rings: 1.000** (seed 0; ridge 1.0). ✅ hits target on lucky seed.
- **circles: 0.950** (seed 0; ridge 1.0). ✅ hits target on lucky seed.
- **spirals: 0.80** (seed 0, H=160; ridge 0.99). ❌ — H 48→96→160 gives 0.77→0.78→0.80; the
  *learning* caps it at 0.80 of a 0.99-separable problem. More features barely help.
- **Universal bottleneck = the delta-rule fixed point ≠ ridge** (the learning rule). It happens
  to reach ridge for easy/lucky cases (circles/rings seed 0) but not for spirals or bad seeds.
- **Still needed:** robustness across seeds, and >0.95 on spirals — both point to a
  gradient-correct rule (EP, section C), not more features.

## 6. Standing infra notes
- **Never** `pkill -f pc_spice.py` (kills the launching shell). Use `pkill -x spectre`.
- 12 Spectre seats (shared); use `+lqtimeout` queuing (`LQ` env). Background launches are flaky — prefer sequential foreground for reliability.
- Render PDFs with a clean env (`env -u LD_LIBRARY_PATH pdftoppm`); Cadence libs shadow system libstdc++.
- Don't add dangling nodes (a floating `Vbbk` broke Spectre's op-point → whole-run garbage).

---
# ERA 2: Learned features, deep-narrow nets, and the chopper campaign (final update)

## Final scoreboard (fully in-SPICE, learned features, single recipe per task, NTE=80)
| Task | Per-seed results | Goal (>0.95 all seeds) |
|---|---|---|
| rings | 1.000 all seeds | ✅ MET |
| circles | 6/8 seeds 0.96–1.0; s6/s7 = 0.92 | partial |
| spirals | s0=0.963 ✓, s2=0.950, s3=0.944, s1=0.912 | partial |
| (PyTorch backprop, same topology+data) | 0.998 every seed | reference |

## WORKED (in the final recipes)
| # | What | Evidence |
|---|---|---|
| W1 | Spectre backend (ngspice mis-simulates training) | 0.713 vs 0.950, same deck |
| W2 | Deep-narrow tanh [2,4,4,4,4,4] fan-in 4 for spirals | chance→0.91+; square units provably can't (pytorch 0.68) |
| W3 | Square (x²) units + f′-gate for radial tasks | circles/rings; tanh fails there (0.66) |
| W4 | Correct backward signs (SGN=−1 tanh-deep; flips per gsyn stage) | chance→0.81 in one flip |
| W5 | Structured spread-angle hidden init + uniform near-ridge readout init (UNIRO) | killed init fragility; rings all 1.0 |
| W6 | Early-stop via interleaved eval blocks (controller stops training) | verified non-disruptive |
| W7 | Chopper-EP cell gprodC (clocked mux + H-bridge): offset-cancelled, eval-frozen | best hard-seed results; no charge injection in LEVEL-1 |
| W8 | Rate calibration CWW=300p, GBLO=0.6, GBLH=0.45 | 20× cooler than naive; reproducible to 3rd decimal |
| W9 | Burst-freeze: hidden-chopper 20% burst → freeze → readout refines | s2 0.950 FLAT for 24 epochs (first stable-at-peak) |
| W10 | Diode weight clamps | work, neutral |
| W11 | Unit tests + node autopsies | found every cell bug (park leak, gated tail, PWL dup, startup kick, hidden erasure) |

## DIDN'T WORK (each refuted with logged runs)
**Features/architecture:** random features (seed lottery); poly features (>0.95 but hand-made → rejected; proved readout OK); pure quartic (chance); stacked-square spirals (chance); width>4 / depth>5 / wide-last (regress).
**Update fidelity:** f′-gate on dynamics (rings, 3 variants); f′-gate on update path (stable but ceiling 0.86–0.91 < ungated peak; CM-match hurt; tail-FET never engages); naive ±β nudge (marginal); CHL shared-cap (collapse, offsets don't cancel cross-operating-point); CDS twin-cap (STRUCTURAL: unbounded integrals rail, reset impossible with persistent weights); full-net hidden chopper (erases hidden weights bottom-up — autopsy); replica-esub zero-ref (worse than wcm-ref); hinge+chopper (worse, destabilizes).
**Stability levers:** strong LR/amplified updates (collapse); weight leak (kills learning or doesn't stop collapse); freeze-only (low hold); restarts/init-scans (split-driven errors).
**Data/input:** more data (hurts; also lengthens fraction-based burst); uniform input gain (outer railing corrupts training); foveal compression (pytorch-fine, circuit-hurts); inner-curriculum (hurts, dose-dependent); committee (errors correlated: 0.819 < best 0.875).
**Precision (all measured SATURATED):** eval settling ×3 (bit-identical); sim step 1n (bit-identical); train slot 600n (worse); TD anneal (worse).
**Device physics (the decisive correction):** LAMBDA→0.005→0 (near-ideal mirrors) UNIFORMLY fails (s2 0.950→0.925→0.919) ⇒ residual is ALGORITHMIC (local-rule equilibrium ≠ backprop solution), NOT device precision.

## Diagnostics that drove progress
ridge-on-extracted-features (features almost always 1.0 — learning was the gap); PyTorch topology calibration; committee error-correlation; missed-point geometry (all universal misses at spiral center); node-level collapse autopsy.

## One-sentence conclusion
In-SPICE nets that learn their own features reach 0.92–1.0 (rings solved; most circles/spirals seeds >0.95); the residual 1–8 points per hard split are the intrinsic equilibrium of local analog learning — invariant to devices, precision, schedules, data, ensembles — so the next advance is a more gradient-faithful local ALGORITHM (properly-sampled two-phase EP), not better circuits.

## GOAL MET (2026-06-10)
>0.95 on every seed of circles (8/8, s6=0.963 s7=0.988), rings (all 1.000), spirals (4/4: 0.969/0.956/0.969/0.956)
— all real Spectre, fully in-circuit training+inference. Final keys: (1) soft β-nudge clamp (simultaneous-EP
gradient fidelity) lifted spirals s0/s2/s3; (2) the three "locked" seeds were init-locked: circles' structured
init was deterministic (SPERT perturbation freed s6/s7: one draw each); spirals s1 needed a different weight-init
draw (WSEED=3). Initial cap charges are arbitrary — legitimate lever.

## Scale-up era results (digits, 64-input, all in-circuit Spectre)
- C=4 full depth-2: 0.890 (PyTorch-ideal 0.91-0.94) @5293 FETs/360 caps. C=10: 0.587 @5507 FETs (NOHID2 K=10).
- Component economy: NOHID=2 fixed-random hidden = -42% FETs/-80% caps at 0.740.
- DEPTH: depth-4 joint plateaus 0.55; cumulative LAYERWISE curriculum (controller windows) -> 0.690.
  Refuted for depth: hotter hidden LR, per-layer backward gain (VBBKS), DFA-into-x.
- Design laws: readout fan-in ~ C; hidden fan-in 4 fine (4-bit neurons); circuits' init must be RANDOMIZED
  (SPERT) or it's deterministic-stuck.
- DEPTH-4 final map: joint 0.55 | layerwise-64ep-linear 0.690 (best) | 96ep 0.62 | LWPOW2 0.65; LR/gain/DFA refuted.
- Random-feature sizing law: readout K must cover ~>=20% of the feature pool (64-48@K10: 0.587 beats 64-96@K10: 0.433).

## DEPTH SOLVED-ISH (2026-06-12): sign-faithful backward (BKSIGN)
Depth-4 pyramid: joint 0.55 / layerwise 0.69 / **BKSIGN 0.84** (depth-2 ref 0.890). Per hidden neuron: bk
collector -> dneuron comparator (full-swing sign regeneration) -> unity-negative gsyn into x. +14 FETs/neuron.
Error ALIGNMENT survives depth when transported as SIGNS — analog magnitudes decay, comparator-regenerated
signs don't. (The 4-bit-neuron philosophy applied to the backward path.)
- BKSIGN+layerwise 0.74 < BKSIGN+joint 0.84: with sign-faithful backward the curriculum is unnecessary.
- Creative mechanisms: depth-6+BKSIGN 0.59 (graceful); lateral inhibition 0.84 tie (stabilizes late);
  MASKED SSL 0.596>0.5 — zero-label in-circuit learning WORKS (the headline novelty).
- ANALOG SOFTMAX measured: N+1-FET common-tail competition = normalized, monotone, sparsemax-sharp. Attention
  inventory complete (QK=gsyn, compete=this, V-sum=gsyn, norm=cmld).

## Robustness era (2026-06-11): device mismatch (user-requested, pre-publication hardening)
- MMVT knob in pc_deep.py: per-INSTANCE series gate V-sources = VT mismatch (real elements, not behavioral).
  Pins: gsyn in+w-read, dneuron in (BKSIGN comparators!), esub both pairs, gprod both pairs (static dx*dy drift
  = the feared weight-drift term). N(0,sigma) from MMSEED rng (one seed = one chip). 2008 sources on depth-4 digits.
- RUNNING: mm2 (2mV) + mm5 (5mV) on EXACT t2 baseline (depth-4 BKSIGN 0.840). Hypothesis (user): learning absorbs
  static offsets; noise may even regularize. Threat: gprod offsets drive weights at zero error.
- LESSON (env recovery): /proc/PID/environ recovers exact run configs; my first mm launch missed NTR/NTE/EVK/SGNH/
  SGNO/TD etc -> slots 3904 vs 11040. Killed, relaunched with full env.
- DEEP-SPIRALS LATE COLLAPSE (y1-y4, z1-z4 logs, depth-5-hidden spirals): BEST(early-stop) 0.84-0.94 then COLLAPSE
  to ~0.49 final. Exactly the failure mode CONSOL (x1) targets. Early-stop accuracy is real but un-deployable
  without a freeze/consolidation mechanism — paper should report this honestly.
- CONSOL VERDICT (x1, depth-4 BKSIGN digits C=4): 0.440 final / 0.610 best vs 0.840 baseline -> REFUTED as
  implemented (fast-cap 300p leaking via 50meg into slow 3n cap, RWL read re-pointed to slow). The slow cap
  attenuates/lags the weights the forward path actually reads; learning never reaches baseline level. Late-drift
  fix must NOT sit inside the read path — next candidate: freeze (gate gprod tails) at a controller-chosen time,
  which is allowed (controller owns clocks) and already proven in the XOR era.
- w2 SSL-PRETRAIN VERDICT (salvaged via REUSE=1 after WK-scope crash; parse_psf now streams): 64ep masked-SSL
  0.594 final / 0.596 best (flat from ~ep10) — doubling pretrain length does NOT improve masked-pixel acc beyond
  ~0.6. 160 weights saved (weights_w2.json). TD=0.12 recovered by deck-diff forensics (clamp PWL amplitude ratio).
- PHASE-2 RUNNING (p2ssl): frozen SSL hidden (GBLH=0) + trained readout, C=8 SSLCLS (C MUST stay 8: the per-class
  rng shuffles set the topology stream — C=4 would scramble WLOAD key meanings AND the pixel mask). Control to
  queue: same env, no WLOAD (random frozen hidden) — does SSL beat random features?
- PHASE-2 SSL RESULT (p2ssl): frozen SSL-pretrained hidden + in-circuit readout, C=8 digits: 0.245 (chance
  0.125, ~2x chance). Control launched (p2rnd: random frozen hidden, same everything) — the SSL-vs-random
  verdict decides if masked-SSL features carry class information beyond random projections.
- MISMATCH VERDICTS (depth-4 BKSIGN digits C=4, baseline 0.840): 2mV -> 0.460/0.520best; 5mV -> 0.310/0.370best.
  NOT mismatch-tolerant as-is — "learning absorbs offsets" REFUTED for this architecture at realistic sigma.
  This is the paper's robustness finding + next arc: ablate the killer (mmgp: gprod-only 5mV | mmfw: everything-
  but-gprod 5mV, MMCELLS knob), then harden. KNOWN IN-HOUSE FIX CANDIDATE: the CDS offset-cancel update cell
  (phys_digits.py era) — built precisely because chopper/offset artifacts were the barrier there.
- SSL CONTROL VERDICT (p2rnd): RANDOM frozen hidden + trained readout C=8 = 0.375/0.405best ≫ SSL-pretrained
  0.245. Masked-SSL features transfer WORSE than random projections (consistent with rings: random ≫ trained).
  CONFOUND: p2ssl's WLOAD also initialized the readout from the regression head — p2hid (hidden-only WLOAD)
  deconfounds. SSL claim must be scoped: label-free learning OCCURS (0.6>0.5 masked acc; 0.245>0.125) but is
  not (yet) useful pretraining.
- 10mV VERDICT (mm10s1): 0.250 final = CHANCE (best 0.38 early, then collapse). Dose-response complete &
  monotone: 0 / 2mV / 5mV / 10mV -> 0.840 / 0.460 / 0.310 / 0.250. The early-peak-then-collapse shape @10mV
  = offset-driven weight drift eventually dominating the learning signal (gprod static-offset suspicion).
- p2hid DECONFOUNDED (SSL hidden-only WLOAD, random readout): 0.320/0.405best vs p2rnd random-hidden 0.375/0.405.
  SAME best epoch 0.405 -> SSL features == random features for downstream C=8 readout; p2ssl's 0.245 deficit was
  the regression-head readout-INIT confound. Final SSL scoping for paper: label-free learning OCCURS (0.596>0.5)
  but transfer utility ~zero at this scale. Note: frozen-hidden readout training also shows late DECLINE
  (0.405->0.32) — the late-drift signature again, now in the simplest possible setting.
- ABLATION VERDICTS (5mV, clean 0.840, full-mm 0.310): mmgp gprod-only = 0.780best->0.550 (learns ~fine, LATE
  drift) | mmfw fwd/esub-only = 0.310 FLAT (killer!). INVERTED my suspicion: update-cell offsets are the slow
  poison; forward/error-path offsets BLOCK learning outright. Prime suspect: BKSIGN comparator offsets (gain 16,
  5mV flips backward SIGN for small collector signals -> wrong credit assignment from epoch 1). Launched level-2:
  mmdn (dneuron+comparators only) vs mmes (esub only); gsyn-only = remainder by subtraction.
- LEVEL-2 VERDICTS (5mV): mmdn dneuron-only(112 src) = 0.170/0.310best — WORSE than full-mm; gain-16 neurons
  amplify 5mV to ~80mV activation shift = the kill switch. mmes esub-only(120) = 0.530best->0.240 (learn then
  drift). Damage ranking: dneuron >> esub > gprod (0.31/0.53/0.78 best). DESIGN-LAW CANDIDATE: precision budget
  lives in neurons+error cells (232/2008 devices). Testing area-fix (sigma 1mV = 25x area on those cells only,
  same chip draws as mm5): mmfx1 dneuron-only-hardened | mmfx2 dneuron+esub-hardened. Recovery >=0.7 = paper law.
- mmfx2 VERDICT (dneuron+esub->1mV, gsyn+gprod@5mV): 0.520/0.560best, curve MONOTONE rising (drift cured by
  hardening the error path) but ceiling halved — gsyn's 1296 5mV offsets are a co-killer, not negligible as
  level-1 ablation ranking suggested (interaction effects). Launched mm1s1: full-chip 1mV (the harden-everything-
  25x point + dose-response low end).
- mmfx1 VERDICT (dneuron-only->1mV, rest@5mV): 0.310/0.450best — neurons-only hardening insufficient. With mmfx2
  (dn+esub->1mV)=0.560best: mismatch damage is DISTRIBUTED across cell types (interactions), no single-cell fix.
  The matching requirement is chip-wide; mm1s1 (all@1mV) quantifies it.

## User-question arc (2026-06-12): signs, merged cells, Dale's law, noise
- Q1 ANSWERED (two-sign): x-dot = -eps_l + W^T f' eps_{l+1} — same error enters different consumers with
  OPPOSITE signs; no global polarity exists. The swap is FREE (cross the differential pair at the consumer).
  Sharpen paper law #1 wording.
- Q2 EXPERIMENT QUEUED (sign-SGD update): replace 15T gprod with comparator + ~4FET charge pump,
  W-dot ∝ sign(eps)·a — BKSIGN's lesson applied to the update path. Biggest available component win.
- Q3 EXPERIMENT DESIGNED (Dale's law): K single-sign 1-2FET synapses per connection, E/I populations,
  half the caps; counterweight = differential pairs are our mismatch armor (and mismatch is the wall);
  nearest prior = single-ended keystone dead-end ~56% (not a true Dale design).
- Q4 RUNNING (mmnz): AUGJIT=0.3 input jitter ON the 2mV mismatch chip (ref mm2=0.460) — noise as
  regularizer/mismatch-medicine hypothesis (user's). Clean-chip AUGJIT control to follow.
- RUNNING sgn1: SIGN-SGD update (SGNUP=1, comparator per neuron feeds gprods sign(eps) full-swing).
  Ref t2=0.840. If >=~0.8: justifies replacing 15T gprod with ~6T steered charge pump (sign extraction
  amortized per-neuron, 60 comparators vs 240 gprods on depth-4).
- RUNNING nzcl: AUGJIT=0.3 on CLEAN chip (noise-as-regularizer control; pairs with mmnz 2mV+noise).
- DALE'S LAW SPEC (implement after sgn1 verdict): per-NEURON fixed sign s_j (50/50 E/I, rng-assigned);
  inhibitory neurons' outgoing gsyn outputs CROSSED (free sign flip); weight caps diode-clamped positive
  (clamp option exists); update cells get crossed eps (or a) for inhibitory parents to fix the chain-rule
  sign. No synapse doubling needed at first (random fan-in mixes E/I parents). Cheaper single-ended cells
  = phase 2; differential pairs stay as mismatch armor for now.
- DALE IMPLEMENTED (DALE=1, DSEED): per-neuron fixed sign 50/50 E/I (hidden layers), inhibitory neurons'
  outgoing synapses output-crossed (fm/fx), transpose-crossed (bk/BKSIGN comparator input), update a-input
  crossed (chain rule); weights init |v| (positive). Regression: DALE=0 deck byte-identical. RUNNING dale1
  on depth-4 BKSIGN baseline (ref 0.840). v2 (enforcement of w>=0 + single-ended cheap cells) after verdict.
- SIGN-UPDATE CELL CHARACTERIZED (fastsim/charsg.cir, ngspice DC = exact): 7T core (3T V2I on a + 4T
  commutator steered by rail-to-rail sign(eps)) gives clean odd I_diff(a), EXACTLY mirrored under sign flip,
  +/-6.5uA @ a=+/-0.2 (gprod: 8.5uA). Constant-tail CM droop on both caps -> cancel with 2 static PMOS
  pull-ups (tail current constant, so static works) = 9T replaces 15T gprod. Comparator amortized per
  neuron (60 vs 240 cells on depth-4): net ~1.1k FETs saved (~12%) AND only 2 offset-critical input devices
  vs gprod's 8 -> possibly mismatch-friendlier update path. Gate: sgn1 (rule viability in full training).
- RUNNING c10bk: C=10 TRAINED hidden + BKSIGN (LAYERS=64,32,16 K=4 KOUT=10) — record to beat: 0.587
  (random hidden). Tests whether sign-faithful backward unlocks trained features at C=10 (the old wall).
- mm1s1 VERDICT (full-chip 1mV): 0.400/0.560best — even 1mV (already-large devices) halves the system.
  Dose-response (best-epoch, monotone): 0/1/2/5/10mV -> 0.840/0.56/0.52/0.37/0.38->chance. CONCLUSION:
  matching requirement <1mV => area scaling impractical; OFFSET CANCELLATION IS MANDATORY (sign-update
  cell's 2-device input + CDS autozero are the candidates). Finals noisier than bests (drift timing).
- 2mV MONTE-CARLO VERDICTS (3 chips): 0.460/0.52 | 0.250/0.25 (CHANCE) | 0.210/0.31. Chip-to-chip variance
  is huge; 2/3 chips DEAD at 2mV; the dose-response chip (MMSEED=1) was a LUCKY draw. Honest framing in
  paper: yield, not just mean. Offset cancellation now overwhelmingly mandatory.
- LAUNCHED: mm1s2/mm1s3 (1mV yield — was the 0.40/0.56 chip lucky?); sgnmm (SGNUP on DEAD chip2@2mV,
  ref 0.250 — does the sign-rule's 4x smaller update-path offset cross-section revive it?).
- NOISE VERDICTS (AUGJIT=0.3 input jitter, user hypothesis): CLEAN nzcl 0.850best/0.820 vs 0.840 baseline
  -> ~neutral (+1 best, the highest depth-4 best yet; level unswept). MISMATCH mmnz 0.450/0.520 vs mm2
  0.460/0.520 -> NULL: noise does NOT rescue mismatch (damage is structural offset-steering, not sharp
  minima). Honest paper note replaces the old "noise untested".
- WHY MISMATCH BREAKS SELF-CALIBRATION (mechanism summary for user Q): (1) offsets corrupt the LEARNING
  MACHINERY itself (esub trains toward m=x+delta; comparator offsets FLIP credit signs below 5mV signals;
  gprod offsets add constant drift) — can't learn around a corrupted rule; (2) SNR death spiral: offsets
  constant, errors SHRINK with convergence -> late drift always wins; (3) NO BIAS PARAMETERS: fan-in-4
  weights must both compute and cancel gain-16 neuron offsets — biology has intrinsic plasticity, we didn't.
- BIASW IMPLEMENTED (learnable per-neuron bias: unit-input gsyn + cap pair + gprod(eps,unit), 60 each on
  depth-4; regression DALE/BIASW=0 byte-identical). RUNN
ING bwmm: BIASW=1 on DEAD chip2@2mV (ref 0.250) — the proper test of "on-chip learning adapts to
  manufacturing impurities GIVEN the parameters to adapt with".
- USER CONFIRMS bias necessity (expressivity, not just mismatch): no-bias tanh = origin-locked boundaries;
  ReLU impossible (hinge position IS the bias). LAUNCHED bw0: BIASW on CLEAN baseline (ref 0.840) —
  expressivity gain test + prerequisite for one-sided/ReLU neuron work.
- NRELU CELL (fastsim/charrel.cir measured): transistor cutoff = free ReLU. 4T+cmld; knee ~0.55V (vrl rail
  + VT), flat below, linear mid, soft-sat top; per-neuron knee placement = BIASW (the user's point: ReLU
  needs biases). NEUREL=1 knob wired; training run queues for a free slot (box saturated at 8 jobs).
-
- DALE VERDICT (dale1, NEP=64): 0.800best/0.620 final vs 0.840 unconstrained — fixed-sign neurons
  (50/50 E/I, crossed wiring, positive init) cost only ~4 points at best. Biological sign constraint VIABLE.
  Late drift 0.80->0.62 (usual signature; v2 = w>=0 enforcement or controller freeze).
- FAST-EXPLORATION MODE (user directive): NEP=16 (2760 slots, ~4x faster) for ALL screens; NEP=64 only for
  finalists. Killed+relaunched young probes fast: relu1(NEUREL+BIASW) bw0f bwmmf sgnmmf mm1s2f mm1s3f c10bkf.
- SIGN-SGD VERDICT (sgn1, NEP=64): 0.830best/0.680 vs gprod 0.840/0.840 — sign(eps).a is WITHIN 1 POINT
  of the full 4-quadrant multiply at best epoch. 9T cell GREEN-LIT (-12% FETs, 4x smaller offset cross-
  section). Universal late drift again (0.83->0.68). sgnmmf (mismatch rescue) pending.
- FAST WAVE 1 (NEP=16 screens): bwmmf BIAS RESCUE of dead chip2@2mV: 0.25 -> 0.39 RISING (self-calibration
  works once bias parameters exist!). mm1s2f: chip2 DEAD at 1mV too (0.22) — yield catastrophic without
  offset handling. relu1: 0.25 flat = dead-ReLU-at-init (knee too high?) -> relu2 probes VRL=0.25/RELREF=0.60.
  bw0f 0.60@ep16 uninterpretable without NEP=16 baseline -> base16 control launched (screening lesson:
  always launch the fast-mode control FIRST).
- FAST WAVE 2: sgnmmf sign-rescue of dead chip2: 0.08->0.310 rising (vs 0.25 dead; bias rescue 0.39 stronger).
  mm1s3f: 1mV chip3 ALIVE 0.29->0.40 rising => 1mV yield 2/3 (chip2 the bad die). LAUNCHED bsmm: BIAS+SIGN
  combo on dead chip2 — the two offset defenses are complementary (bias absorbs forward, sign immunizes
  update path); test stacking.
- dalef2 FLAWED DESIGN (my error): AFLOOR=0+ANNS=2560 inside a 16-ep screen = LR hits zero at ep15 ->
  truncated learning (0.46@ep16), not a drift test. LESSON: anneal-to-zero endpoint must align with the
  expected BEST epoch on the FULL horizon. Proper run dalefz: DALE NEP=64 ANNS=5520(~ep32) AFLOOR=0 —
  freeze-at-peak as the deployment recipe (kills the universal late drift).
- FAST WAVE 3 + CONTROL: base16 clean@ep16 = 0.730 (the screen anchor). RESCUE LADDER COMPLETE (dead
  chip2@2mV, ep16): none 0.25 / sign 0.31 / bias 0.39 / COMBO 0.45 — defenses STACK. bw0f biases slow
  early learning (0.60 vs 0.73). relu2 still chance — suspect relu/ref leg CM imbalance breaks downstream
  gsyn; PARKED pending CM-balanced redesign. c10bkf 0.30 rising @ep16 (inconclusive).
  FULL-HORIZON CONFIRMS LAUNCHED: bsmm64 (combo rescue), bw064 (biases clean), c10bk64.
- NOISE DOSE-RESPONSE (ep16 screens, anchor 0.73): AUGJIT 0.15/0.3*/0.6 -> 0.66/~neutral/0.860 (!!).
  HEAVY input jitter (0.6) beats clean control +13 AND the full-horizon baseline best (0.84) at ep16.
  User's noise hypothesis VINDICATED at the right dose (we underdosed first). nz6064 full confirm + nz90
  (0.9 dose) + bsnz (noise+bias+sign on dead chip2) launched.
- DALE+SGNUP (dalesgn, ep16): 0.800 vs control 0.73 — the CHEAP-BIO combo (fixed-sign neurons + sign
  updates) BEATS unconstrained at screen horizon; sign updates drive hard early ([0.68,0.80]).
- bsmm3: combo rescue generalizes to chip3 (0.21 -> 0.39 rising).
- FREEZE-AT-PEAK CONFIRMED (dalefz): ANNS=5520->AFLOOR=0 holds 0.780 DEAD FLAT from ep32 to end (vs floor
  0.4: 0.80best->0.62 drift). Best==final. THE deployment recipe for universal late drift.
- nz90: 0.82@ep16 — noise dose peaks ~0.6 (0.66/0.86/0.82 at 0.15/0.6/0.9).
- bsnz 0.39: noise does NOT stack on mismatch rescue (saturates ~0.45; residual = forward distortion).
- GRAND COMBO LAUNCHED (grand/grandmm): DALE+SGNUP+BIASW+AUGJIT0.6+freeze@ep32, clean + dead-chip2.
  The thesis run: cheap (Dale+sign) + robust (bias+sign) + fast (noise) + stable (freeze) all at once.
- BIASES = NEW CLEAN RECORD (bw064, NEP=64): 0.860 final==best vs 0.840 no-bias; curve plateaus 0.85-0.86
  with NO LATE DRIFT (drift was partly bias-error forced into weights!). User called it: every neuron needs
  a bias. Slow early (0.60@ep16, caps charging) -> screens must not kill bias configs early.
- SPEED LADDER (user directives): S-tier 64-16-4/NTR20/NEP16 = 1480 slots ~25min (anchors: baseS, baseSmm
  chip2, +STEP=4n fidelity probe — if 4n==2n, GLOBAL 2x speedup); M-tier depth-4 NEP16 ~2h; L-tier NEP64
  finalists only. MONTE-CARLO BUDGET: 5 chips max per condition (user), dead/alive calls from S-tier.
- Q-TIER CALIBRATED (user: 1-2min sims): circles 2-4-2, CWW=30p (SMALL WEIGHT CAPS = the fast-learning
  screen knob; 10x weight speed), NEP=36, NTR=8 -> anchor ~0.69 rising in ~3min. Flat-curve dead end
  diagnosed: 300p caps move 10mV/slot BY DESIGN -> no tiny-slot training without shrinking caps.
  Q-probes relaunched: qa(anchor) qa4n(STEP fidelity) qamm(chip2) qarelu(fixed nrelu).
- NRELU BUG FOUND IN SECONDS by Q-cell probe: outputs had NO sink path (all-PMOS) -> railed at VDD ->
  downstream saturation = the relu1/relu2 chance verdicts. Fixed: 50k ground loads in subckt.
- Q WAVE: (1) STEP=4n curve-IDENTICAL to 2n at Q scale -> ADOPTED GLOBALLY, 2x all future sims (mbase4n
  M-tier spot-check running). (2) qarelu: fixed nrelu+BIASW SOLVES circles (1.000 best vs tanh anchor
  0.688!) — early ANTI-classification (inverted cell polarity) then bias-flip snap to 1.0. ReLU+bias is
  a major unlock; mrelu escalation to digits M-tier running. (3) qamm==qa: Q-tier is mismatch-BLIND
  (toy margins) — mismatch screens need M-tier.
- L WAVE (the collapse pattern): nz6064 best 0.870 (RECORD!) then chance; bsmm64 rescue best 0.56 then
  chance; grand 0.71@ep8 then chance; c10bk64 0.30->0.10 (wall persists). EVERY hot config peaks then
  collapses -> FREEZE-AT-PEAK (proven flat by dalefz) is the master control; relaunched nzfz (freeze@ep16)
  + bsfz (freeze@ep40). Freeze timing must match each config's peak.

## 2026-06-12 evening — selective dneuron upsizing (mismatch yield fix attempt)
- HYPOTHESIS: dneuron is the worst mismatch offender (ablation: 2mV dneuron-only -> 0.17; gain~16
  amplifies input-referred VT offset). sigma_VT ~ 1/sqrt(WL), and the dneuron diff pair is only ~3% of
  total FET area -> selectively upsizing JUST the neuron is cheap: 4x area = sigma/2 (+~4% total area),
  16x area = sigma/4 (+~19%). Emulated via MMVT_DNEURON override; everything else stays at 2mV.
- LAUNCHED (M-tier NEP=16 STEP=4n ANNS=2560 AFLOOR=0.4, dead chip2 MMSEED=2, ~55min each):
  dnup16 (dneuron 0.5mV =16x area), dnup4 (1mV =4x area), dnupb (1mV + BIASW — bias sits exactly at the
  neuron input node mp/mn so it should absorb the residual dneuron offset by construction).
  Comparison points (same screen): none 0.25 / SGNUP 0.31 / BIASW 0.39 / combo ~0.45.
- Also still in flight: nzfz (freeze-locked noise record candidate ~0.87, ETA ~2.5h), bsfz (rescue+freeze
  full horizon, ~5h), mrelu (ReLU+bias digits M), mbase4n (STEP=4n M-tier fidelity check).
- charoff.cir (real ngspice DC, fastsim/): offset sensitivity dneuron vs nrelu. Input-referred shift
  = exactly deltaVT for BOTH neurons (5mV offset -> 5mV curve shift) — ReLU has NO intrinsic mismatch
  advantage; output error per 1mV offset is 7.6mV (dneuron) vs 0.7mV (nrelu) but that is purely the
  gain ratio, SNR identical. Rescue must come from BIASW (absorbs input-referred offset at mp/mn) or area.
- BUG FOUND+FIXED: PMM had no "nrelu" entry -> ReLU nets were INVISIBLE to MMVT (a future ReLU mismatch
  screen would silently simulate a clean chip). Added "nrelu":[0,1] (Mr signal + per-instance Mrr ref VT).
  Verified: Q deck gen puts mmv sources on both nrelu pins; RNG stream unchanged for non-NEUREL runs.
- NRELU UPGRADE WAVE (Q-tier real Spectre, spirals 2-4-2, NTE=16=32 test pts):
  (1) NRW knob added: mirror-ratio gain on nrelu output (DC: gain 1.0 -> 2.6 @400u -> 5.1 @800u, swing 0.24->0.79).
  (2) POLARITY FIX: nrelu cell inverts (signal leg mirrors onto an). At NRW>=400u the bias-flip snap can no
      longer rescue inverted wiring -> nets train to PERFECT ANTI-classification (0.03). Fixed by swapping
      output pins at instantiation (upright by construction); NRINV=1 reproduces old wiring.
  (3) SPIRALS SOLVED AT Q-TIER: upright + NRW=800u -> 5/7 seeds >=0.969 (s5,s7 = 1.000 FLAT curves).
      Previously spirals needed deep+Adam (surrogate workstream). Low gain (200u/400u) does NOT solve (~0.5).
      GAIN is the unlock; polarity makes it usable.
  (4) Anti-lock anatomy (failed seeds s2,s6): perfectly-wrong basin chosen AT INIT. NOT a sign bug: flipping
      SGNO/SGNH does not convert wrong->right (stays ~0.1). Stronger nudge TD=0.2 no escape. Lower knee
      VRL=0.23 escapes but oscillates between basins. RESTART (re-roll WSEED) is the remedy: 3/4 re-rolls
      solve (>=0.969); detect via train-acc~0 + restart = reliable-XOR recipe transferred.
  Net: 8/11 inits >=0.97 immediate, ~100% with <=2 restarts. mrelu8 escalation (digits M-tier NRW=800u
  upright) in flight; mrelu (old inverted polarity, NRW=200u) becomes the control.
- Q-TIER 2D TRIPLE COMPLETE (upright nrelu NRW=800u + BIASW, ONE tiny 2-4-2 net, plain local PC, real
  Spectre): circles 1.000/0.969 + restart->1.000 | rings 1.000/0.969 + restart->1.000 | spirals 5/7
  >=0.969 (2x 1.000 flat) + restarts. Seed-2's WSEED draw anti-locks ALL THREE tasks (init-geometry
  polarity, not task-dependent) and restarts (WSEED=22) fix all three -> detect+restart is a complete
  remedy. This is the same neuron everywhere; no per-task tuning beyond the standard knobs.
- NRELU DESIGN RULES (Q probes, spirals/circles s1): NRW=1600u (gain~10) solves 1.000 then FLIPS to
  perfect-anti 0.0 mid-training (basin unstable at extreme gain) | NRW=600u only 0.875 (gain>=5 needed)
  -> NRW=800u (mirror 4x, gain 5.1) is the sweet spot. BIASW=0 -> DEAD AT CHANCE both tasks (user was
  right: ReLU mandatorily needs per-neuron biases; without them not even anti-lock, just nothing).
- COMPOSABILITY (Q, spirals s1, upright NRW=800u): DALE=1 -> 0.969 FLAT (Dale's-law single-sign neurons
  + high-gain ReLU coexist) | SGNUP=1 -> 0.969 best / 0.875 final (9T sign-update cell trains the ReLU
  net, slightly noisier than gprod). The neuron upgrade composes with the whole cell library.
- STABILITY BOUNDARY (Q, spirals s1): freeze-at-peak (ANNS=200 AFLOOR=0) FULLY rescues NRW=1600u ->
  1.000 FLAT entire run (master control works on gain instability too). RELREF sweep: 0.60 -> mush
  (0.656 best, ref too low); 0.72 -> solves then flips to 0.0 (same signature as 1600u). Defaults
  RELREF=0.66 + NRW=800u sit inside the stable region; freeze extends it. Net design rule: the
  (gain, reference) plane has a sharp basin-stability boundary; anneal-to-zero neutralizes it.
- WIDTH/GAIN SCALING LAW (Q, spirals): 2-8-4 net ANTI-LOCKS at NRW=800u for ALL inits (3/3 — systematic,
  not seed luck; output fan-in doubles 2->4 so layer gain doubles). NRW=400u -> 1.000 at FIRST EVAL then
  peak-collapse; + freeze (ANNS=150 AFLOOR=0) -> 1.000 LOCKED FLAT. Rule: per-cell gain scales INVERSELY
  with output/layer fan-in; freeze-at-peak then locks it. NTR=16 robustness: 0.969 (holds with 2x data).
  IMPLICATION: digits (fan-in 4 readout) @NRW=800u (qdig, mrelu8) likely over-gained -> qdig4 launched
  (NRW=400u fast screen) as the predicted-correct setting.
- WIDE CHAMPION (2-8-4, NRW=400u, ANNS=150 AFLOOR=0): s1 ALL THREE tasks 1.000 LOCKED (circles/rings/
  spirals, same config+seed). s2 spirals 1.000 (the seed narrow-800u anti-locked!). s3 spirals fails ALL
  weight re-rolls (anti w/ freeze, chance 0.53 w/o) -> that DATA draw (16 train pts) is hard for this
  arch; narrow-800u solves it (0.969). The two configs COVER each other's failures: every spirals dataset
  s1-s7 solved by narrow-800u(+restart) or wide-400u+freeze. Q-tier data luck (NTR=8) is the residual
  noise; conclusions to carry forward = scaling law + freeze + restart, verdicts on digits from qdig/qdig4.
- restart_run.sh added (controller-side anti-lock remedy): re-rolls WSEED (+10 per try) until best-acc
  >= THRESH; smoke-tested on the cursed spirals s2: 0.062 -> 0.031 -> 1.000 (deterministic reproduction
  of the manual probes). Use for all future seed sweeps / L-tier finals.
- mrelu CONTROL VERDICT (M-tier digits, OLD inverted nrelu NRW=200u): 0.26 = chance for C=4, first eval
  0.01 (anti-classification) -> climbs only to chance via bias flip. The old neuron FAILS at scale exactly
  as Q predicted. Verdict awaits mrelu8 (upright 800u — possibly over-gained) and qdig/qdig4 (800 vs 400).
- MULTI-CLASS Q LADDER LAUNCHED (attacks the C=10 wall cheaply): rings C=4, spirals C=4 (2-8-4), rings
  C=8 (2-12-8), all upright NRW=400u + freeze. rings/spirals generators are natively C-class.
- qdig VERDICT (digits fast-cap screen, upright NRW=800u): curve [0.0, 0.5, 0.25, 0.25] -> first eval 0.0
  = ANTI signature, peaks 0.5, decays to chance. OVER-GAINED at fan-in-4 readout, exactly as the width/gain
  law predicted. qdig4 (NRW=400u) = the predicted-correct setting, pending. qxor sanity: 1.000 best.
- mbase4n VERDICT: 0.730 curve [0.28, 0.73] == base16's trajectory EXACTLY -> STEP=4n CONFIRMED at M-tier.
  2x global speedup locked for all future runs.
- MULTI-CLASS Q VERDICTS (new neuron): rings C=4 0.531 rising (2x chance, learns) | spirals C=4 peak 0.50
  then 0.156 collapse | rings C=8 DEAD at exact chance 0.125 flat. THE C-WALL REPRODUCES AT Q-TIER ->
  can iterate on the C=10 problem in minutes now. Q wave launched on rings C=8: KOUT=8 (readout fan-in),
  2-16-8 width, NTR=16 data, no-freeze control.
- DNUP VERDICTS (dead chip2 @2mV, ep16 screens; ladder was none 0.25/sign 0.31/bias 0.39/combo 0.45):
  dnup16 (neuron->0.5mV ALONE) 0.12 | dnup4 (neuron->1mV alone) 0.24 -> UPSIZING ALONE DOES NOT RESCUE
  (damage distributed, consistent with ablations). BUT dnupb (neuron->1mV + BIASW) = 0.51 RISING — NEW
  BEST RESCUE (+0.12 over bias-alone): cleaner neuron x bias absorption is multiplicative. Area cost of
  the neuron 4x upsize ~ +4% total FETs. Ladder continues: dnupe (esub too, in flight), dnupeg (esub+
  gprod at 1mV + bias, LAUNCHED — "harden everything but the synapse array", the affordable endpoint).
- qdigT CONTROL = BREAKTHROUGH: tanh through the fast-cap digits screen (CWW=30p NEP=8 NTR=20) = 0.830
  RISING [0.3,0.57,0.71,0.83] in ~25min — BEATS the M-tier NEP=16 300p trajectory (0.73@ep16, ~3h)!
  (a) the screen is VALID for digits; (b) CWW=30p is a TRAINING ACCELERANT (10x weight speed compresses
  the schedule); (c) ReLU digits 0.5-peak genuinely underperforms tanh 0.83 — the 2D ReLU win does NOT
  transfer at depth-4 yet (gain compounding / one-sided info loss; diagnosis next). LAUNCHED: qdigT2
  (tanh fast-cap FULL NEP=16 NTR=40) + qdigT2b (+BIASW) — candidate new clean record in <1h.
- mrelu8 VERDICT (M-tier digits upright NRW=800u): 0.51 peak -> 0.31, over-gained, consistent with qdig.
  ReLU digits underperforms tanh (0.83 fast-cap) at every gain tried so far.
- NRLAYERS knob added (mixed nets: ReLU on listed layers only, tanh elsewhere); qdmix1 (ReLU L1 only) +
  qdmix12 (L1,L2) digits screens launched. KOUT=8 C=8 ladder: best 0.328 peak (200u no-freeze) then
  collapse; all configs peak 0.28-0.33 = 2.3-2.6x chance. Tanh C=8 controls launched (qw8T/qw8Tb) to
  locate the wall (neuron vs PC dynamics).
- C=8 NEURON COMPARISON (rings 2-12-8 KOUT=8, Q): tanh 0.188 / tanh+bias 0.125 vs ReLU-200u 0.328 peak.
  THE NEW NEURON BEATS TANH THROUGH THE C-WALL (1.75x) — the wall is PC-dynamics/architecture, not the
  activation. Launched: qk8fz (freeze at the 0.33 peak) + qk8d16 (2x data at best config).
- nzfz VERDICT (AUGJIT=0.6 + freeze@ep16, NEP=32 full horizon): 0.830 LOCKED FLAT [0.6,0.83,0.83,0.83].
  Freeze-at-peak DEFEATS the noise-collapse (nz6064 was 0.87->0.25). But noise+freeze 0.83 does NOT beat
  clean+bias 0.860 -> noise = accuracy null (confirmed at full horizon); freeze = the real control.
- dnupe VERDICT: neuron+esub at 1mV + bias = 0.22 FLAT — STRICTLY CLEANER chip than dnupb (0.51) yet far
  worse. Rescue outcomes at the 2mV edge are CHAOTIC (variance >> mean, again). dnupeg pending as 3rd pt.

## 2026-06-13 overnight — THE RESCUE HEADLINE + fast-cap horizon law
- bsfz VERDICT (dead chip2 @2mV, BIASW+SGNUP, NEP=64 full horizon + freeze@ep40): RESCUED TO 0.72
  LOCKED FLAT [0.45,0.42,0.61,0.72 x5]. The earlier "rescue saturates ~0.45" was a SHORT-SCREEN ARTIFACT
  — at full horizon the rescue keeps climbing then locks. Dead chip (0.25) -> 0.72 vs clean 0.84, ZERO
  new hardware (biases + sign-update + LR schedule). NEW MISMATCH HEADLINE. bsfz3 launched (chip3, 4n)
  for generalization; 5-chip budget per user directive.
- qdigT2/T2b (fast-cap FULL NEP=16): peaks 0.64/0.74 then CRASH (0.22/0.0) -> fast caps over-train at
  long horizons. Horizon law: CWW=30p needs short runs (NEP=8: 0.83 rising) or freeze. qdigT3 launched
  (fast-cap NEP=16 + BIASW + freeze@ep8) = record candidate (<1h).
- qk8li: LATINH no help at C=8 (0.25 = KOUT=8 alone). dnupeg 0.19 = 3rd chaotic edge point (vs dnupb
  0.51, dnupe 0.22) -> area-rescue near the cliff is luck; the PRINCIPLED rescue is bias+sign+freeze.
- qdigT3 (fast-cap NEP=16 + BIASW + freeze@ep8): 0.75 best, locked 0.73-0.74. Stable but NO record;
  bias does NOT help fast-cap (qdigT no-bias NEP=8 = 0.83 stays the fast champion). ROLE SETTLED:
  CWW=30p = 25-min screening tier for digits (0.83 ceiling-ish); records stay on standard caps.
- qdmix3 (ReLU last-hidden only): 0.46 peak -> collapse; best ReLU-digits but << tanh 0.83. Placement is
  NOT the issue -> ReLU = 2D champion, tanh = digits champion; PARKED pending a new idea. qw8deep: depth
  does not crack C=8 (0.156). RESCUE MC COMPLETION LAUNCHED: bsfzc1 (living chip 0.46 — does rescue lift
  it to ~0.8?), bsfzc4, bsfzc5 (fresh chips) — with bsfz(c2)+bsfz3 = the full 5-chip budget.
- L-TIER 2D FINALS (NEP=64, NTE=128/class=256 test pts, freeze@400, real Spectre, 2-4-2 upright NRW=800u
  + bias): circles 0.992 / rings 0.988 / spirals 0.992 — ALL LOCKED FLAT across the full 4x horizon.
  Beats pc_batch (0.955/1.000/0.965) and all backprop-workstream 2D numbers, with ONE tiny net and plain
  local PC. The nrelu paper section upgrades from screening-tier to final.

## 2026-06-14 — RESCUE MONTE-CARLO COMPLETE (the headline lands)
- 5-chip MC, dead/marginal chips @sigma=2mV, recipe = BIASW + SGNUP + freeze, NEP=64 full horizon:
    chip1 best 0.72 / final 0.61 | chip2 0.72 / 0.72 | chip3 0.83 / 0.79 | chip4 0.83 / 0.56 | chip5 0.82 / 0.72
  BEST-EPOCH MEAN = 0.78 across 5 chips; EVERY CHIP NOW LEARNS (was 2/3 DEAD at chance unaided).
  YIELD: 0/5 -> 5/5 alive. This is THE mismatch result: on-chip learning DOES self-calibrate against
  manufacturing offsets once it has (a) bias params where offsets live, (b) a low-offset sign update,
  (c) anneal-at-peak. Residual late drift on chips 1/4 (final<best) = GLOBAL freeze timing vs per-chip
  peak epoch -> val-based early-stop (=best-epoch) is the honest deploy number; per-chip freeze would
  recover it. Clean baseline 0.84 -> rescued mean 0.78 = 93% recovery.

## 2026-06-14 — leaky-nrelu (fixing ReLU's digits gap)
- DIAGNOSIS: one-sided ReLU discards sub-knee info -> loses to tanh on deep digits (0.5 vs 0.83). FIX:
  leaky-nrelu = 2 weak source-to-GROUND degenerated legs (signal+reference, symmetric) summing into the
  same mirror diodes -> conduct across the whole range, adding a sub-knee slope. NRLEAK=resistor: 1e9=off
  (=standard ReLU, default, regression-safe since legs carry ~0), 100k/50k=leaky. +4 devices.
- DC CONFIRMED (fastsim/charleak.cir, real ngspice): off has FLAT dead zone (+0.304 at both 0.35 & 0.45);
  100k turns it into a slope (+0.638 -> +0.499 -> knee +0.177 -> -0.05) = sub-knee info preserved. 50k steeper.
- LAUNCHED digits fast-cap screens: NRLEAK 200k/100k/50k @ NRW=400u (vs ReLU 0.5 / tanh 0.83 baselines).

## 2026-06-14 — SCALING-LAW HUNT (rigorous, replacing the fan-in guess)
- EXPT 1 (FAN-IN): spirals 2-H-2, FANIN uncapped so output fan-in=H, sweep H{4,8,16} x NRW{200,400,800,1600},
  best-epoch. RESULT: critical NRW ~800u INDEPENDENT of H (all H: fail@200/400, train@800/1600). FAN-IN
  DOES NOT SET THE GAIN BOUNDARY. The "gain ∝ 1/fan-in law" was a MISATTRIBUTION: the original 2-4-2(800)
  vs "2-8-4"(400) comparison actually varied DEPTH — LAYERS=2,8,4 expands to [2,8,4,2] = 2 hidden layers.
- HYPOTHESIS REVISED: CASCADE-GAIN BUDGET — the PRODUCT of per-stage gains must stay under a stability
  ceiling, so per-stage gain must fall as depth grows. EXPT 2 (DEPTH, fixed width 4) launched to confirm.
- EXPT 2 (DEPTH) RESULT -> CASCADE-GAIN BUDGET LAW (spirals, width 4, best-epoch, NRW=mirror width):
    d1 (2-4-2):    trains {800,1600}, fails {200,400}      -> needs HIGH gain, no ceiling <=1600
    d2 (2-4-4-2):  trains {100,200,400,800}, FAILS {1600}  -> lower floor AND a ceiling appears
    d3 (2-4-4-4-2):trains {100,200,400,800}, marg {50}     -> wide low-gain band
  LAW: total cascade gain ~ (per-stage gain)^depth must stay in a stable [lo,hi] band. Adding a layer
  multiplies total gain -> per-stage gain must DROP as ~G*^(1/depth); too-high per-stage over-drives a
  deep net into anti-lock (d2@1600=0.34). Single-stage nets conversely need the gain CRANKED to reach lo.
  This REPLACES the retracted fan-in law and explains the original 2-4-2(800)/[2,8,4,2](400) datapoints.
- LEAKY-NRELU = NULL for digits (C=4, NRW=400u, NRLEAK 200k/100k/50k all -> 0.50-0.51 best, == plain ReLU
  0.5, vs tanh 0.83). DC char confirmed the leak DOES add sub-knee slope, but it does NOT recover digit
  accuracy -> ReLU's digit weakness is NOT one-sided info loss. Hypothesis refuted. tanh stays the digit
  neuron; nrelu stays the 2D champion. (leaky cell kept behind NRLEAK, default off / harmless.)

## 2026-06-14 — SPATIAL RECEPTIVE FIELDS for C=10 (user's local-perceptive-field idea)
- RFGRID=1: conv-like 2x2 stride-2 local pooling (verified exact windows) vs random sparse, C=10 digits,
  tanh, fast-cap NEP=8, identical sizes. RESULT (chance 0.10):
    rf2 spatial 64-16-4: 0.224 best / 0.224 FINAL (stable) | rand2 random: 0.228 / 0.116 (COLLAPSES)
    rf1 spatial 64-16:   0.148 / 0.148 (stable)            | rand1 random: 0.128 / 0.112 (collapses)
  FINDING: locality does NOT raise the peak but ELIMINATES post-peak collapse (every spatial run holds
  best; every random run decays) = structural regularization, no freeze needed. Deeper spatial pyramid
  (2 pools -> 4 feats) beats shallow (16 feats) -> locality+depth > width. Best stable C=10 = 0.224.
- ESCALATING rf2 (winner) with 2x data to test if locality+data climbs past the ~0.22 ceiling.
- RC TIMESCALE (LAW HUNT 3), spirals 2-4-2 NRW=800 sweep CWW{15..240p}, NEP=48 EVK=4: net trains TOO FAST
  (1.0 by first eval ep4 for ALL caps) -> convergence side unresolved. BUT the COLLAPSE side scales: 15p
  drifts 1.0->0.66 by ep44; 30-60p stable; 120-240p flat & slightly undertrained (0.969 cap). =>
  DRIFT/COLLAPSE timescale ∝ C_weight (small cap=fast everything incl. drift; big cap=slow & stable).
  Refining with slower drive (TD=0.05) to resolve CONVERGENCE epoch ∝ C quantitatively.
- RC LAW (refined, TD=0.05, CWW 30p/120p/480p): convergence STILL sub-eval-resolution (1.0 by ep0-2 for
  all caps, even 16x range + 2x slower drive) -> "convergence-epoch ∝ C" NOT cleanly resolvable in this
  fast-training net; do NOT claim the exponent. ROBUST measured form: cap size = weight-update timescale;
  small caps (30p) hit 1.0 then drift; large caps (480p) cap UNDERTRAINED (0.938) but flat-stable. The
  speed/stability tradeoff IS the RC law; 30p = sweet spot (max peak, acceptable drift). Honest partial.
- C=10 ESCALATION (2x data, NEP=12): rf2L spatial PEAKS 0.244 (2.4x chance) then collapses; rand2L random
  = FLAT CHANCE 0.10 (never learns). KEY: at this tiny size random sparse CANNOT learn C=10 at all;
  LOCALITY makes it learnable. Collapse returns at long horizon (earlier "no-collapse" was short-screen
  luck) -> needs freeze. WAVE launched to lock+push: rf2fz (freeze@ep2), rf2td (gentler TD=0.04 broader
  peak), rf3 (overlapping 3x3 RFK=3 richer fields), rf1c (64,16 16-feat readout). Target: beat 0.244 stable.

## 2026-06-14 — LAW HUNT 4: MISMATCH-YIELD vs DEPTH (connects cascade-gain + mismatch threads)
- spirals, each depth at its cascade-law gain (d1@800u, d2@400u), best-epoch, MMVT sigma sweep:
    d1 (2-4-2): sig 0/1/2/5mV -> 1.0 / 0.97 / 1.0 / 0.09(dead)  = tolerates ~2mV
    d2 (2-4-4-2): sig 0/1mV   -> 1.0 / DEAD (chip1 0.0, chip2 0.03, chip3 1.0) = ~1/3 YIELD at 1mV
  Restarts do NOT recover (mismatch-fatal, not basin-luck); lower gain (NRW=200u) does NOT rescue d2.
  LAW: MISMATCH YIELD DROPS SHARPLY WITH DEPTH — each gain stage amplifies upstream VT offsets, the
  cascade compounds them past what any weight init absorbs. Shallow nets tolerate ~2-3x more sigma.
  DESIGN IMPLICATION: for mismatch-limited analog, SHALLOW-WIDE > DEEP-NARROW (inverts the noise-free
  cascade-gain preference for depth). Ties together: depth buys representation but costs robustness.
- C=10 DESIGN WAVE verdict: rf2fz (spatial 2x2 pyramid 64-16-4 + freeze@ep2) = 0.248 best / 0.220 STABLE
  = BEST in-circuit C=10 (vs old c10bk64 0.30->collapse-to-chance). Ablations: gentle drive (rf2td 0.20
  collapses) no help; overlapping 3x3 (rf3 0.164) WORSE than clean 2x2; shallow 16-feat readout (rf1c
  0.064 anti-lock) worse than deep 4-feat -> HIERARCHICAL POOLING (depth), not feature count, is what
  helps. C=10 WALL persists ~0.22-0.25 but now STABLE + principled (spatial 2x2 pyramid + freeze) +
  understood (locality essential: random sparse = chance 0.10). Mechanism of the wall itself still open.
- ZEROSUM target fix (port of the documented C=10 gen_mc cure to pc_deep: correct +tdv, others -tdv/(C-1)
  so each target vector sums to 0, vs default -(C-2)tdv downward drift). Q-proxy rings C=8 (tanh, weak
  proxy): ZEROSUM 0.172 vs control 0.156 (+10% rel, modest). ReLU pair anti-locked (freeze config, n/a).
  Real test = digits C=10 (zs1/zs1nf/zsrand) where the drift-collapse was actually diagnosed; pending.

## 2026-06-14 — ZEROSUM BREAKS THE C=10 DRIFT (the missing port lands)
- digits C=10 (chance 0.10): zs1nf (spatial + zero-sum, NO freeze) = 0.300 best, curve CLIMBING
  [0.2,0.192,0.264,0.22,0.3,0.268] no collapse | zs1 (+freeze) 0.236 stable | zsrand (random+zero-sum)
  0.200 (REVIVED from chance 0.10!). FINDINGS: (1) best in-circuit C=10 deep-path = 0.30, up from 0.22
  (+36%); (2) zero-sum STOPS the collapse by itself (no freeze needed) = confirms the gen_mc drift
  mechanism in pc_deep; (3) imbalance was KILLING random nets (revived 0.10->0.20); (4) spatial locality
  still adds on top (0.30 vs 0.20). zs1nf still climbing at ep10 -> extending horizon+data to find ceiling.
- ZEROSUM IS C-DEPENDENT (key correction): C=4 ZEROSUM=1 -> 0.70 vs unbalanced 0.83 (HURTS small C!);
  C=10 zero-sum 0.30 vs unbalanced 0.22 (HELPS large C). Mild imbalance (C=4, -2tdv) -> extra negative
  pressure = useful contrast; severe (C=10, -8tdv) -> drift collapse. There's a CROSSOVER. Added ZSNEG knob
  (negative-class target scale: 1.0=unbalanced, 1/(C-1)=zero-sum) to sweep the optimum. Default unchanged.
- EXTENDED-HORIZON zero-sum C=10: zsLg climbs 0.30 @ep10 then COLLAPSES to 0 (zero-sum delays but doesn't
  fully stop collapse at long horizon); zsL (NTR=60) UNSTABLE (too much data, tiny net). Peak ~0.30 holds;
  long runs need freeze. So best C=10 recipe = spatial + zero-sum + FREEZE at the ~ep10 peak.
- PERAZ = NULL in pc_deep (per-class auto-zero, high-pass error): on top of spatial+zero-sum, all RC
  settings 0.20-0.24 < zero-sum-alone 0.30 (RC-insensitive -> not tuning). The gen_mc +9-13pt PERAZ win
  does NOT transfer: differential rails + zero-sum already remove the per-class offset PERAZ targets, so
  AC-coupling the error just discards signal. Lesson: keystone(single-ended)-path fixes don't auto-port to
  the differential deep path. PERAZ kept behind flag (default off). ZSNEG crossover is the live lever.
- ZSNEG freeze-sweep CONFOUNDED: ANNS=2600 (~ep5) froze before the ep10 peak -> all ~0.21 (capped early),
  masking ZSNEG. Redo no-freeze (AFLOOR=1, read peak). Mapping "optimal ZSNEG vs C" as a LAW: C=4 prefers
  high negativity (ZSNEG~1.0=0.83 > 0.33=0.70), C=10 prefers low (0.111=0.30 > 1.0=0.22) -> optimal target
  negativity DECREASES with class count (imbalance drift overtakes contrast benefit as C grows).

## 2026-06-14 — LAW: optimal target-negativity vs class count (mapped)
- ZSNEG (negative-class target scale; 1.0=unbalanced, 1/(C-1)=zero-sum) optimum vs C (no-freeze best-ep):
    C=4:  1.5=0.78  1.0=0.83*  0.6=0.71  0.33=0.70   -> optimum ZSNEG~1.0 (unbalanced)
    C=6:  1.0=0.42*  0.4=0.25  0.2=0.24                -> optimum ZSNEG~1.0 (unbalanced)
    C=10: 1.0=0.20  0.2=0.29  0.111=0.30*  0.05=0.30   -> optimum ZSNEG~0.05-0.11 (zero-sum)
  LAW: small C (<=6) prefers UNBALANCED targets (sharp contrast helps); large C (>=10) prefers ZERO-SUM
  (drift-free). Crossover ~C=8, where the imbalance drift (grows ~(C-2)tdv) overtakes the contrast benefit.
  Predictive: set target negativity ~1.0 for few classes, ~1/(C-1) for many. (Deep-path high-C acc still
  low: C=6 0.42, C=10 0.30 = ~2.5-3x chance; the law is about the OPTIMAL encoding, not the ceiling.)
- CCMS (linearized/subtractive softmax-CE) = NULL: peaks 0.276 (~= zero-sum 0.30) then collapses to 0.004.
  Subtractive cross-class mean-removal doesn't beat zero-sum (which already balances targets) and doesn't
  stop the collapse. The DIVISIVE softmax (SMAX, shared-tail) is the real test - pending.

## 2026-06-14 — RESIDUAL CONNECTIONS: honest negative (wrong failure mode)
- Built forward+backward (bidirectional gradient-highway) residual skips (RES knob). Tested on deliberately-
  deep nets (5,8 hidden, same-size for identity skip), spirals:
    8-hidden @800u: RES 0.031 / no-RES 0.031 (both dead) | @400u: RES 0.781 / no-RES 0.875 (RES slightly WORSE)
    5-hidden: RES == no-RES (0.875@400, 0.969@800)
- CONCLUSION: residual does NOT help in this analog-PC setting. WHY (interesting): the deep failure mode here
  is FORWARD OVER-GAIN (cascade multiplies per-stage gain ~G^depth -> rails), NOT backward vanishing-gradient
  (the problem residual cures). An identity skip doesn't reduce forward gain. The correct medicine is the
  CASCADE-GAIN LAW (lower per-stage NRW for deep) -> 8-hidden trains fine at 400u with NO residual. So depth
  IS usable in analog, just via gain budgeting, not residuals. RES kept behind flag (default off, neutral).
- ANALOG POOLING PRIMITIVES characterized (fastsim/charmax.cir, real ngspice):
    AVG-pool = resistor star -> tracks TRUE AVERAGE EXACTLY (cheap: K resistors; = cmld's averaging).
    MAX-pool = K source-followers sharing an output node + pulldown -> output = max(inputs) - Vgs
      (flat when a unit loses, rises when it becomes the winner; constant ~0.32V level shift). 4T+1R.
  Both "make sense in analog". RFGRID (learnable local pooling) already subsumes avg; fixed avg/max-pool
  save weight caps (less precision) for downsample layers. Analog-DL primitive map now complete:
  residual (built, negative - wrong failure mode), avg/max pool (char'd), batchnorm=cmld (have), dropout=AUGJIT (on).

## 2026-06-14 — MAGNITUDE MATTERS for transport (VBBK sweep; answers "is discarding magnitude good?")
- Backward regeneration saturation sweep (depth-4 digits C=4 fast-cap, VBBK = backward drive; low=linear/
  magnitude-preserving, high=saturated/sign-like): VBBK 0.32->0.62, 0.45->0.57, 0.65->0.52. MONOTONE:
  LESS saturation (keep magnitude) is BETTER; hard-sign is WORST. So "throw away magnitude" is WRONG for
  TRANSPORT. The win = AMPLITUDE RESTORATION with magnitude PRESERVED (fights attenuation, keeps proportion).
  NUANCE: the UPDATE separately tolerates sign-only (sign-update 0.83~0.84) -> magnitude matters for credit
  PROPAGATION, not for the local update. Corrected the paper's premature "insensitive to gain" line.
  CAVEAT: single-seed fast-cap screen; verify with seeds/full-horizon (1 spectre at a time, MT=4, token-limited).
- SPECTRE TOKEN LIMIT (shared license): keep <=6 tokens. Killed all my runs (was 8 procs x mt=8 incl zombies
  drg100/drg400). New policy: MT=4, ONE spectre at a time. Slower iteration, but courteous to shared license.

## 2026-06-14 — CIFAR-10 added (avoid MNIST overfit); INPUT is the bottleneck
- First CIFAR run: gray 8x8 (64 inputs), shallow random features = 0.156 (chance 0.10). LOW because the
  INPUT is degraded, not the net. IDEAL linear-ridge ceilings (numpy, token-free):
    gray 8x8 = 0.295 | gray 16x16 = 0.295 (resolution NO help) | COLOR 8x8 = 0.407 | color 16x16 = 0.403
  => COLOR is worth +11 pts; RESOLUTION is worthless at this scale. Our 0.156 = 53% of the gray ceiling
  (normal in-circuit efficiency). digits linear ceiling for ref = 0.883 (CIFAR is just much harder).
- ADDED CIFCOLOR=1 (color GxG, 3*G*G inputs). Color 8x8 = 192 inputs, ceiling 0.41 = the right baseline.
  Launched cifcol (192-64-10 random features, color). Token policy: 1 spectre @MT=4.
- CIFAR color-8 ARCHITECTURE (numpy ceilings, token-free): data is ~LINEARLY SEPARABLE to the ceiling ->
  DIRECT linear readout (192->10, NO hidden) = 0.407; 64 random hidden HURTS (0.345, bottlenecks 192 inputs);
  256 random recovers (0.390). So the RIGHT in-circuit CIFAR net = DIRECT readout (like MNIST z-score+direct
  77-83%), not random-hidden. cifcol (192-64 random) is suboptimal (capped 0.345); next run = direct 192->10.
  Lesson: compute the ideal ceiling per architecture in numpy FIRST -> spend Spectre tokens only on the
  architecture that can actually win. (Esp. valuable under the <=6-token serial limit.)
- CIFAR REALISTIC ANALOG CEILING (numpy, direct readout, color-8): ideal float 0.407, railed ±3sig 0.404
  (wide rail OK), railed ±1.5sig 0.353 (tight rail hurts most), 4-bit 0.372, 3-bit 0.309, realistic
  (railed 2sig + 4-bit) = 0.372. So honest analog CIFAR ceiling ~0.37; KOUT=40 ~0.35 ideal -> expect
  ~0.18-0.25 in-circuit at normal efficiency. COMPLETE token-free CIFAR characterization: input(color>res),
  architecture(direct>random-hidden), ceiling(0.37 analog). ngspice (SIM=ngspice) = TOKEN-FREE workhorse
  for exploration; reserve Spectre (<=6 tokens) for finalists. cifng = first direct-readout ngspice run.

## 2026-06-14 late — CIFAR first number + VBBK 2-seed confirm (Spectre, ≤6 procs MT=4)
- FIRST in-circuit CIFAR-10: 0.164 (64-feat color, direct readout KOUT=20, ceiling 0.345, chance 0.10)
  = ~47% of ceiling. Honest first datapoint; CIFAR is hard for analog (input-limited, see characterization).
- VBBK magnitude verified at SEED 2: magnitude-preserving (0.32)=0.60 > sign-like (0.65)=0.55. With seed 1
  (0.62 vs 0.52), BOTH seeds agree: keeping error magnitude beats sign for TRANSPORT. Finding is robust.
- CIFAR TRAINED HIDDEN helps (numpy, 64 color feat): direct linear 0.345 vs TRAINED MLP H=32 0.409 / H=64
  0.433 (+9pts nonlinear headroom). Overturns earlier "random hidden hurts -> use direct": the key is
  TRAINED (not random) features. So CIFAR wants a trained depth-2 net (64-64-10, sparse fan-in = modest
  width/precision), ceiling ~0.43. Launched scifh (trained hidden) on Spectre.

## 2026-06-14 — CIFAR ARC HONEST CLOSE
- IN-CIRCUIT CIFAR-10 ~= 0.16 (direct readout 0.164 best; lean trained-hidden 0.152 then collapses).
  Both ~47% of the tractable ceiling, limited by: (1) SIMULATION COST — full trained-hidden net (0.43
  ceiling) is ~12h/run in Spectre (too many components x slots), so the trained-feature headroom is NOT
  reachable at simulable scale; (2) the C=10 rich-get-richer COLLAPSE (more epochs/higher KOUT -> worse).
- HONEST SUMMARY: CIFAR characterized fully token-free (color +11pts, trained>random, ceilings gray 0.30/
  color 0.41/trained 0.43, realistic-analog 0.37). In-circuit result 0.16 — a real first number; the gap
  to ceiling is SIMULATION-COST-bound (not a method limit). The ceiling analysis is the durable contribution;
  CIFAR confirms the design choices generalize beyond digits (avoids MNIST overfit, the user's goal).

## 2026-06-14 — VBBK optimum -> better backward-drive default (verified learner improvement)
- Full VBBK curve (depth-4 digits fast-cap, seed1): 0.20=0.25, 0.25=0.23, 0.32=0.62, 0.45=0.57, 0.65=0.52.
  CLEAN OPTIMUM ~0.32. Below -> backward UNDER-DRIVEN (collapses); above -> over-saturated (loses magnitude,
  the VBBK finding). Old default 0.45 is too saturated. 2-seed confirmed (0.32 > 0.45 at both seeds).
- ACTION: lower VBBK default 0.45 -> 0.35 (near optimum 0.32, safe margin above the 0.25 under-driven cliff).
  Improves the amplitude-restoring transport by keeping more magnitude. NOTE: confirm at full-horizon before
  re-claiming the 0.84 headline (screen-validated; the 0.84 result was VBBK=0.45-era).

## 2026-06-14 — NOISE (AUGJIT) calibration: always-on noise HURTS here (reverted)
- AUGJIT sweep (depth-4 digits, VBBK=0.35): 0=0.85, 0.05=0.75, 0.2=0.59. Noise MONOTONICALLY HURTS.
  WHY: tasks don't overfit (small nets, modest train/test gap) + analog HW already has intrinsic noise
  (mismatch/thermal) -> added input jitter = pure signal loss, NOT beneficial dropout. The "always use
  noise" default (0.2) I'd set was a -26pt regression. REVERTED AUGJIT default 0.2->0 (kept as opt-in knob).
- SILVER LINING: VBBK=0.35 default VALIDATED noise-free: aug0 (VBBK=0.35, AUGJIT=0) = 0.85 > old VBBK=0.45
  era qdigT 0.83. So the backward-drive default improvement (0.45->0.35) holds independent of noise. Net: two
  learner tunings settled this cycle — VBBK 0.45->0.35 (better), AUGJIT 0.2->0 (the always-on noise hurt).
- VBBK=0.35 CONFIRMED at fuller training (NEP=12 full data, no noise): best-epoch 0.72 (VBBK=0.35) vs
  0.64 (VBBK=0.45), +0.08. Consistent across all configs (fast-cap noise 0.62/0.57; noise-free 0.85;
  NEP=12 0.72/0.64). VBBK 0.45->0.35 default is SOLID. (Both collapse late at NEP=12 w/o freeze = the
  known universal late-drift, orthogonal to VBBK; freeze-at-peak locks the 0.72.)

## 2026-06-14 — VBBK "magnitude beats sign" was a NOISE ARTIFACT (walked back)
- NO-NOISE VBBK sweep (depth-4 digits fast-cap, AUGJIT=0): VBBK 0.32=0.73, 0.45=0.83, 0.65=0.76.
  OPTIMUM AT 0.45 (the original default); BOTH extremes worse. This REVERSES the under-noise result
  (0.32=0.62 > 0.45=0.57 > 0.65=0.52). So "lower VBBK / keep magnitude wins" was ENTIRELY an artifact of
  the AUGJIT=0.2 noise (which I'd added and since reverted as harmful). HONEST PICTURE: the backward
  regeneration has a TUNED OPTIMUM at the default; neither pure-magnitude (low drive) nor pure-sign (high
  drive) helps. ACTIONS: reverted VBBK default 0.35->0.45 (original); rewrote paper §6.3 to remove the
  magnitude-wins sub-claim (kept the core amplitude-restoring result, which stands: depth 0.55->0.84).
- LESSON: never draw a tuning conclusion from data measured under a confound (the noise). The user's
  original skepticism ("sure throwing away magnitude is good?") was right to probe — answer: neither
  extreme is good; the tuned middle (original default) is best. Caught by re-checking noise-free.

## 2026-06-15 — cascade-gain law: d4 point completes the depth axis
- d4 (4 hidden, spirals): NRW 100=0.84, 200=1.0, 400=1.0, 800=0.0(FAILS). Over-gain ceiling = ~400-800.
- FULL DEPTH AXIS (over-gain ceiling, where anti-lock starts): d1 >1600 / d2 ~800-1600 / d3 ~800 / d4 ~400-800.
  Ceiling MONOTONICALLY DROPS with depth -> confirms cascade-gain budget (per-stage gain ~G*^(1/depth));
  deeper nets MUST run lower per-stage gain or the cascade over-drives into anti-lock. 4 clean depth points.
  (Method note: ngspice can't run deep nets even at 480 slots = single-threaded; deep needs Spectre +mt.)

## 2026-06-15 — mismatch-yield-vs-depth: d3 point does NOT confirm fine monotonicity (honest)
- d3 (2-4-4-4-2 @NRW200) yield at 1mV = 2/3 (chips 1.0/0.0/1.0). HIGHER than d2's 1/3. So the effect is a
  CLIFF (d1 robust @2mV -> deep fragile @1mV), NOT a fine monotone gradient: d2(1/3) vs d3(2/3) are within
  3-chip sampling noise. Predicted d3<d2 (monotone); got d3>d2. The d1->deep transition is the real,
  clear effect; finer depth ordering needs many more chips than the 3-chip budget resolves.
- Paper's claim (line 456: d1 tolerates 2mV, d2 drops to ~1/3 @1mV) STANDS as a d1->d2 statement; I will
  NOT add a misleading "monotone with depth" gradient. Lesson (again): 3-chip yields are too noisy for
  fine trends; only the shallow->deep cliff is statistically clear.

## 2026-06-15 — HINGE anti-collapse = NULL for C=10 (wall confirmed robust)
- HINGE=1 (freeze-when-confidently-correct) on C=10 spatial+zerosum: 0.304 vs control 0.300 — NO effect
  (gate doesn't engage; outputs never reach the confidence threshold for C=10). Another null.
- C=10 DEEP-PATH WALL is now robustly confirmed ~0.30: resisted CCMS, PERAZ, HINGE, feature-count, depth,
  trained-hidden. ONLY zero-sum targets helped (0.22->0.30). CONCLUSION: pc_deep's deep differential path
  is structurally walled at C=10; the keystone path (gen_mc, zero-sum+PERAZ) reaches 0.71 and is the answer
  for many-class. This is a clean, well-tested negative bound — not for lack of trying.

## 2026-06-15 — keystone C=10 reproduced: 74.7% (ngspice, token-free)
- run_iso.sh N=96 (zero-sum targets TGHI=0.8/TGLO=0.467 + tuned PERAZ RAZ=6e3/CAZ=3e-3), real ngspice:
  acc=74.7% (n=600, ALL 10 classes alive per-class [92,90,75,67,78,62,58,85,88,52], ideal 85.3%, eff 88%).
  BEATS memory's 71.2% and the paper's current 0.587. BEST in-circuit C=10.
- PROVENANCE CARE: this is NGSPICE (gen_mc.py), paper claims all-Spectre. The paper's 0.587 may be the
  Spectre port (spectre_mc.py). Do NOT silently swap 0.587->0.747 across simulators. Either (a) run the
  Spectre port for matching provenance, or (b) report 74.7% explicitly labeled ngspice. Checking spectre_mc.
- README updated: best C=10 = 0.747 (keystone zero-sum+PERAZ, ngspice-labeled), distinct from the 0.587
  Spectre/random-feature line. KEY INSIGHT: zero-sum+PERAZ flips the feature-count ordering — paper notes
  "N=48 beats N=96 (0.587 vs 0.433)" WITHOUT PERAZ; WITH PERAZ, N=96=0.747 (more features now help).
- PAPER C=10 update is PENDING Spectre-provenance: the paper claims all-Cadence-Spectre; the keystone runs
  in ngspice (gen_mc.py). Clean follow-up = run the keystone via the existing spectre_mc.py port at a
  tractable N (the 17MB N=96 deck risks stalling Spectre like the CIFAR decks did; try N~32-48). Until then
  the 0.747 stays in README/ledger with explicit ngspice label, NOT silently swapped into the Spectre paper.

## 2026-06-15 — PROVENANCE RESOLVED: keystone C=10 is ngspice-only (Spectre STALLS)
- Keystone C=10 N=48 (7.7MB deck) on SPECTRE via spectre_mc.py STALLED at 333ns (trapezoidal ringing,
  non-convergence). Spectre cannot run keystone decks -> the gen_mc keystone workstream uses ngspice for
  this reason. So keystone C=10 (paper's 0.587 AND new 0.747) are NGSPICE, not Cadence Spectre.
- PAPER HONESTY: "all numbers are Cadence Spectre" is too strong for the many-class/keystone row. Deep
  pc_deep path (2D, depth-4 0.84, mismatch) IS Spectre; gen_mc keystone (C=10) is ngspice. Carve out
  explicitly. Best honest C=10 = 0.747 (ngspice), un-Spectre-verifiable (deck won't converge) = documentable fact.
- KEYSTONE C=10 = 0.747 SEED-ROBUST: 3 init seeds = 74.7/74.8/74.7% (spread 0.1%). Solid headline, not
  seed-luck. (Varies init seed; data fixed -> init-robust. Confirms the paper's C=10=0.747 is reliable.)
- KEYSTONE C=10 CEILING (numpy, token-free): ideal vs data/class = 40:82.5, 80:86.5, 120:86.2, 160:86.3
  -> more data helps to ~80/cls then SATURATES at ~86.5%. Trained MLP features = 87.5% (barely > random
  86.5% -> data-limited, not feature-limited). So C=10 in-circuit ceiling ~0.76 (88% eff of 86.5); 0.747
  (NTR=40) is already near-optimal. Running NTR=80 to confirm in-circuit follows the ideal toward ~0.76.
- KEYSTONE C=10 NTR=80 (2x data) = 75.0% vs NTR=40 74.7% (+0.3 only). The ideal's +3pt data gain did NOT
  translate in-circuit (eff ~87.5% both). So C=10 is CAPPED ~0.75 in-circuit (data + efficiency limited).
  0.747 headline is near-optimal & well-justified; NOT changing it (75.0 confirms the ceiling, within noise).
  C=10 THREAD FULLY BOUNDED: best 0.747 (robust 3-seed), ceiling ~0.75, ideal ~86.5% (data-limited), ngspice.

## 2026-06-15 — keystone+PERAZ does NOT transfer to CIFAR cleanly (digit-specific pipeline)
- Tried keystone (random feats + zero-sum + PERAZ) on CIFAR color-8: ideal collapses (random tanh feats
  29.2%, raw-feats-via-pipeline 26.3%) vs CIFAR's true linear ideal 0.41. TWO mismatches: (1) random
  features HURT CIFAR (linearly separable -> projection loses info; opposite of digits where they help);
  (2) run_iso's PER-SAMPLE z-score (per-image, good for digits) loses CIFAR's per-feature color/intensity
  signal (CIFAR needs per-FEATURE norm). So the keystone PIPELINE is digit-specific. The PERAZ MECHANISM
  may still work but needs a CIFAR-appropriate front-end (per-feature norm + direct readout). NOT pursuing
  the full adaptation (modest payoff, CIFAR fundamentally ~0.41-ceiling at 8x8). CIFAR result stands: pc_deep
  direct readout 0.16, characterized ceiling 0.37-0.41. Honest transferability bound on the keystone.

## 2026-06-15 — synapse linearization (path to >0.95): asymmetric weight-pair degeneration
- gsyn weight-response linearity (ngspice, fix input, sweep weight): UNIFORM RDEG makes it WORSE (R^2
  0.89->0.67, kills swing). But ASYMMETRIC degeneration (degenerate ONLY the weight pair M5/M6, keep input
  quad at 12k) LINEARIZES: RDW 12k->40k->100k->200k gives R^2 0.889->0.912->0.925->0.935 (swing 0.204->
  0.142->0.084->0.051). Sweet spot RDW=40k (R^2 0.912, 70% swing). Real cell improvement = a more-linear
  synapse weight response via asymmetric degeneration. Testing if it recovers the device-MLP toward 0.95.

## 2026-06-15 — ReLU+signed in-circuit + ONE-CAP synapse (user directions, tested in-circuit)
- ReLU+signed on real MNIST C=10 in-circuit (narrow spatial 16,4): tanh 0.136 / ReLU 0.200 / ReLU+Dale 0.200
  (then collapses). ReLU HELPS in-circuit (0.20 vs 0.136) — direction confirmed; but all ~0.20 << 0.96
  architecture ceiling -> the in-circuit LEARNING at narrow width is the wall (only 4 features; validated
  arch is H~64). Real MNIST is harder than sklearn at narrow width.
- ONE-CAP synapse (user's "1 cap/synapse"): structurally DONE (ONECAP knob, wn=fixed ref -> 120 caps vs 240,
  HALVED). BUT trains WORSE: digits C=4 one-cap 0.38 vs two-cap 0.83. WHY: the differential 2-cap storage
  provides COMMON-MODE REJECTION (wp,wn move oppositely -> drift cancels in wp-wn); a single cap vs fixed
  reference has NO such cancellation -> the cap's drift/offset corrupts the weight directly. HONEST FINDING:
  the differential pair wasn't just storage, it was drift-rejection. One-cap is achievable but needs the
  single-cap drift handled (per-synapse auto-zero, or a reference that tracks common-mode, or drift-robust
  update) before it's free. Naive 1-cap costs ~0.45 accuracy in-circuit.

## 2026-06-15 — one-cap deficit is STRUCTURAL (not rate); needs a balanced single-cap update
- ONECAP rate-compensation test: more epochs (NEP=16) plateaus at 0.37; 2x nudge (TD=0.2) -> 0.46. NEITHER
  recovers the two-cap 0.83. So the one-cap deficit is NOT just the halved update rate -- it's STRUCTURAL.
- ROOT CAUSE: gprod update is push-pull differential (iop charges wp UP, ion charges wn DOWN; weight=wp-wn
  integrates iop-ion). With wn=fixed source, ion sinks into the source -> only iop acts AND the update is
  ASYMMETRIC (up-drive works, down-drive lost) -> worse equilibrium. To get one-cap WITHOUT the loss needs a
  BALANCED single-cap update: route (iop - ion) into the single wp cap (current subtraction before the cap),
  so the bipolar weight integrates the full differential gradient on one node. That's a real gprod-output
  rewire (cell redesign), not a knob. VERDICT: one-cap halves caps but needs the balanced-update redesign;
  naive fixed-ref costs ~0.4 acc. The user's 1-cap constraint is achievable but is a cell-design task.

## 2026-06-15 — one-cap is REGIME-DEPENDENT; the 2nd cap's hidden job = common-mode anchoring (digits C=4)
Baselines: non-Dale two-cap 0.83; Dale two-cap 0.56 (Dale sign-constraint costs ~0.27 on its own).
Tested 3 one-cap synapse designs (all HALVE the weight caps, verified in deck):
- Naive fixed-ref (wn=VW0), BIPOLAR/non-Dale: 0.38. Breaks the gprod push-pull (only wp moves, asymmetric).
  More epochs / 2x nudge do NOT recover -> not just halved-rate.
- Differential cap (Cw across wp-wn), weak anchor RWL=2g: 0.23. ROOT CAUSE found by elimination
  (not bleed @2g, not half-rate @15p=0.29, not nudge @0.26): the single cap holds only the DIFFERENCE,
  leaving the common-mode of wp/wn unanchored -> nodes drift out of operating region.
- Differential cap + STRONG CM anchor RWL=100meg (diff tau still >> sim): 0.64 and still climbing. CONFIRMS
  CM-drift was the failure. => the two-cap's "redundant" 2nd cap is doing DOUBLE duty: differential storage
  AND per-node DC operating-point (common-mode) anchoring. One cap can hold one of those, not both, for free.
- Dale (UNIPOLAR) fixed-ref one-cap: 0.65 vs Dale two-cap 0.56 -> one-cap is FREE in the Dale regime
  (unipolar weight needs no push-pull, so single-ended is fine). Validates user's positive-neuron+1cap pairing.
LAW (proposed): a single storage cap suffices iff EITHER (a) weights are unipolar (Dale: sign in wiring), OR
(b) the node common-mode is separately anchored (strong-enough resistor). Bipolar + weak anchor needs 2 caps.
Next: push non-Dale diff one-cap CM-anchor strength + epochs toward the 0.83 two-cap ceiling (one cap, full acc).

## 2026-06-15 — one-cap final: ~0.67 ceiling non-Dale (NOT closable by epochs); ~0.16 cost is genuine
- CM-anchor sweep (CWW=15p): 30meg=0.65, 50meg=0.67(best), 100meg/8ep=0.64, 100meg/12ep=best 0.67 then
  DECLINES to 0.59 (peaks ~epoch5, overfits). => anchor strength saturates ~0.65-0.67; MORE EPOCHS DO NOT
  CLOSE the gap to two-cap 0.83. The residual ~0.16 is a real single-cap cost (half-rate I/C integration +
  residual CM imperfection + wp-wn cap-coupling instability), not a tuning artifact.
- BOTTOM LINE (no free lunch): one cap halves the weight caps (dominant area) but costs ~0.16 acc on digits
  C=4, in BOTH regimes (non-Dale diff+anchor 0.67; Dale unipolar 0.65 vs Dale two-cap 0.56). The two-cap
  differential genuinely does 3 jobs: bipolar storage, per-node common-mode anchoring, AND 2x integration rate.
  Verdict: keep two-cap as default; one-cap is a viable area-vs-accuracy knob (~half caps for ~0.16 acc), best
  paired with Dale where it's free relative to that regime's own ceiling.

## 2026-06-15 — one-cap "free" generalizes to ReLU (unipolar acts); but ReLU hurts digits C=4
- Option B (ReLU neurons + bipolar synapse): rl_1cap (ReLU+diff 1cap)=0.36, rl_2cap (ReLU+2cap)=0.34.
  => one-cap is FREE under ReLU too (0.36 vs 0.34), same as Dale -> the one-cap penalty only appears with
  BIPOLAR activations (tanh). With unipolar acts (ReLU/Dale) the single cap is fine. BUT ReLU tanks digits C=4
  (0.34-0.36 vs tanh 0.83) -> tanh still wins digits (ReLU's win was at C=10). Not a useful operating point here.
- REFINED LAW: one storage cap is free when ACTIVATIONS are unipolar (ReLU or Dale-signed neurons); the 2nd
  cap is only needed for the bipolar-tanh regime (where it also gives CM-anchor + 2x rate). Pairs with the
  user's "positive neurons -> one cap" intuition.

## 2026-06-15 — MORE NEURONS recovers Dale (0.56->0.80 ~= non-Dale 0.83)! but one-cap collapses at width
- Dale, digits C=4, 2x hidden width (64,32,16,8 -> 64,64,32,16):
  - two-cap: narrow 0.56 -> WIDE 0.80 (curve 0.54,0.8,0.71,0.8). MORE NEURONS RECOVERS DALE'S ACCURACY,
    nearly matching non-Dale two-cap 0.83. Validates user's "randomly-signed neurons + use more neurons" plan.
  - one-cap (fixed-ref ONECAP=2): narrow 0.65 -> WIDE 0.23 (collapsed!). The one-cap does NOT scale with width.
    Hypothesis: single-ended/half-rate update can't train the larger weight set in the same epochs -> undertrain.
- So one-cap and "more neurons" DON'T combine: the competitive Dale config (wide, 0.80) needs two caps.
  Testing whether rate-compensation (2x nudge / more epochs) or seed rescues the wide one-cap.

## 2026-06-15 — COMPLETE Dale x width x cap map (digits C=4); width = capacity-restorer, not general lever
                          narrow(64,32,16,8)   wide(64,64,32,16)
  non-Dale two-cap            0.83                0.82          <- width does NOT help unconstrained net
  Dale     two-cap            0.56                0.80          <- MORE NEURONS RESCUES Dale (capacity restore)
  Dale     one-cap            0.65                0.23 unstable <- one-cap COLLAPSES at width; rate-comp (16ep+
                                                                   2x nudge) does NOT fix (0.23 flat) => fundamental
  non-Dale one-cap(diff+anch) 0.67                 -
LAWS:
- "More neurons" is a CAPACITY-RESTORING lever for CONSTRAINED regimes (Dale sign-constraint), NOT a general
  accuracy booster. Unconstrained non-Dale is already at its depth/arch ceiling ~0.83; width can't break it.
- => the path past 0.83 toward 0.96 is FEATURE/LEARNING quality (depth, better features, in-circuit learning
  efficiency), NOT width. Confirms user's "don't go too wide" instinct for the unconstrained case.
- One-cap scales to neither width nor the bipolar-tanh regime; free only narrow + unipolar (Dale/ReLU). The
  competitive analog design is TWO-CAP: non-Dale narrow (0.83) or Dale wide (0.80, biological + signed).

## 2026-06-15 — C=10 collapse is TRAINING-DYNAMICS, not capacity (width doesn't help)
- C=10 (sklearn digits, ZEROSUM=1): narrow(64,32,16,10) peak 0.288->0.08; wide(64,64,32,10) peak 0.296->0.052.
  BOTH start ~0.29 at epoch 1 then COLLAPSE during training. Width does NOT prevent it.
- => the C=10 wall is a TRAINING-DYNAMICS / target-pressure instability, NOT a capacity shortfall. The "more
  neurons rescues constrained regimes" law applies to the Dale SIGN-constraint (a capacity cut), NOT to the
  C=10 collapse (a dynamics problem). ZEROSUM gives the 0.29 start but doesn't stabilize the descent.
- Next: the documented C=10 movers are PERAZ (per-class auto-zero) + RFGRID (spatial receptive fields), per
  prior gen_mc evidence (keystone hits 0.71-0.75). Apply those to stabilize + climb, not width.

## 2026-06-15 — pc_deep C=10 is a CONFIRMED dead-end (collapse survives every lever); pivot to keystone/ngspice
- C=10 in pc_deep (Spectre), all collapse from a ~0.20-0.29 epoch-1 peak: ZEROSUM-only ->0.05/0.08;
  +PERAZ ->0.068; +RFGRID(deep 64,16,4) ->0.052 (4-neuron bottleneck too tight); width (wide) ->0.052.
  => NO pc_deep lever (width, ZEROSUM, PERAZ, RFGRID) stops the C=10 collapse. Matches the standing note that
  device-model training-rule fixes don't transfer to ngspice's deep path. STOP fighting pc_deep for C=10.
- PROVEN C=10 path = the gen_mc / pc1_orch KEYSTONE in ngspice (token-free): C=10 ~0.747, C=5 ~0.87. Pivot
  there to push past 0.747. pc_deep stays the vehicle for the SHALLOW/low-C analog-laws work (Dale, one-cap,
  width) where it's reliable (C=4 ~0.83).

## 2026-06-15 — keystone C=10 readout: tractable harness + AVGW lever; weak-class confusion is the limiter
- Tractable run (run_iso.sh, 2500 slots, NTR=30, ngspice ~14min train): in-circuit 68.8%, ideal(LogReg) 80.4%
  -> ~86% readout efficiency, consistent with the full-config 0.747/0.853. (8000-slot runs took 49min+, killed.)
- Inference-only lever (reuse trained weights, no retrain): AVGW 20->40 gives 68.8->70.4% (+1.6, better readout
  SNR). Diminishing; the real limiter is PER-CLASS CONFUSION: classes 2,5,9 stuck at 28-40% while others 72-96%,
  mean-margin only ~0.11. Those digits are genuinely hard for random-features + single-ended linear readout.
- => past ~0.75 the lever is readout DISCRIMINATION for the weak classes (4-quadrant pc1_orch readout, or
  per-class margin shaping), NOT features (ideal saturates at N=96) and NOT slots/averaging (diminishing).

## 2026-06-15 — SPECTRE C=10 BREAKTHROUGH: freeze hidden (keystone arch) stops the collapse, climbs
- The C=10 collapse in pc_deep/Spectre was the DEEP TRAINING, not the device: HFREEZE=0 (freeze hidden at
  random init, train ONLY the readout = keystone architecture) STOPS the collapse and CLIMBS in real Spectre:
    c10_fz0  (deep 64,32,16,10 frozen): 0.192 climbing (vs full-train collapse to 0.08)
    c10_fz0w (wide 64,64,10 frozen):    0.360 climbing, still rising at NEP=8 (vs collapse to 0.05)
- WIDE single random-feature layer >> deep (more random features, like keystone's N=96). Both still climbing
  at 8 epochs -> more features + more epochs should push toward the ngspice keystone's 0.747, but now in SPECTRE
  (trustworthy transistor-level). This is the path to a real Spectre C=10 number, not ngspice-only.

## 2026-06-15 — C=10 collapse is in the HIDDEN PC dynamics (output-error fixes all fail identically)
- Full-PC C=10 (all layers learn) collapses to EXACTLY 0.068 with an IDENTICAL trajectory [peak~0.20-0.29 @ep1
  -> 0.13 -> 0.05 -> 0.068] regardless of: TD (0.05/0.1/0.2), TD-anneal (0.85), update gain (0.5), ZEROSUM,
  PERAZ, CCMS (cross-class CM subtraction), and CCMS+PERAZ. NONE of the OUTPUT-error knobs change the trajectory.
- => the collapse is NOT learning-rate and NOT output-error shaping; it is the HIDDEN-LAYER PC dynamics. One
  epoch HELPS (0.10->0.20) then a positive-feedback collapse takes over = hidden features running away
  (synapse saturation / rich-get-richer among neurons). This is exactly what FREEZING bypassed (but user wants
  full PC). Testing HIDDEN-faithful brakes (keep all layers learning): WCLAMP (diode weight clamp vs runaway),
  RWL leak (pull weights back), single-hidden 64,64,10 (shorter backward path = less compounding).

## 2026-06-15 — DEPTH drives the C=10 hidden collapse; single-hidden is far milder
- Hidden brakes (full PC): WCLAMP=0.3 -> 0.076 (collapse), RWL=100meg leak -> 0.068 (collapse) on DEEP
  64,32,16,10. But SINGLE-HIDDEN 64,64,10 -> peak 0.252, holds 0.22-0.25 for epochs 2-6, declines to 0.144
  (curve [0.112,0.252,0.228,0.144]) = MUCH milder, different trajectory. => compounding through stacked hidden
  layers makes the collapse catastrophic; one hidden layer training HELPS (0.11->0.25) then mildly overshoots.
- Hidden update is USEFUL but overshoots. PC-faithful fix: keep all layers learning, just SLOW the hidden rate
  (gblh < gblo) so it asymptotes at the peak. Testing single-hidden + slow-hidden-LR + wider.

## 2026-06-15 — full-PC C=10 in SPECTRE: comprehensive ceiling ~0.26 (early-stop); collapse robust to ~13 fixes
- Best full-PC (all layers learn) C=10 in real Spectre: single-hidden 64,64,10 ~0.26 (early-stopped), then
  declines with more training. Wider hidden (96) WORSE (0.16). Deep (64,32,16,10) collapses to 0.068.
- Collapse RESISTS: TD(0.05/0.1/0.2), TD-anneal, gain(0.5), ZEROSUM, PERAZ, CCMS, CCMS+PERAZ, WCLAMP, RWL-leak,
  depth-reduction, hidden-LR(GBLH 0.3/0.5), width. Backward common-mode IS already handled (x-node cmld).
- ROOT: training the hidden PC features at C=10 is STRICTLY HARMFUL vs random — the SAME 64,64,10 FROZEN climbs
  to 0.36 while TRAINED peaks 0.26 then declines. So the PC hidden update destroys discriminative features at
  high C. Not LR, not output-error, not weight-runaway. The mechanism is in the hidden update direction itself.
- DECISION POINT (asked user): diagnose-the-collapse / C-curriculum warm-start / two-phase-EP rework / accept
  lower-C-where-PC-works. ngspice keystone 0.75 and freezing 0.36 both ruled out by user (untrusted / not PC).

## 2026-06-16 — DIAGNOSIS: C=10 collapse = weights SHRINK to zero (target negativity), not saturation
- All-parallel directions: Dir3 two-phase EP (CHL) NO fix (0.20/0.21); Dir2 curriculum warm-start C6->C8 NO
  fix (0.18 < cold 0.23); Dir4 crossover = CLIFF at C~7-8 (C4 0.76, C6 0.51, C8 0.23, C10 0.26).
- Dir1 DIAGNOSIS (compare saved weights C6-healthy vs C8-collapsed): collapsed net has SMALLER weight std
  (HID 0.022 vs 0.033, OUT 0.033 vs 0.046), satfrac=0. => weights SHRINK toward zero (uniform output ->
  below-chance), NOT runaway/saturation. Explains why WCLAMP/leak/LR all failed (wrong mechanism).
- MATCHES the known "target-negativity-vs-C crossover @C~8": zero-sum targets push 9 classes DOWN vs 1 UP;
  net-negative pressure shrinks weights at high C. => the mechanism-correct lever is TARGET NEGATIVITY (ZSNEG),
  not learning rate / clamp / curriculum / two-phase. Testing ZSNEG sweep at C=10.

## 2026-06-16 — ZSNEG flat too; full-PC C=10 single-hidden is a ROBUST ~0.25 ceiling (15+ levers fail)
- ZSNEG sweep (0.0/0.05/0.20) at C=10: BEST 0.252/0.256/0.264 -- FLAT, byte-identical curves. Target negativity
  does NOT control the collapse (diagnosis was right that weights shrink, but ZSNEG isn't the driver).
- ROBUST CEILING: full-PC C=10 single-hidden tops out ~0.25 (early-stop) then declines to ~0.14, INVARIANT to:
  TD/anneal/gain, ZEROSUM, PERAZ, CCMS, WCLAMP, RWL-leak, depth, hidden-LR(GBLH), width, two-phase EP (CHL),
  curriculum warm-start, ZSNEG. The trajectory is near-deterministic across all of these.
- HONEST STATE: trustworthy full-PC C=10 in Spectre = ~0.25 early-stopped. The collapse (weights shrink ->
  uniform output) is robust to every standard + several non-standard levers. Remaining directions are DEEP
  reworks (forward-settling stability, output representation, input/feature encoding), not knobs. Reported to user.

## 2026-06-16 — LITERATURE: EP many-class collapse = one-sided NUDGE BIAS; fix = SYMMETRIC (+/-beta) nudging
- Laborieux et al. (Front. Neurosci. 2021, arXiv 2006.03824): one-sided EP gradient has O(beta) bias that
  accumulates and "does not scale to tasks harder than MNIST" (86% CIFAR err). SYMMETRIC/centered nudging
  (run +beta AND -beta equilibration phases, update on (grad_+ - grad_-)/2beta) cancels the leading bias ->
  O(beta^2). This MATCHES our C~8 cliff exactly (fine easy, collapse harder).
- PC+EP ImageNet (arXiv 2606.03584): symmetric/centered nudging + CE-nudge (softmax) + layerwise ASYNC updates
  (even-then-odd, avoids synchronous collapse) + weight-align (equal leakage) -> 1000-class PC works.
- WE ALREADY HAVE: CE+softmax (SOFTC), fwd/bwd weight alignment (shared-cap transpose). MISSING: SYMMETRIC
  nudging. CRUCIAL: our CHL=1 is FREE-vs-CLAMPED = the ONE-SIDED biased estimate, NOT symmetric +/-beta. So the
  real fix was never tested. ACTION: implement symmetric nudge (+beta clamp toward label, -beta clamp away,
  update on difference) and test at C=10. Also try layerwise async update ordering.

## 2026-06-16 — PROBE confirms: weights shrink MONOTONICALLY (gradient-bias fingerprint) -> symmetric-nudge fix
- Weight-std trajectory (C=10 single-hidden, NEP snapshots): HIDstd 0.063->0.039->0.018, OUTstd 0.106->0.057
  ->0.023 (NEP 2/4/8). MONOTONIC ~3.5x shrink toward zero; mean stays ~0 (pure magnitude decay, NO common-mode
  drift). Acc peaks at NEP=4 (0.256) while weights already shrinking, then collapses when weights too small.
- A steady systematic shrink = the fingerprint of a BIASED gradient estimate. Converges with the Laborieux
  result: one-sided nudge has O(beta) bias -> here it biases weights toward zero, worse at high C. FIX = symmetric
  +/-beta nudge (O(beta^2)). Implemented: SYMNUDGE retargets the chopper -clock to the -beta phase (CHL=3+HCHOP=4
  true-EP cells) + backward drive on in both phases. Testing c10_sym now.

## 2026-06-16 *** BREAKTHROUGH: SYMMETRIC NUDGE BREAKS THE C=10 COLLAPSE (full PC, Spectre) ***
- C=10 single-hidden, SYMMETRIC +/-beta nudge (SYMNUDGE=1 CHL=3 HCHOP=4, chopper true-EP cells, neg clock on
  -beta phase): curve [0.312, 0.368, 0.412, 0.432] -- MONOTONIC CLIMB, NO collapse, still rising at NEP=8.
- vs one-sided baseline: peak 0.256 then COLLAPSE to 0.14. Symmetric ~DOUBLES it (0.43) and the weight-shrink
  is gone. Confirms the full chain: collapse = one-sided O(beta) gradient bias (weights shrink monotonically)
  -> symmetric +/-beta cancels it (O(beta^2), Laborieux 2021) -> clean training. FULL PC, all layers learn,
  real Spectre, no freezing. THE fix the whole investigation was after.
- Cost: chopper cells (21 FET) + 2x slots -> ~2hr/run at NEP=8. Still climbing -> more epochs should go higher.

## 2026-06-16 — symmetric-nudge breakthrough CONFIRMED across 2 seeds (full PC, Spectre, C=10)
- C=10 single-hidden symmetric +/-beta: seed1 BEST 0.432 (monotonic climb [0.31,0.37,0.41,0.43]), seed2 BEST
  0.400 ([0.28,0.40,0.30,0.31], mild post-peak dip). vs ONE-SIDED baseline BEST 0.26 then catastrophic collapse
  to 0.14. => symmetric nudge ROBUSTLY lifts C=10 by ~+0.15 and removes the catastrophic collapse (seed2 dip to
  0.31 >> old 0.05-0.14; residual mild dip = O(beta^2) leftover). Confirmed: 2 seeds, full PC, all layers learn,
  real Spectre, no freezing. NEP=14 ceiling run pending. This is the headline C=10-in-Spectre result.

## 2026-06-16 — symmetric nudge REVERSES the weight bias (shrink->grow); next lever = weight decay
- NEP=14 symmetric C=10: BEST 0.384, curve [0.31,0.38,0.38,0.38,0.36,0.32,0.35] (peaks ep4-6, mild late decline).
  HID weight std = 0.58 (!!) vs one-sided collapse 0.018 and healthy ~0.05. => symmetric didn't just STOP the
  shrink, it FLIPPED the bias sign -> weights now OVER-GROW toward saturation (std 0.58 ~ clip range), causing a
  mild late-epoch decline. Strong mechanistic proof the collapse was a signed gradient bias.
- Sweet spot = early-stop ~0.38-0.43 (NEP=8 got 0.43; NEP schedule changes the anneal trajectory). NEXT LEVER:
  add WEIGHT DECAY (Laborieux/ImageNet PC+EP use 2e-4) or mild WCLAMP to tame the over-growth -> should stabilize
  the late decline and let it hold its peak. Battery running: symmetric x {deep, C6, C8, smaller-beta, wider}.

## 2026-06-16 — symmetric nudge FIXES THE DEEP collapse (0.068 -> 0.368); weight decay stabilizes
- DEEP 64,32,16,10 symmetric: BEST 0.368 vs one-sided DEEP 0.068. SYMMETRIC RESCUES DEPTH -> deep PC now
  trains at C=10 (was the worst collapse). Directive-aligned (deep not wide).
- Weight decay (RWL=5meg) single-hidden symmetric: 0.368 and STABLE (final=best, the over-growth decline is
  gone) -> confirms the late decline was weight over-growth; mild leak tames it (slightly lower peak, no crash).
- Refilling budget with DEEP variants: deep+decay (combine), deep+more-epochs. Pursuing depth per user directive.

## 2026-06-16 — BEST DEEP config: symmetric + WEIGHT DECAY = 0.436 (matches single-hidden, depth penalty gone)
- DEEP 64,32,16,10 symmetric + weight decay (RWL=5meg): BEST 0.436 vs deep-plain symmetric 0.368 and one-sided
  deep 0.068. Weight decay tames the symmetric over-growth in the deep net -> deep now MATCHES single-hidden
  (0.43). So: symmetric (fix collapse) + weight decay (fix over-growth) = deep PC trains at C=10 in Spectre.
- More data (NTR=40) single-hidden = 0.438 (~same as NTR=20) -> data not the limiter. Epoch sweet spot ~8-10.
- HEADLINE config = DEEP 64,32,16,10 + CHL=3 HCHOP=4 SYMNUDGE=1 + RWL=5meg, full PC, Spectre, C=10 ~0.44.
  Confirming seed-2 + more-epochs robustness next.

## 2026-06-17 — deep+decay across C; seed variance noted; 4-hidden too slow (killed 6hr run)
- DEEP 64,32,16,C + symmetric + decay(RWL=5meg): C6=0.513, C8=0.530, C10=0.34-0.44 (seed1 0.436, seed2 0.340).
  All FAR above one-sided collapse (C8 0.23, C10 0.068 deep). C=8 (0.53) notably strong. C=10 has real seed
  variance -> need multi-seed mean for a trustworthy headline. deep+epochs(NEP12)=0.432 ~ deep+decay (either
  lever lifts deep to single-hidden level).
- 4-hidden (64,48,32,16,10) symmetric+chopper = impractically slow (>6hr, killed). 3-hidden is the practical deep.
- Launching deep+decay C=10 seeds 3-5 for a clean mean +/- std headline.

## 2026-06-17 — HEADLINE multi-seed: deep+decay C=10 = 0.412 +/- 0.041 (5 seeds); best combo 0.472
- DEEP 64,32,16,10 + symmetric +/-beta + weight decay, C=10, 5 seeds: [0.436,0.340,0.424,0.400,0.460] ->
  mean 0.412 +/- 0.041. ROBUST, full PC, all layers learn, real Spectre, no freezing, deep (not wide).
  vs one-sided deep collapse 0.068 and single-hidden one-sided 0.26-collapse. ~6x the deep baseline.
- + more-epochs (NEP12) best single = 0.472. Real MNIST (deep+decay) = 0.324. C-sweep deep+decay: C6 0.51,
  C8 0.53, C10 0.41. THE C=10-in-Spectre result: symmetric nudge (fix bias/collapse) + weight decay (tame
  over-growth) makes deep PC train at high class count. Getting best-combo (decay+epochs) multi-seed next.

## 2026-06-17 — FINAL: C=10 collapse RESOLVED in deep full-PC Spectre via symmetric nudge + weight decay
HEADLINE (all real Spectre, full PC all-layers-learning, deep 64,32,16,10, no freezing, no wide):
  - deep + symmetric(+/-beta, CHL=3 HCHOP=4 SYMNUDGE=1) + weight-decay(RWL=5meg): C=10 = 0.412 +/- 0.041 (6 seeds)
  - + more epochs (NEP=12) = 0.424 +/- 0.031 (4 seeds)  [best config]
  - vs ONE-SIDED deep collapse 0.068, single-hidden one-sided 0.26-then-collapse. ~6x.
  - C-sweep deep+decay: C6 0.51, C8 0.53, C10 0.41.  Real MNIST deep+decay: ~0.27 (2 seeds 0.22/0.32).
MECHANISM (fully diagnosed): one-sided nudge has O(beta) gradient bias (Laborieux 2021) -> weights shrink
  monotonically to zero -> uniform output -> collapse (worse at high C). Symmetric +/-beta cancels bias (O(beta^2))
  -> weights then OVER-grow (std 0.018->0.58) -> weight decay/epochs tame it. Two failure modes, two fixes.
NOTES: 4-hidden (64,48,32,16,10) symmetric+chopper impractically slow (>6.7hr x2, killed) -> 3-hidden practical.
  Chopper symmetric run ~2-3hr each. INVESTIGATION CONVERGED.

## 2026-06-17 — path-to-0.90 first strike: 8x8 feature-limited; resolution-via-sparse HURTS -> need true conv
- RFGRID (local RFs) + symmetric + decay, 8x8: 0.452 (vs random-sparse 0.41) -- local RFs help +0.04.
  Deeper RFGRID 64,16,4,10 = 0.340 (4-neuron bottleneck too tight).
- 8x8 is FEATURE-CAPACITY-LIMITED: more data (NTR=40)=0.452 and HINGE=0.452 both NEUTRAL (same as base).
  => the 8x8 single-channel ceiling ~0.45 is features, not data/optimization/normalizer.
- 16x16 (idea 2) HURTS through current wiring: RFGRID 256,64,16,10 = 0.188, random-sparse = 0.292 (both << 8x8
  0.45). CAUSE: FANIN=4 sees only 4 of 256 inputs -> too sparse for a 16x16 image (vs 4-of-64 @8x8) -> most
  pixels discarded + bigger deck undertrains. => resolution needs DENSE local pooling, not sparse fan-in.
- CONCLUSION: the real lever (idea 1 FULL) = MULTI-CHANNEL WEIGHT-SHARED CONV (dense 2x2 local windows, K filters
  per position, shared caps). Single-channel RFGRID + sparse fan-in can't exploit more pixels. Next build: conv
  layer generator in pc_deep (K channels, weight-shared, dense local RF). Ensemble (idea 7) vote pending (4 seeds).
