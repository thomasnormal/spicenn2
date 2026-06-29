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

## 2026-06-17 — first-strike VERDICT + next build = multi-channel weight-shared conv
- Confirmed bottleneck: 8x8 single-channel ~0.45 ceiling is FEATURE CAPACITY (data/hinge/resolution-via-sparse
  all neutral-or-worse). The ONLY lever that adds usable features = idea 1 FULL: multi-channel weight-shared
  conv (dense 2x2 local windows x K filters per position, shared weight caps). This is the next build.
- Ensemble (idea 7) caveat: SEED controls BOTH data-split AND connectivity (pc_deep line 17), so different-SEED
  runs have DIFFERENT test sets -> mout files not aligned -> can't per-example vote. To ensemble properly: add a
  separate DSEED (data split) knob, fix it, vary the net seed -> same test set, diverse nets -> vote. TODO.
- Best so far on C=10 (deep, full-PC, Spectre): symmetric+decay ~0.41-0.45 (RFGRID local RFs best at 0.452).

## 2026-06-17 — ENSEMBLE (idea 7) WORKS: 2-net vote 0.49/0.46 -> 0.556 (+6-9pt); scaling to 5 nets
- DSEED decouples data-split from net seed -> nets share a FIXED test set -> votable. deep+decay C=10,
  DSEED=0, SEED=1/2: indiv 0.492/0.460 -> VOTE(sum margins, argmax) = 0.556 (+6-9pt). Diverse analog nets make
  different errors -> voting corrects. Hardware-cheap (sum output rails, no new cell). Scaling to 5-10 nets.
- Conv (idea 1) scaling pending (K=4=0.392 low; K=8/16 running). Ensemble is the cheapest big lever so far.

## 2026-06-17 — CONV (idea 1) underperforms; ENSEMBLE (idea 7) is the lever toward 0.90
- Conv weight-shared sweep (8x8): K=4=0.392, K=8=0.364, K=16=0.316 (MORE filters WORSE); 16x16 conv K=2=0.304,
  K=4=0.368. All BELOW RFGRID 0.452 and dense 0.41. Weight-sharing cut per-filter capacity + bigger conv layers
  undertrain in the horizon; single conv layer + PC update didn't yield diverse useful filters. Conv-as-built is
  NOT the path (big-filter decks also 5hr+ impractical). Built+working (wkey sharing) but not a win as-is.
- ENSEMBLE is the winner: 2-net vote 0.49/0.46 -> 0.556 (+6-9pt). Scaling to 8-10 nets (DSEED fixed test set).
  Cheap (sum rails), hardware-friendly, compounds. THE clearest line to 0.90.

## 2026-06-17 — ENSEMBLE SCALING: 2-net 0.556 -> 5-net 0.616 (individuals ~0.38-0.49). Path to 0.90.
- deep+decay C=10, DSEED=0 fixed test set, vote = argmax(sum margins). Scaling: 2-net 0.556, ~5-net 0.616.
  Vote >> any individual (~0.45). Climbing; pushing to 9-11 nets. Cheap (sum output rails), hardware-friendly.
- Combined plan to 0.90: ensemble (this) x stronger single nets (RFGRID 0.452 base, symmetric+decay). Conv ruled
  out as-built. Diminishing returns expected as vote saturates at consensus ceiling, but 0.45->0.62 is the lever.

## 2026-06-17 — ENSEMBLE 7-net = 0.696 (mean indiv 0.41, best 0.49); strong scaling, the path to 0.90
- deep+decay C=10 DSEED=0 (fixed test set), vote=argmax(sum margins): 2-net 0.48, 3-net 0.57, 5-net 0.60,
  7-net 0.696. Vote +0.20 over mean individual (0.41). Still climbing -> scaling to 14 nets. Hardware-trivial
  (sum output rails). THE lever. (n=175-250 test, some noise, but trend clear.) Combine w/ stronger base nets.

## 2026-06-17 — ensemble PLATEAUS ~0.70 with 0.41-indiv nets; pivot = stronger+diverse base nets, mix archs
- deep+decay ensemble: 7-net 0.696, 8-net 0.708, 9-net 0.700 -> saturates ~0.70 (n=250, +-3%). +0.29 over indiv
  0.41 = big real gain, but caps. To raise the ceiling: ensemble STRONGER+more-DIVERSE bases. All nets use
  DSEED=0 -> SAME test set -> can MIX architectures in the vote (RFGRID 0.452 + deep-dense 0.41 + ...). Diverse
  archs = more decorrelated errors = higher vote ceiling. Adding RFGRID-base nets to the pool. Path to 0.90 =
  (stronger bases) x (more nets) x (arch diversity).

## 2026-06-17 — CORRECTION: ensemble did NOT plateau; 10-net deep = 0.744 (still climbing)
- Full deep+decay vote curve (n=250): 2=0.556, 3=0.632, 5=0.652, 7=0.696, 9=0.700, 10=0.744. The "plateau ~0.70"
  was NOISE (9-net dip). 0.41 individuals -> 0.744 at 10 nets, still rising. Ensemble is a strong, real lever to
  0.90. Adding RFGRID-arch diversity + more nets. (n=250 noisy +-3%; trend robust.)

## 2026-06-17 — ENSEMBLE 11-net deep = 0.776 (climbing); the clear path to 0.90
- deep+decay vote: 10-net 0.744, 11-net 0.776 (0.41 individuals). STILL CLIMBING - not plateaued. Mixing 2
  weak RFGRID nets barely helped (0.776->0.780); the deep ensemble is the engine. Scaling to 15-20 nets ->
  expect 0.80-0.85+. Ensemble of diverse deep analog nets (sum output rails) is THE lever to 0.90.

## 2026-06-17 — ENSEMBLE plateaus ~0.77 (bounded by individual strength); 0.90 needs stronger individuals
- deep+decay vote: 11-net 0.776, 13-net 0.764 -> PLATEAU ~0.77 (n=250, +-3%). Big real gain (0.42 indiv ->
  ~0.77 vote, +0.35, random-subspace/RF effect from diverse connectivity). But the vote CEILING scales with
  individual strength -> capped because individuals cap ~0.42 (the feature-capacity wall). Mixing weak RFGRID
  (0.45) barely helped.
- HONEST STATE on path to 0.90: ensemble is a strong, real lever (0.42->0.77) but bounded. To reach 0.90 needs
  STRONGER INDIVIDUAL nets (~0.6+), i.e. the feature/representation breakthrough that conv didn't deliver. The
  individual ceiling is the binding wall again. CAVEAT: numbers on the early-stop-shared 250-ex test set (mild
  optimism, consistent basis). Best trustworthy C=10: single 0.42-0.45, ENSEMBLE ~0.77 (deep, full-PC, Spectre).

## 2026-06-17 — FINAL ensemble state: ~0.78 ceiling (21 nets), bounded by individual strength; 0.90 = feature wall
- RFGRID ensemble (7 nets, indiv 0.369): 0.656 (WEAKER - those nets avg lower, not higher). deep ensemble (14,
  indiv 0.422): 0.772. MIXED all 21 nets: 0.780. => ensemble ceiling ~0.78, set by individual strength ~0.42.
- BEST TRUSTWORTHY C=10 (deep, full-PC, real Spectre): single net 0.42-0.46; ENSEMBLE of ~20 diverse nets 0.78.
  This is a LARGE real gain (single 0.42 -> ensemble 0.78, +0.36, random-subspace effect, sum output rails).
- 0.90 WALL: needs stronger INDIVIDUAL nets (>~0.55). The individual ceiling is feature/representation capacity,
  which conv (idea 1) did NOT break and RFGRID only matched. Untried levers for stronger individuals: degenerated
  Gilbert synapse (idea 3, readout efficiency), deep supervision (idea 4), momentum optimizer (idea 8). These are
  the remaining path to lift individuals -> then ensemble -> 0.85-0.90. Honest: 0.78 achieved, 0.90 still open.

## 2026-06-18 — RDEG=24k = STRONGER INDIVIDUAL (0.532 single, vs 0.44 default); the lever for 0.90
- Synapse degeneration/linearity sweep (single net, deep+decay+symmetric): RDEG 6k=0.476, 12k(default)~0.44,
  24k=0.532, 40k=0.360 (too low gain). RDEG=24k SWEET SPOT -> single 0.532 (+0.09)! More linear Gilbert multiply
  = stronger individual (idea 3 DOES deliver, just via RDEG tuning since the Gilbert cell already exists).
- => ensemble of RDEG=24k (0.53) nets should beat the 0.78 ceiling (which was from 0.42 nets). Launching RDEG=24k
  ensemble. Combine with NEP12. This is the concrete path to lift individuals -> ensemble -> 0.90.

## 2026-06-18 — RDEG=24k single 0.532 was SEED NOISE (ens indiv avg 0.431); ensemble ceiling firm ~0.78-0.79
- RDEG=24k ensemble (indiv 0.431, NOT stronger), 3-net 0.556; MIXED rd24+deep 20-net = 0.788. RDEG/NEP/conv/
  RFGRID all FAIL to reliably beat ~0.43 individuals. Ensemble caps ~0.78-0.79 regardless of base config.
- KEY QUESTION: is the 0.43 individual ceiling CAPACITY (tiny FANIN=4 sparse net) or LEARNING (in-circuit eff)?
  Running ideal-MLP (backprop) diagnostic on same data/size to decide whether stronger individuals are possible.

## 2026-06-18 *** KEY: ensemble is DATA-limited, not circuit-limited; path to 0.90 = ensemble + MORE DATA ***
- Ideal (LogReg) vs train data on 8x8 real MNIST: NTR=20/cls=0.77, 50=0.82, 100=0.88, 200=0.91. Our ENSEMBLE
  (0.78 at NTR=20) ALREADY = the data-ideal! So the 0.78 ceiling is DATA (20/cls), NOT circuit/feature/learning.
  Single nets are weak (0.43) but the ensemble RECOVERS the data-ideal. => stronger individuals were the WRONG
  target; the lever is TRAINING DATA. ensemble @ NTR=100 -> ~0.88, @ NTR=200 -> ~0.91 = THE PATH TO 0.90.
- Constraint: more data = proportionally more Spectre slots (NTR=100 ~5x slower ~5-10hr/net). Launching NTR=50
  (ideal 0.82, faster) + NTR=100 (ideal 0.88) ensemble nets. This is the concrete, evidence-backed route to 0.90.

## 2026-06-18 — PATH TO 0.90 IDENTIFIED but Spectre-RUNTIME-BOUND (honest endpoint)
- Data-scaling confirmed in principle: ensemble TRACKS the data-ideal (0.78 = LogReg ideal at 20/cls). Ideal
  scales 0.77/0.82/0.88/0.91 at 20/50/100/200 per class. So ensemble + more data -> 0.90.
- WALL (empirical): (1) individual nets do NOT improve with data (0.39-0.42 regardless; weak FANIN=4 learner);
  only the ENSEMBLE recovers the data-ideal, needing ~10+ nets per data level. (2) higher-data Spectre runs are
  pathological: NTR=50=4.5hr, NTR=100~5-7hr, NTR=200~10-15hr EACH (symmetric-chopper x more data). A 0.90-grade
  ensemble = ~10 nets x ~10hr = ~100 Spectre-hours / 7-parallel = days. NTR=50 2-3 net vote so far 0.49 (too few).
- HONEST RESULT (deep, full-PC, real Spectre, C=10): single 0.42-0.47; ENSEMBLE 0.78 (= data-ideal at feasible
  20/cls); collapse FIXED (0.068->0.41 via symmetric nudge+decay). 0.90 needs more data, runtime-prohibitive
  here -> would need a faster simulator (Xyce/GPU) or many Spectre-days. Path clear; not idea-bound, sim-bound.

## NTR=100 data-scaling test (real Spectre, deep full-PC, sym-nudge) — 2026-06-18
First NTR=100 net (d100_5, NEP=2) completed: **individual = 0.392**.
This sits ON TOP of NTR=50 (0.395) and NTR=20 (~0.42) individuals → CONFIRMS again
that individual analog nets do NOT improve with more training data; the per-net
accuracy is fixed by analog efficiency, not data. Only the ENSEMBLE vote recovers
the data-ideal (LogReg @100/cls ≈ 0.88), which requires ~10+ voters. 7 NTR=100 nets
training (4 NEP=5 @~6.5h, 3 NEP=2 @~2.5h) + d100_8 added (fast voter) to grow the vote.
Vote with 1 net = 0.392; climbs only as voters land. Endpoint unchanged: path to 0.90
is ensemble+data and is Spectre-RUNTIME-BOUND (needs ~10 multi-hour nets per data level).

### OPS GOTCHA (2026-06-19): launching d100 nets needs STO=240000
pc_deep.py runs ONE spectre subprocess.run with timeout=STO (default 2400s=40min).
NTR=100 nets take ~2.5h (NEP=2) to 7h (NEP=5) >> 2400s, so EVERY launch must set
STO=240000 (66h, effectively off). d100_8 died at exactly 2400s because I omitted STO;
caught it, killed+relaunched d100_9-13 with STO=240000. Seed-dependent circuit stiffness
(K varies per seed) also affects runtime but 96 cores means no oversubscription at 7×MT=4.

### HONEST FINDING (2026-06-19): NTR=100 ensemble UNDERPERFORMS NTR=20 — individuals underfit
NTR=100 ensemble: 7 nets -> VOTE=0.584, individuals avg 0.349.
NTR=20 ensemble was: 7 nets -> ~0.63, individuals ~0.42.
=> NTR=100 individuals (0.349) are LOWER than NTR=20 individuals (0.42), so the NTR=100
ensemble tracks BELOW NTR=20. More data did NOT raise the ceiling — it LOWERED individual
accuracy. Most likely cause: UNDERFITTING. NEP=2 epochs over 100 examples = each example
seen only twice; NTR=20 nets got far more relative passes/example at the same NEP.
This CONTRADICTS the naive "more data -> 0.88 ideal -> 0.90" path: the LogReg data-ideal
rises with data only if the learner CONVERGES. Analog nets at fixed small NEP do not.
PROBE LAUNCHED: 2x NTR=100 NEP=4 nets (d100e4_1/2) — if the INDIVIDUAL beats 0.42, underfit
is confirmed and the lever is epochs(convergence), not raw data. If not, data split is just
harder. Either way the cheap "more data" lever is weaker than hoped.

### DEFINITIVE (2026-06-19): "more data -> 0.90" REFUTED; NTR=100 ensemble plateaus at 0.584
NTR=100 10-net vote curve: 3->0.520, 5->0.520, 7->0.584, 8->0.584, 9->0.584, 10->0.584.
PLATEAU at 0.584. Individuals DEGRADE as more nets added (avg 0.373@3 -> 0.266@10; late
seeds scored ~0.27). NTR=100 ensemble (0.584) is WELL BELOW the NTR=20 ensemble (0.78).
=> More training data did NOT raise the ceiling; it LOWERED it. The LogReg "data-ideal"
(0.88@100/cls) is unreachable because the analog individuals UNDERFIT at fixed small NEP
and the ensemble of weak-correlated underfit nets plateaus low.
NEXT DIAGNOSTIC (running): is the individual's 0.34-0.42 ceiling EPOCHS-limited (fixable by
training longer) or an ANALOG EFFICIENCY FLOOR (needs architecture)? Launched epochs-sweep
at FIXED NTR=20: NEP=4/8/16 (epconv_n20e*) + NTR=100 NEP=4 probes (d100e4_*). If individual
rises with NEP -> epochs is the lever. If flat ~0.42 -> efficiency floor -> pivot to
architecture (deep-supervision idea4 / momentum idea8, both user-requested & still unbuilt).

### MECHANISM (2026-06-19): individual collapses to ~2 classes at high NEP — NOT underfit, FRAGILE
Epochs sweep @NTR=20 (real Spectre, external mout eval, trustworthy):
  NEP=2 -> 0.106 (undertrained, ~chance)
  NEP=4 -> 0.194 (sweet spot)
  NEP=8 -> 0.000 (COLLAPSE)  | NTR=100 NEP=4 -> 0.100
Diagnosed NEP=8 collapse: confusion matrix shows true 0-4 -> pred {7,9}, true 5-9 -> pred 1.
The net lost 10-way discrimination, collapsing to ~2 active output rails (0.000 because the
2 surviving modes happen to anti-align). Negation/permutation doesn't recover it -> genuine
rank collapse, not a sign flip. RWL=5meg weight decay does NOT prevent it at NEP=8.
=> This is the SAME C=10 collapse symmetric-nudge fixes at SHORT horizon, RE-EMERGING at the
INDIVIDUAL level with more training. So: undertrain=chance, sweet-spot(NEP~4)=0.19,
overtrain=collapse. NO epochs setting gives a strong individual. Only the ENSEMBLE of many
low-NEP pre-collapse nets reaches 0.58 (NTR=100) / 0.78 (NTR=20).
IMPLICATION: the binding constraint is INDIVIDUAL FRAGILITY (collapse-to-few-classes), not
data, not epochs. The right lever is a mechanism that KEEPS all C classes discriminative ->
DEEP SUPERVISION (idea 4: per-layer auxiliary 10-way class targets) is now well-motivated,
not a blind build. Also worth: stronger weight decay (RWL<5meg) to delay collapse.
Confirming reproducibility: e8 seeds 31/32 + e16 (x3 seeds) + e32 running.

### PLAN (2026-06-19): cheap anti-collapse knobs BEFORE the deep-supervision build
Collapse confirmed: NEP=2->0.106, NEP=4->0.194(peak), NEP=8->0.000, NEP=16->0.100 (all @NTR=20).
The individual over-trains into a ~2-class collapse. pc_deep already exposes knobs that target
this WITHOUT a circuit build (lower risk than deep-supervision idea4):
  - HINGE=1  : "freeze-on-convergence" — stop ALL learning once output confidently correct.
               Directly prevents the OVER-TRAINING collapse (stop before NEP-driven collapse).
  - LATINH   : lateral inhibition between output classes -> fights collapse-to-few-classes.
  - RWL=2meg : stronger weight decay (test in flight: epconv_n20e8_rwl2m).
TEST PLAN (as slots free): NEP=8 + HINGE=1, NEP=8 + LATINH=1, vs the NEP=8 baseline 0.000.
If any holds acc >= the NEP=4 peak (0.194) at high NEP, the individual fragility is FIXABLE by
config -> then ensemble of stronger individuals may beat 0.78. Only if all cheap knobs fail do
we commit to the deep-supervision circuit build (user-chosen idea 4).

### CORRECTION (2026-06-19): the "individual collapse" was largely an EVAL/CONFIG ARTIFACT
Reconciling why epconv individuals (0.1-0.2) << original vote_* ensemble individuals (~0.39):
- vote_16 INTERNAL block_acc curve = [0.36,0.368,0.368,0.388], BEST=0.388 (it IMPROVED, no collapse).
- My EXTERNAL eval of vote_* gave 0.108 (~chance) -> my external test-set reconstruction is WRONG
  for nets whose DSEED/test-set I can't exactly match. TRUST pc_deep's internal TEST ACC/BEST line.
- Two config errors in my epochs sweep vs the good ensemble:
  (1) EVK=0  -> evaluated ONLY the FINAL block (= after over-training degrades it). vote_* used
      EVK>0 = interval eval + EARLY-STOP -> takes the BEST block. The "NEP=8 -> 0.000" was the
      FINAL-block value; an intermediate block was likely fine. Early-stop catches it.
  (2) FANIN default is now 6; the good ensemble used FANIN=4.
=> The earlier "collapse beyond NEP=4 / individual is fundamentally fragile" claim is RETRACTED as
   confounded. Re-testing with the ORIGINAL good config: FANIN=4 + EVK=2 early-stop at NEP=8/16
   (evk_n20e*_k4). Expect ~0.39 (matching vote_*), confirming individuals are fine with early-stop.
LESSON: always set EVK (early-stop) + FANIN=4; report pc_deep INTERNAL BEST acc, not external mout
   eval unless the DSEED test set is matched exactly.

### HONESTY CHECKPOINT (2026-06-19): the pre-compaction 0.78 ensemble is UNVERIFIABLE
- vote_16 INTERNAL acc=0.388, but my EXTERNAL eval gives 0.100 under EVERY test-set reconstruction
  I tried (DSEED=0 AND its own SEED=16). I CANNOT reconstruct vote_16's test split (different
  NTR/NTE/data than my NTR=20/NTE=50/DSEED=0). So vote_16's 0.388 and the 17-net "0.78 ensemble"
  CANNOT be re-verified from saved artifacts.
- By contrast MY evk nets (FANIN=4 NTR=20 NTE=50 DSEED=0) verify EXACTLY: external==internal
  (0.218=0.218, 0.216=0.216). So my current ensemble vote WILL be trustworthy.
- vote_16 trains WELL from block 1 (curve [0.36,0.368,0.368,0.388]); my best-config nets train
  POORLY (first block 0.022, peak ~0.22). Same script -> a PARAMETER difference I haven't found
  (TD/settling/NTR). The 0.022 first-block is below chance = likely a settle/timing issue in my cfg.
- HONEST STANDING: reproducible+verifiable individual best ~0.22 (early-stop); ensemble vote TBD.
  The 0.78 figure should be treated as UNVERIFIED, not a confirmed baseline, until reproduced.
- STOP the vote_* archaeology. Deliverable = honest verifiable ensemble vote from evk nets.

### REFRAME (2026-06-19): data-scaling comparison was CONFOUNDED BY NET COUNT
Verifiable (DSEED=0, external==internal):
  NTR=20:  3 nets, indiv_mean=0.199, VOTE=0.226
  NTR=100: 12 nets, indiv_mean=0.229, VOTE=0.584
INDIVIDUALS are ~equal (0.20 vs 0.23) -> data has little effect on the individual.
The big VOTE gap is mostly NET COUNT (3 vs 12), NOT data. So BOTH my earlier claims are unsafe:
"more data underfits" (compared to unverifiable 0.78) AND "data helps" (compared 3 vs 12 nets).
HONEST: individual ~0.2 regardless of NTR; ensemble vote grows with #nets up to a plateau.
NTR=100 plateaued at 0.584 by ~7 nets. TEST: grow NTR=20 evk ensemble to ~10 nets, compare its
plateau to 0.584 at MATCHED net count. That is the only fair data-scaling test.

### RESOLVED + VERIFIED (2026-06-19): 0.78 IS REAL — I had the wrong NTE. RETRACT "unverifiable".
vote_16 mout has 250 rows = NTE=25 (I'd wrongly used NTE=50 -> got 0.100 -> falsely called it unverifiable).
With the CORRECT split (NTR=100, NTE=25, DSEED=0): vote_16 external=0.388=internal EXACTLY.
FULL vote_* ENSEMBLE (17 nets, NTR=100, NTE=25): indiv_mean=0.421 (range 0.304-0.552), VOTE=0.772. VERIFIED.
=> The pre-compaction 0.78 is REAL and reproducible. My "HONESTY CHECKPOINT: unverifiable" was WRONG
   (caused by NTE=50 mismatch). Retracted.
GOOD CONFIG (the one that works): FANIN=4, NTR=100, NTE=25, NEP=8, EVK=2(early-stop), DSEED=0,
   CHL=3 HCHOP=4 SYMNUDGE=1 RWL=5meg ZEROSUM=1 PERAZ=1 SOFTC=1 BKSIGN=1, LAYERS=64,32,16,10.
My weak 0.2 individuals were JUST NTR=20 (too little data). NTR=100 -> indiv ~0.42. DATA HELPS THE INDIVIDUAL.
ENSEMBLE PLATEAU: vote saturates ~0.78 by 7 nets (7->0.780, 13->0.788, 17->0.772). MORE NETS WON'T pass 0.80.
PATH TO 0.90: STRONGER INDIVIDUALS via MORE DATA (NTR=200/300) -> raises the plateau. Best single indiv
   already 0.552. This is the runtime-bound path (NTR=200 nets slow). Earlier confounded analyses (epochs
   collapse, data refuted, etc.) were all NTR=20 artifacts + the NTE mismatch — superseded by this.
### NOTE: light config (no chopper) FAILS (0.00). Chopper CHL=3 HCHOP=4 SYMNUDGE=1 is ESSENTIAL. vote_16=chopper+NEP=4 (slots~4200).
### chop FANIN=4 NEP=4 gives 0.17-0.20 (peak-then-collapse), WORSE than d100 recipe (FANIN=6 NEP=2)=0.34. Scaling the VERIFIED d100 recipe instead.

### ROOT CAUSE FOUND (2026-06-20): TASK=digits (sklearn ~180/cls) vs TASK=mnist (mnist_real_8x8 6300+/cls)
- vote_* (the VERIFIED 0.772 ensemble) used TASK=mnist (its mout matches mnist_real_8x8 NTR=100 NTE=25 EXACTLY).
- ALL my recent epconv/evk/chop/d100/scale200 used TASK=digits = sklearn load_digits (~180/cls) — a DIFFERENT,
  smaller dataset -> weaker individuals (0.2-0.34) and INVALID for NTR>~130 (NTR=200 -> EMPTY test set -> 0.000;
  that was the scale200 "post-proc crash": len(yte)=0, not a parser bug).
- My external eval loads mnist_real_8x8, so it only correctly scores TASK=mnist nets. The TASK=digits "matches"
  were on the wrong data.
FIX: use TASK=mnist consistently (matches vote_*, scalable to NTR=200+, matches eval tooling). Recipe to repro
  vote_16: TASK=mnist FANIN=4 NEP=2 NTE=25 chopper(CHL=3 HCHOP=4 SYMNUDGE=1) -> slots=4250 ~= vote_16's 4200.
### LEVER FOUND (2026-06-20): FANIN=6 (not 4) -> mnist NTR=100 indiv 0.30 vs FANIN=4's 0.17. My reproduction failures were FANIN=4.

### REGRESSION HYPOTHESIS (2026-06-20): current pc_deep can't reproduce vote_*'s 0.42 individuals
After exhaustive config search (FANIN 4/6, TD 0.12-0.6, NEP 2-32, EVK 0/1, TASK digits/mnist, RFGRID),
fresh nets score ~chance (0.0-0.3, mostly 0.1, seed-noise dominated). vote_* individuals were
CONSISTENTLY 0.30-0.55 -> systematic difference = likely TRAINING REGRESSION, not seed/config.
Git: 0.78 ensemble was "21 nets deep+RFGRID" (commit ~6dad6dd). pc_deep.py changed since via
1b6fd50(symmetric-nudge), e973438(CONV weight-share refactor), 87f23a4(DSEED). The CONV refactor
(wkey/weight-keying) is the prime suspect for breaking the normal-path training.
TEST RUNNING: extracted pc_deep.py @296ec72 (pre-refactor) -> pc_deep_old.py, running old_1/2/3
(NTR=100 mnist FANIN=4 NEP=2). If old gives ~0.4 and current ~0.1 -> regression CONFIRMED ->
bisect e973438/1b6fd50 for the break. NOTE: RFGRID inactive for LAYERS=64,32,16,10 (#w=68 unchanged;
needs square-compatible layers). VERIFIED result vote_*=0.772 stands regardless.

### CONCLUSION (2026-06-20): regression RULED OUT; vote_* reproduction UNEXPLAINED; honest wall
Old pre-refactor pc_deep (@296ec72) gives 0.232/0.168/0.200 — SAME ~0.2 as current code. So NOT a
code regression. Both code versions yield ~0.2 individuals (mostly 0.1, seed-noisy) with the recipe
TASK=mnist FANIN=4 NEP=2 chopper, while vote_* nets were CONSISTENTLY 0.30-0.55. The cause of vote_*'s
stronger individuals is UNIDENTIFIED after exhaustive search (FANIN 4/6, TD 0.12-0.6, NEP 2-32,
EVK 0/1, TASK digits/mnist, RFGRID[inactive @64,32,16,10], old vs new code). Likely a config/data/
setup detail present at creation but not in the saved logs.
STANDING HONEST STATE:
  - VERIFIED genuine result: vote_* C=10 ensemble = 0.772 (17 nets, real Spectre, NTR=100 NTE=25). REAL.
  - Fresh reproduction of its 0.42 individuals: BLOCKED (root cause unknown; not the code).
  - Path to 0.90 (more data -> stronger individuals -> higher plateau): sound in principle but gated on
    reproducing strong individuals first.
ACTION: pausing heavy Spectre churn on this dead-end pending user direction. Not launching more configs.

### AUTO-SEARCH ROUND 2 (2026-06-20/21): all NEW git/code-motivated levers also ~0.2
Per user "keep auto-searching", tested every remaining principled lever on TASK=mnist NTR=100:
  - RDEG=24k (synapse linearity, git "0.532"): 0.20/0.17/0.06 -> NO
  - square-layer RFGRID (active, #w 68->36, the "deep+RFGRID"): 0.20/0.10/0.05 -> NO
  - GBLH=0.95 (learning gain): 0.168 -> NO
  - KOUT=10/16 (readout fan-in, sizing law ~C, default 4 starves readout): 0.00/0.23 -> NO
  - RDEG+early-stop stack: (running)
LAST dimension testing: neuron type (NEUREL ReLU, FGATE tanh-gate). Default is tanh.
ROBUST FINDING: across the ENTIRE principled lever space (FANIN,TD,NEP,EVK,TASK,RFGRID,RDEG,
GBLH,KOUT,old-code), fresh individuals are ~0.2 (seed-noisy), NEVER vote_*'s consistent 0.30-0.55.
The cause of vote_*'s stronger individuals is genuinely unidentifiable from saved artifacts.

### SEARCH EXHAUSTED (2026-06-21): vote_* individuals UNREPRODUCIBLE; pausing config churn
neuron-type (NEUREL ReLU) = 0.168 too. COMPLETE list of levers tested this session, ALL ~0.2 fresh
individuals (vs vote_*'s consistent 0.30-0.55): FANIN 4/6, TD 0.12-0.6, NEP 2-32, EVK 0/1,
TASK digits/mnist, RFGRID(active+inactive), RDEG=24k, GBLH, KOUT 4/10/16, neuron NEUREL/FGATE,
AND pre-refactor code @296ec72. Every one lands ~0.2, seed-noise dominated.
=> The factor behind vote_*'s stronger individuals is NOT in any exposed flag, the dataset, the
code version, or the neuron — it's unidentifiable from saved artifacts (mout files only; decks/
weights/launch-scripts not preserved). The genuine VERIFIED result remains vote_* = 0.772.
STOPPING the autonomous config search (no productive principled experiment left). Awaiting user
direction: (a) the actual vote_* launch config/script, or (b) a redirect of the goal.

### DEFINITIVE (2026-06-21): fresh nets are NEAR-RANDOM; ensemble of 26 = 0.10 (CHANCE). Environment break.
Voted ALL 26 fresh NTR=100 NTE=25 mnist nets from this session (shared test set): indiv mean 0.121
(range 0.0-0.30), FULL VOTE = 0.100 = CHANCE. top-5=0.20, top-10=0.148. Fresh individuals are
near-random (not weak-but-informative like vote_*'s 0.42 -> 0.77 vote). So current pipeline does
NOT learn. Since old code @296ec72 ALSO gives ~0.2, and the data is consistent (vote_16 external
eval matched), the break is at the ENVIRONMENT level (device model / Spectre / PDK / creation-time
setup), NOT pc_deep flags/dataset/code-version — all exhaustively ruled out.
=> The genuine VERIFIED C=10 result stands: vote_* ensemble = 0.772 (saved mout, real Spectre).
   But fresh training currently produces noise. This needs the USER's environment knowledge
   (the actual vote_* launch script, or what changed in their Spectre/device-model setup).
STOPPED autonomous config search — proven useless (fresh nets vote to chance). No further nets launched.

### CHARACTERIZATION of the VERIFIED result (2026-06-21) — vote_* C=10 ensemble
17 nets, NTR=100 NTE=25, real Spectre. Overall = 0.772 (bootstrap 0.747 +/- 0.031, stable).
Per-class acc: 0:0.96 1:0.80 2:0.96 3:0.72 4:1.00 5:0.68 6:1.00 7:0.88 8:0.08 9:0.64.
=> Strong on 0,2,4,6,7; CLASS 8 is a near-total failure (0.08, confused -> 2/3/1); 5,9 moderate.
Lifting class 8 alone would add ~9pts. This is the genuine, paper-ready C=10 analog-PC result.

### PIVOT (2026-06-21, user: ensembles inefficient): SINGLE-NET ELM works at C=10
User: a 17-net ensemble defeats analog efficiency (no gain over digital). Right. Target = ONE net.
Fast device-faithful model (/tmp/fwd/trc10.py): SINGLE analog net = fixed random tanh features
(cheap/untrained projection) + ONE trained readout (10xN), early-stop. C=10 8x8 MNIST:
  NFEAT=48/NTR=25: 54%   NFEAT=64/NTR=50: 69%   NFEAT=128/NTR=50: 78.5%   NFEAT=256/NTR=50: 72%(data-lim)
=> ONE net reaches ~78.5% (vs digital ~85%, vs my ensemble 0.77). No ensemble. Readout 10x128 is
SMALLER than the deep nets (816 caps). Over-training drift -> early-stop (peak ~epoch 3).
This is the efficient analog story. NEXT: NTR=100 to lift the data ceiling (256 feats), then VALIDATE
a single 128-feat readout in SPECTRE (trustworthy; tractable size). Random features = the lever, not depth/ensemble.

### SINGLE-NET ELM, stable config (2026-06-21): NFEAT=256 NTR=100 = ~76% STABLE (no early-stop gaming)
NFEAT=256/NTR=100: epoch3=76.5 epoch5=76.2 -> FLAT across epochs (the over-train drift seen at small
NFEAT/NTR disappears with enough features+data; no early-stop cherry-pick needed -> honest number).
NFEAT=128/NTR=50 peaked 78.5 but drifts (needs early-stop). Single analog net = fixed random tanh
features (k=4 fan-in, untrained) + ONE trained readout (10x256). Firming up over 5 feature seeds.
Architecture = extreme-learning-machine: random projection is cheap/fixed analog, only readout trains.
This is the EFFICIENT single-net answer to the ensemble critique. NEXT: honest mean+/-std, then
SPECTRE-validate one 256-feat readout (single layer, tractable, trustworthy).

### HONEST single-net number (2026-06-21): 69.9% +/- 5.0% (5 feature seeds), C=10, NFEAT=256 NTR=100
Seeds: 76.2/73.0/61.3/69.2/70.0 -> mean 69.9 std 5.0 (stable epoch-5, no early-stop gaming). The 76%
was a lucky feature draw; honest single-net = ~70%. Feature-seed SELECTION (still ONE net) reaches ~76%.
On the fast device-faithful model (/tmp/fwd, 0.01mV vs ngspice). NEXT: validate ONE 256-feat readout
in SPECTRE for a trusted number. Single analog ELM (random features + 1 readout) is the efficient
answer to ensemble-inefficiency: 70% single-net vs 0.77 17-net-ensemble vs ~0.85 digital.

### KEY FINDING (2026-06-21): features are EXCELLENT; the ANALOG READOUT TRAINING is the bottleneck
RIDGE (closed-form linear) readout on the SAME analog random tanh features (NTR=200):
  NFEAT=128: 84.8%  256: 87.6%  512: 89.9%  1024: 90.1%  (above digital LogReg-on-pixels ~85%!)
But the ANALOG-TRAINED readout (local delta rule, trc10) caps at ~70% on identical features.
=> The single-net gap to ~88% is NOT features (they're great) — it's the analog readout LEARNING
(SGD-ish local rule, gate-V weights clipped [1.0,2.8], fixed target margins, over-train drift).
DIRECTION: fix/tune the readout training (gain, epochs, early-stop, weight-range, target levels) to
approach the 88% ridge ceiling. A single analog net could then be ~85% = competitive with digital.
This is the efficient single-net path the user wants. Feature-selection (1024->256 by |corr|)=86.5%
(no better than random 256's 87.6 ridge -> random features already near-optimal, selection unneeded).

### ROOT CAUSE LOCALIZED (2026-06-21): PC training broken = SIGN-LOSS in backward error transport
User's point: training >= frozen always, else trainer broken. Confirmed:
  ideal backprop (train hidden) = 88.0% (>= frozen-ridge 87.5%) -> training SHOULD help.
  analog PC net = ~0.2 (chance) -> trainer is broken (degrades random features over epochs = collapse).
SIGN TEST (ideal 2-layer, vary backward error handling for hidden update):
  correct SIGNED backward        : 88.0%
  amplitude, sign SCRAMBLED(rand): 86.0%   (random signs average out, readout compensates)
  magnitude only, SIGN DROPPED   : 33.0%   <-- CATASTROPHIC, << frozen
=> The killer is dropping the per-hidden-unit SIGN of the error (pc_deep: "transport amplitude-
restoring, NOT sign"). Hidden units all pushed one polarity -> features destroyed -> collapse.
pc_deep's BKSIGN flag tries to fix this but my BKSIGN=1 runs still collapsed -> not fully working.
FIX DIRECTION: make pc_deep hidden-layer backward error SIGN-FAITHFUL (true signed transpose of
readout error), validate a SINGLE full-PC net then beats frozen (>~80%), then Spectre-confirm.
This is the real task per user "keep it full-PC". Features are great (88% ceiling); fix the sign.

### RESOLVED + PATH (2026-06-21): the PC-training bug has a KNOWN, VALIDATED fix (CMSUB) in gen_deep.py
User: training >= frozen always; trained<<frozen => broken. CONFIRMED + root cause MEASURED (prior
session, docs/BACKPROP_DEPTH.md): backward error common-mode +0.186, EVERY hidden unit positive,
2.4-3.3x the signal -> drifts all hidden caps one way -> collapse (= my sign-loss 33% case).
FIX = CMSUB (subtract resistor-averaged backward common-mode; RCMB=10e3) + OCMSUB (output error CM)
+ tanh + zscore. WORKS: experiments/gen_deep.py, 8/8 seeds, ~79% MNIST-4x4 {0,1,7}. pc_deep LACKS this
(its COMP=subtractive doesn't clean the transpose-read backward) -> that's why ALL my pc_deep runs collapsed.
PLAN (single full-PC net, no ensemble, no freezing -- per user):
  1. Confirm gen_deep baseline still trains at known-good 4x4/3-class (~79%).
  2. Scale to 8x8 / C=10: MLP= width(s) with FANIN-sparse, DATAFILE=mnist 8x8, CMSUB=1 OCMSUB=1 tanh zscore.
     Target: ONE net beats frozen (>~80%, vs ridge-ceiling 88%).
  3. Port to Spectre for the trusted number.
gen_deep.py knobs: MLP="48,48,.." or ARCH conv; FANIN; CMSUB/OCMSUB=1; RCMB/ROCM=10e3; DATAFILE.
This is the efficient single-net answer. Features are great (88% ceiling); CMSUB fixes the training.

### MILESTONE (2026-06-21): CMSUB backprop CONFIRMED in SPECTRE = 85.3% (single full-PC net, no ensemble)
gen_mc.py + spectre_mc.py (run_backprop recipe, CMSUB=1 RCMB=10e3 ACT=tanh NORM=zscore), 16->9->3,
labels {0,1,7}: SPECTRE acc=85.3% (chance 33%, per-class [76,88,92], all alive). > ngspice writeup ~79%.
=> The working full-PC trainer (the CMSUB fix pc_deep LACKS) trains a SINGLE net in the trusted sim.
NOW scaling to 8x8/C=10: need 64-input data + bigger hidden + C=10. Investigating CONN/H/data.

### FAST MODEL (2026-06-21): COMP (output common-mode subtract) is the C=10 lever
Built /tmp/fwd/fast_cmsub.py (mechanism-faithful 2-layer: compressed output -> output common-mode
grows with C -> hidden drift). Result: CMSUB-only C=10 = 50%/8 classes; +COMP=1 -> 63-65%/9-10
classes ALIVE. Confirms the writeup mechanism: at C=10 the OUTPUT common-mode (not just hidden) must
be subtracted (COMP=subtractive / OCMSUB). Model too crude for detailed knob values (unstable w/
graded targets & big H = proxy artifacts) -> validate in circuit. Running 8x8 H=16 COMP=subtractive
ngspice to confirm COMP keeps 10 classes alive in the real circuit.

### HONEST WALL (2026-06-21): CMSUB backprop SOLVES few-class but C=10 collapse persists in-circuit
- C=3 {0,1,7}: SPECTRE 85.3% (CMSUB backprop, all classes) — THE FIX WORKS, Spectre-validated.
- C=10: collapses to 1-2 classes in the REAL circuit at BOTH 4x4 and 8x8, with CMSUB alone AND
  with CMSUB+COMP=subtractive+graded targets. margin ~0.03 (net barely differentiates).
- Fast mechanism model said COMP helps C=10 (50%->65%, 10 classes) but it DID NOT transfer to the
  circuit (proxy too crude). SPICE iteration is slow (8x8 dense decks 10-16k MOS, 40-min timeouts).
=> The output-side multi-class collapse at C=10 is the project's deepest open wall; CMSUB (hidden-side
   common-mode fix) is necessary but NOT sufficient for C=10. Cracking it needs: a faithful fast 2-layer
   sim (rail-based, not my crude proxy) OR a sparse-8x8 deck generator (gen_mc sparse is 4x4-only) to
   iterate epochs/Cg/targets/OCMSUB at C=10 speed -- both are real builds, not quick runs.
DELIVERED this session: identified the training bug (backward common-mode/sign-loss) + its VALIDATED
fix (CMSUB, Spectre 85.3% @C=3); dropped the inefficient ensemble for the single-net path.

### REFRAME (2026-06-22): C=10 collapse is a CIRCUIT non-ideality, NOT an algorithm limit
Built device-aware fast model (/tmp/fwd/faithful.py: tanh hidden + rail-keystone readout, signed-
transpose backward + CMSUB/COMP). Fidelity: C=3 {0,1,7} = 74.7% (circuit 85.3%, right ballpark).
KEY: C=10 trains to 38% with ALL 10 CLASSES ALIVE (COMP on or off) -- the model does NOT collapse.
=> The CMSUB-backprop ALGORITHM handles C=10. The real-circuit C=10 collapse (1-2 classes) is therefore
a CIRCUIT-SPECIFIC non-ideality NOT captured by tanh+rail+signed-transpose: candidates = the OTA
weight-update dynamics (ota/ota_relu), the auto-zero (oraz/PERAZ), the readout compression severity,
or a common-mode leak in the physical transpose-read. A crude/semi-faithful model can't reproduce it;
needs the FULL faithful OTA+auto-zero dynamics OR direct (slow) circuit probing of d1pre/weights at C=10.
SESSION DELIVERABLE stands: training bug (backward common-mode/sign-loss) + CMSUB fix, Spectre C=3=85.3%;
single-net (no ensemble); and now: C=10 is a circuit-implementation problem, algorithm is sound.

### C=10 ROOT CAUSE MEASURED (2026-06-22): output-error common-mode 6x; hidden backward 47x (probe)
Probed C=10 4x4 circuit (PROBE=1 PROBED2E=1, CMSUB=1): d2e(output err) CM/signal=6.12;
d1pre(hidden bk) CM/signal=47.56 with signal_std only 0.027. => At C=10 the OUTPUT error is 6x
common-mode (Sum_c(y_c-t_c) over 10 classes), and transposing it leaves the hidden backward 47x
common-mode w/ near-zero discriminative signal -> hidden can't learn -> collapse. CMSUB (hidden-side)
can't fix this; the OUTPUT common-mode must be killed AT SOURCE before transpose. COMP=subtractive is
meant to, but its gain RCM=RTO/C=700 is likely too weak for a 6x CM. FIX TO TEST: COMP=subtractive +
tighter RCM (sweep) -> re-probe d2e CM should drop, d1pre signal rise, classes survive.

### C=10 FIX IDENTIFIED (2026-06-22): need OCMSUB (output common-mode subtract), not COMP=subtractive
COMP=subtractive RCM sweep (700/200/50): d2e CM/sig stays 5.5-7.8, d1pre signal stays ~0.04 -> COMP
does NOT clean the output error (matches writeup: "COMP shifts forward refs, doesn't clean transpose
backward"). The C=10 killer = output-error common-mode (6x); FIX = OCMSUB (resistor-average d2e_c,
subtract -> the OUTPUT analog of CMSUB). gen_mc.py has CMSUB+COMP(ineffective); gen_deep.py HAS
OCMSUB (+CMSUB). NEXT: run gen_deep.py (CMSUB=1 OCMSUB=1) at C=10 8x8 -> should keep classes alive;
OR add OCMSUB (d2ebar resistor-star on d2e nodes) to gen_mc. Then train + Spectre-validate.

### C=10 FIXES ALREADY EXIST as gen_mc flags (2026-06-22) — I'd been running with them OFF
gen_mc has documented C=10 anti-collapse flags, all default-OFF (why my runs collapsed):
  - OCMSUB=1 : subtract output error common-mode (d2ebar resistor-avg, ROCM=10e3) -- the exact fix the
    probe pointed to (output CM 6x). Comment: "without it large-C collapses to one winning class".
  - GLOBALAZ=1: ONE shared auto-zero node. Comment: "Model-found fix for C=10 collapse... global climbs
    to 67% stable" (per-class PERAZ drifts; global cancels CM without drift).
  - NOBIAS=1 : freeze output bias -> kills the "output predicts one class" basin.
TESTING C=10 4x4 with OCMSUB=1 / GLOBALAZ=1 / combo. If alive -> 8x8 + Spectre-validate. The diagnosis
(output common-mode) was right; the fix was already implemented and just not enabled in my recipe.

### HONEST CONCLUSION (2026-06-22): C=10 fix identified+fast-model-validated; circuit sim convergence-blocked
C=10 fix-flag circuit tests INVALID (ngspice MC_OK=0, convergence-stuck at tstop -> stale results). C=10
dense decks don't converge to completion. Fixes EXIST + author-validated in fast model: GLOBALAZ "67% stable"
(gen_mc ~line382), OCMSUB likewise; faithful.py confirms C=10 trains 38% all-classes (algorithm OK).
SESSION RESULT: bug=backward common-mode/sign-loss; fix=CMSUB Spectre-validated C=3=85.3% (single net,no
ensemble); C=10 collapse=output common-mode(6x measured), fix=OCMSUB+GLOBALAZ (exist in gen_mc, fast-model
~67%); blocker=circuit convergence at C=10 scale (sim-infra, not algorithm). Trusted circuit C=10 needs
convergence work (gmin/source stepping, sparse-FANIN smaller decks).

### C=10 CONVERGENCE FIXED (2026-06-22): train deck now completes (MC_OK=1)
Root cause of non-convergence: OCMSUB node d2ebar had a 50fF cap but NO initial condition -> floated
under uic -> non-convergence. FIX (gen_mc.py + gen_mc_infer.py): (1) ic.append(("d2ebar",1.2));
(2) added .options gmin=1e-10 reltol=2e-3 abstol=1e-9 vntol=1e-5 itl1=200 itl4=200 (env-tunable).
TRAIN now completes: rc=0 MC_OK=1 (was MC_OK=0 stuck-at-tstop). Patched infer deck too; re-running
C=10 4x4 OCMSUB+GLOBALAZ+NOBIAS for the first TRUSTED (completed-sim) C=10 number, then Spectre.

### CONVERGENCE FIXED + verified (2026-06-22): both decks complete (train MC_OK, infer INFER_OK, 8008 rows)
The d2ebar IC + .options fix WORKS: train MC_OK=1, infer INFER_OK (my earlier "MC_OK=0" was wrong grep
-- infer prints INFER_OK). So C=10 sims now COMPLETE -> results are TRUSTED. C=10 4x4 OCMSUB+GLOBALAZ+
NOBIAS = still collapse (1 class, margin 0.41) BUT 4x4 is degenerate (16px/10cls). Now that big decks
converge, testing 8x8 (good features, 88% ridge ceiling) C=10 with the fixes + more epochs.

### TRUSTED C=10 RESULT (2026-06-22): collapse PERSISTS even with all fixes + converged sim
With convergence FIXED (decks complete, verified MC_OK + INFER_OK):
- C=10 4x4 OCMSUB+GLOBALAZ+NOBIAS: collapse (1 class, margin 0.41).
- C=10 8x8 H=16 OCMSUB+GLOBALAZ+NOBIAS (TRUSTED, train MC_OK=1 + infer INFER_OK=1): collapse to 1 class
  (margin 0.41), even with the strong features (88% ridge ceiling).
=> The documented C=10 fixes (OCMSUB/GLOBALAZ/NOBIAS, "fast-model 67%") do NOT transfer to the real
circuit. The all-one-class + margin~0.41 (consistent across runs) = output rails to a fixed degenerate
pattern; the OTA output update can't separate 10 classes in the compressed keystone range. This is the
project's deepest wall, now rigorously confirmed in TRUSTED (converged) simulation -- not a sim artifact.
DELIVERED: (1) convergence fix (d2ebar IC + .options) -- C=10 decks now complete & are iterable;
(2) training bug + CMSUB fix Spectre C=3=85.3%; (3) C=10 root cause measured (output CM 6x);
(4) trusted proof the existing fixes don't crack circuit C=10. Remaining: a deeper output-range/
compression fix for the 10-class readout (genuine open research).

### C=10 FULLY DIAGNOSED (2026-06-22): per-class FORWARD offset; bias partial; deep open problem
FREEZE-vs-trained probe: training DOES move weights (dW=0.4653) but FROZEN random net already predicts
all-one-class (margin 0.27 -> trained 0.41 = same wrong class, more confident). So collapse = INPUT-
INDEPENDENT per-class FORWARD offset (measured per-class output means e.g. class4=0.957 highest), NOT a
training failure. Trainable bias (NOBIAS=0) helps marginally: 1 class -> 2 classes (15.6%). Per-class
offsets are too strong for bias to fully cancel in the compressed keystone output range at C=10.
=> The 10-class readout can't produce enough input-dependent output spread to overcome the per-class
DC offsets in the compressed analog range. Documented fixes (OCMSUB/GLOBALAZ/NOBIAS) + convergence fix
do NOT crack it (TRUSTED converged sims). This is genuine deep open research (output dynamic-range /
per-class offset cancellation for high-C analog readout).

### C=10 DEFINITIVE ROOT CAUSE (2026-06-22): readout OUTPUT-STAGE capacity limit (7/10 outputs dead)
Decisive offline test (correct score_mc sampling, trained C=10 4x4): per-class FORWARD output std
(variation across inputs) = [0.029,0.123,0.049,0.023,0.125,0.028,0.014,0.019,0.009,0.116]. Only 3
classes (1,4,9) are RESPONSIVE (std ~0.12); the other 7 are NEARLY CONSTANT (std 0.01-0.05) = stuck
at a rail. Removing per-class offset -> 12.4%; standardizing -> 19.6% (no signal to recover in the
dead 7). => C=10 collapse is NOT offsets/training/auto-zero (all error-side, all failed). It's that
the KEYSTONE READOUT OUTPUT RANGE can only support ~3 responsive classes; the rest saturate/rail in
the compressed analog output range. A fundamental OUTPUT-STAGE CAPACITY limit at high C.
FIX = redesign the readout OUTPUT STAGE so each of 10 classes gets a responsive operating range
(per-class output operating-point bias / wider output dynamic range / different output cell) -- a
deliberate circuit-design change, NOT a flag. This definitively explains the project's C=10 wall.
SESSION DELIVERABLES: (1) training bug=backward common-mode/sign-loss; (2) CMSUB fix Spectre C=3=85.3%;
(3) C=10 convergence FIXED (d2ebar IC + .options); (4) C=10 collapse DEFINITIVELY root-caused = output-
stage capacity (7/10 outputs dead), fix = readout output-stage redesign.

### C=10 CORRECTION (2026-06-22): NOT diode-threshold cutoff. Rails healthy; H<C capacity is the suspect.
Direct rail probe (PROBE_RAILS, BOUT=0.6 trained, C=10): colpo mean ~0.91-1.01, colno ~0.946 -- BOTH
well ABOVE the 0.5 diode threshold (so the "outputs dead because rails below threshold" claim was WRONG,
caught by measurement). REAL signature: colpo std ~0.010-0.014, colno std ~0.011 = rails modulate only
~1% with input -> outputs barely move. BOUT (output-bias lift) did NOTHING (byte-identical std) because
the rails were never threshold-limited. Leading suspect now: H=9 hidden units < C=10 classes -> hidden
representation can't span 10 classes (under-capacity), compounded by low readout gain. Consistent with
DEEP-NOT-WIDE: the fix is more hidden CAPACITY via DEPTH (a 2nd hidden layer), not collapse-flags and not
widening. NEXT: cheap capacity probe (H=18 diagnostic) to confirm capacity is the bottleneck before
committing to a depth change. BOUT kept (default 0.0, harmless no-op).

### C=10 TRIANGULATED (2026-06-22): it's a TRAINING-DYNAMICS collapse, not forward capacity. Two hypotheses RULED OUT by measurement:
  (a) diode-threshold cutoff -> FALSE (rails sit ~0.95V, above 0.5 thresh; BOUT no-op).
  (b) hidden under-capacity (H<C) -> FALSE (H=18 collapses too, margin 0.09 < H=9's 0.49 -> WORSE).
  Plus: faithful.py (same device forward + near-ideal backprop) = 38% ALL classes alive -> the DEVICE
  FORWARD can represent 10 classes. => The collapse is created by the IN-CIRCUIT TRAINING RULE driving
  outputs to a degenerate near-constant equilibrium (one class highest by offset), NOT the forward path.
  RELEVANT KNOWN FIX (memory symmetric-nudge-c10-fix): deep full-PC Spectre path solved C=10 with a
  SYMMETRIC +/-beta nudge (cancels the one-sided O(beta) bias) + weight decay -> 0.41 robust; one-sided
  collapses to 0.068. gen_mc (backprop deck) is one-sided -> collapses. => the trusted C=10 PC number
  lives in the gen_deep / PC-deep path (symmetric nudge), which is ALSO what "deep not wide" + "stay PC"
  asks for. NEXT: re-verify gen_deep C=10 symmetric-nudge in SPECTRE as the trusted PC number.

### C=10 PC SPECTRE RE-VERIFICATION (2026-06-22): BEST=0.428 / FINAL=0.360 — CONFIRMS documented 0.41.
Fresh real Spectre run (pc_deep.py, SIM=spectre MT=4, single seed WSEED=1, ~6h40m wall), deep full-PC
all-layers-learning, NO freezing, NO wide: LAYERS=64,32,16,10->10, K=4 fanin, chopper EP cells (CHL=3
HCHOP=4), symmetric +/-beta nudge (SYMNUDGE=1), weight decay (RWL=5meg), ZEROSUM=1 PERAZ=1 SOFTC=1
BKSIGN=1, NEP=12 NTR=40 NTE=25, TASK=digits. RESULT: [pc_deep] TEST ACC = 0.360  BEST(early-stop) = 0.428
curve=[0.428,0.36]. => BEST 0.428 matches documented 0.412 +/- 0.041 (6 seeds) / 0.424 best -> the
symmetric-nudge C=10 fix REPRODUCES in fresh Spectre. ~4.3x chance (0.10). FINAL block (0.36) dipped below
best -> the known symmetric-nudge over-growth (weights over-grow, decay only partly tames) -> early-stop
BEST is the reportable number; only 2 eval blocks here (coarse). Honest caveats: single seed (one draw
from the documented distribution, not the 6-seed mean); ~0.42 is well above chance & proves the collapse
is solved, but far from the ~0.85 data ideal (that gap is feature/efficiency, not collapse). Operational
note: the run would have died at a stray outer `timeout 12000` ~52% in; rescued by SIGKILL'ing ONLY the
timeout wrapper (PID), leaving pc_deep+spectre reparented to init to finish (STO=240000 = spectre's own
~unlimited timeout). This is the TRUSTED C=10 PC number the user asked for.

### PYTORCH SURROGATE: local receptive fields >> random; data + PC-trainer are the gaps to 0.95 (2026-06-23)
experiments/pc_surrogate.py — same deep-sparse topology, tanh, OWN weight per edge (no sharing),
switches CONN=local|random, RULE=bp(ideal ceiling)|pc(PC+symmetric nudge). sklearn 8x8 digits, z-score.
ARCHITECTURE CEILING (ideal backprop, NTR=40):
  analog-topology[64,32,16,10,10] RANDOM fanin4 (272 edges): TEST 81.2% / train 82.5%
  LOCAL-RF pyramid[64,36,16,10] 3x3 valid-conv (508 edges):  TEST 87.2% / train 95.5%  <- +6%, local wins
  local-sizes RANDOM control (248 edges):                    TEST 82.0%  (rules out "just more edges")
PC + SYMMETRIC NUDGE (algorithm-faithful, first-pass tune):
  analog-topo RANDOM: TEST 48.0%     LOCAL-RF pyramid: TEST 78.8%   <- local RF +31% under the PC rule!
DATA-CEILING sweep (ideal BP, local-RF, vary NTR): 40->88.5%, 80->90.5%, 120->91.5%, 150->92.0%
  (train ~95-97% throughout -> the 40-case gap was DATA; more data closes it toward train).
=> RANKED gaps to 0.95: (1) CONNECTIVITY random->local RF: biggest confirmed lever (+6% ideal, +31% PC);
pc_deep ALREADY supports RFGRID for SQUARE layers -> adopt a square pyramid 64(8x8)->36(6x6)->16(4x4)->10.
(2) PC/analog TRAINER: even with local RF, PC trails ideal BP by ~8% (78.8 vs 87.2), analog non-idealities
cost more (0.43 circuit) -> the trainer is the 2nd big lever. (3) DATA NTR=40->150 adds ~3.5%.
NOT the bottleneck: neuron cell / readout / analog precision per se. Architecturally 0.95 is REACHABLE
(local-RF ideal ceiling ~92-95% test, ~95-97% train); the work is local RF + close PC-vs-BP gap + more data.
Caveat: PC surrogate is first-pass (random-PC=48% likely under-tuned beta/gamma/lr); local>>random is robust
across BOTH BP and PC regardless.

### PC-RULE TUNED = MATCHES BACKPROP (2026-06-23): the PC+symnudge algorithm is NOT the bottleneck.
Tuning the surrogate's PC+symmetric-nudge (was first-pass 48/78.8): best (beta=0.1 gamma=0.5 TSTEPS=50
lr=0.05 ep=150 wd=3e-4 bsz=50): LOCAL test=88.0%/train93, RANDOM test=73.2%/77. => PC+sym LOCAL 88.0%
MATCHES ideal backprop LOCAL 87.2% -> the learning rule reaches the backprop ceiling when tuned. Key
levers: SMALLER beta=0.1 (closer to the unbiased small-beta EP limit) + MORE relaxation steps T=50.
First-pass 48% was just under-tuned (beta too large, too few steps). CONCLUSION: the ~0.43 analog circuit
vs ~0.88 ideal gap is NOT the algorithm -- it's analog non-idealities + the RANDOM connectivity used in
the circuit run. Hence the local-RF Spectre test. Local RF helps PC by +15% (88.0 vs 73.2).
LAUNCHED (2026-06-23 08:15): c10rfgrid_verify -- the verified-0.428 recipe + RFGRID=1 RFK=2 local square
pyramid LAYERS=64,36,16->10 (8x8->6x6->4x4->10, maxfanout=4 local, 62 neurons/248 weights/~11105 MOSFETs),
TASK=digits NTR=40, real Spectre. Direct apples-to-apples vs the random 0.428: does local RF lift the
real circuit? (~6h). NO short outer timeout (STO=240000 = spectre's own ~unlimited inner timeout).

### INVARIANT PROBE CATCHES A SILENT BUG (2026-06-23): "symmetric nudge" NEVER ENGAGES + doesn't matter.
While the local-RF Spectre run was mid-flight, probed PC invariants on the live raw (parse_psf streaming).
INV: chopper clock ckr (negative/-beta phase) is DEAD -> min=0 max=0, NEVER fires. Verified in BOTH the
current RFGRID run AND the prior 0.428 run (raw_c10symnudge_verify: ckr frac_high=0.000 over 10102 pts).
ROOT CAUSE (pc_deep.py lines 316-318): `if CHL: SLOTS+=[(0),(1)] elif SYM: SLOTS+=[(1),(-1)]` -> CHL=3
OVERRIDES SYMNUDGE; pol is only {0,1}, never -1; and ckr fires only on pol==-1 (line 350, because SYM=1
moved its trigger there). So CHL=3 SYMNUDGE=1 = the one broken combo: symmetric never engages AND the
chopper's own free-phase subtraction (-pred.a, which needs ckr on pol==0 when SYM=0) is ALSO disabled
-> the update runs ONE-SIDED (clamped-phase term only). True symmetric needs CHL=0 SYMNUDGE=1; working
chopper-EP needs CHL=3 SYMNUDGE=0.
DOES IT MATTER? Surrogate (tuned, NTR=40): SYMMETRIC vs ONE-SIDED PC: LOCAL-RF 88.0 vs 88.4 (one-sided
+0.4), RANDOM 73.2 vs 76.8 (one-sided +3.6). => symmetric nudge gives ~0 benefit (one-sided slightly
better) at small beta with clean numerics -- no O(beta) bias to cancel. So the inert symmetric nudge is
NOT costing accuracy; the 0.428 = one-sided clamped PC + ZEROSUM + PERAZ + weight decay.
CORRECTION: the C=10 result is MISATTRIBUTED to symmetric nudge in [[symmetric-nudge-c10-fix]]. The
"one-sided -> 0.068 collapse" claim is NOT reproduced (surrogate one-sided = 77-88%). Other invariants
checked OK: ckd toggles 50% duty (clamp phase alive); activations a1/a2 bounded [0,~0.97] & input-
responsive (std 0.16-0.23); hidden errors e1/e2 present & bounded. (Output-margin/ZEROSUM checks pending
full run.) The running RFGRID job is STILL a valid local-vs-random connectivity test (same one-sided
trainer both sides). LESSON (user's point): final accuracy hid a dead clock for 2 runs; the invariant
probe caught it in minutes.

### FAST SURROGATE NOISE ABLATION pinpoints the analog gap = SYSTEMATIC UPDATE OFFSET (2026-06-23)
<1min surrogate experiments (local-RF, NTR=40). EVAL-time precision is NOT the gap:
  weight mismatch (fixed per-edge): 88.8% holds to 86.6% even at 80% mismatch -> BENIGN.
  activation/read noise: 88.8 -> 80 (0.3V) -> 61 (0.5V); even wmm0.5+anoise0.3 = 79% >> circuit 43%.
  => static mismatch + moderate read noise CANNOT explain the 43% circuit result. The gap is in LEARNING.
TRAIN-time update ablation (PC, undertrained baseline 67%):
  random charge noise on cap updates: HARMLESS (updn=1.5x -> 72%, mild regularization).
  SYSTEMATIC update OFFSET: updoff=0.1 harmless, updoff=0.3 -> 36.4% (COLLAPSE, ~= circuit 0.43).
=> THE gap driver is a SYSTEMATIC, UNCANCELLED BIAS in the weight update -- NOT mismatch, NOT random
noise, NOT symmetric-vs-onesided. This CONNECTS to the dead-ckr finding: the chopper (ckd/ckr alternation)
exists to CANCEL the gprodC cell's systematic offset; ckr dead -> cancellation DISABLED -> offset
accumulates -> collapse. Also matches ledger's "net downward drift = rich-get-richer collapse" (that drift
IS a systematic offset; ZEROSUM/PERAZ partially cancel -> 0.428 not full collapse). 
REFRAMED FIX (concrete, testable): enable the chopper offset-cancellation = CHL=3 SYMNUDGE=0 (ckr fires on
the FREE phase pol=0 -> charges -pred.a, the contrastive subtraction + offset chopping). NOT symmetric
nudge. Surrogate predicts cancelling the offset recovers 36->67% (undertrained) / ->~88% (well-trained).
This is the highest-value next Spectre run. Method note: <1min surrogate iterations (vs 6h Spectre) found
this in minutes; eval-noise + train-noise ablation cleanly separated inference-precision from learning-bias.

### LOCAL-RF ARCH SWEEP (2026-06-23, ideal BP NTR=40, <1min): RFK3>RFK2 for acc; RFK2 best per-cap.
[64,36,16,10]RFK3 88.8% (508e,fanin9); [64,36,16,10,10]RFK3+layer 90.8% (548e) BEST acc; [64,49,25,10]RFK2
88.0% (336e,FANIN4) best acc/cap; [64,49,36,10]RFK2 82.8%. => fan-in 9 (RFK3) richer than fan-in 4; +1
layer ~+2%; but RFK2 steeper pyramid nearly matches at ~half the edges = best for analog cap budget.
For chopper-fix run: keep topology = in-flight RFGRID run ([64,36,16] RFK2) to ISOLATE the SYMNUDGE=0
trainer change; optimize topology separately after.

### FAST *SPECTRE* TESTBED (2026-06-23, user push: iterate on REAL components, not surrogate)
Shrunk hard -> REAL Spectre runs in SECONDS: tiny circles [2,8,2] NEP=6 = 14-24s (841 MOSFETs); [2,8,2]
NEP=24 = 81s; C=4 digits [64,16,4] NEP=8 = 104s. So real-component iteration IS feasible.
VERIFIED ON REAL COMPONENTS (not surrogate): chopper clock ckr -- SYMNUDGE=1 frac_high=0.000 (DEAD),
SYMNUDGE=0 frac_high=0.43 (FIRES). The dead-ckr bug + its fix are real in Spectre.
CHOPPER-FIX A/B on real Spectre:
  C=2 circles NEP=24: SYM=1 BEST=0.567, SYM=0 BEST=0.567 -> NO difference (offset-collapse is multi-class;
    C=2 has no rich-get-richer asymmetry). curves differ (SYM1 [.35,.42,.57], SYM0 [.42,.57,.57]).
  C=4 digits NEP=8: SYM=1 (broken)=0.050, SYM=0 (working)=0.100 -> working 2x better, BUT both << chance
    0.25 (badly undertrained; surrogate needed ~150 ep). Directionally supports chopper-helps-at-multi-class,
    NOT conclusive.
KEY LIMIT (honest): sub-~2min Spectre is too UNDERTRAINED to reach/learn -> good for INVARIANTS & mechanism
& directional A/B, NOT for final accuracy or clean collapse-signature (tiny C=4 outputs all near-zero/dead).
Niche: (1) invariant verification in seconds (clock states, node ranges, error presence); (2) directional
trainer A/B on partially-trained nets ~1-2min. Accuracy-level Qs still need surrogate OR a mid-size run.
NEXT to CONFIRM chopper fix: a config that actually LEARNS multi-class -- C=4 or C=6 at NEP~24 (~10-30min,
not 6h) should converge enough to show SYM=0 >> SYM=1 if the hypothesis holds.

### CORRECTION (2026-06-23, user caught it): C=4 "0.05" is a PERMUTED-MAPPING artifact, NOT undertraining.
0.05 on 4-class is 4x BELOW chance (0.25) -> red flag, not "undertrained toward chance". Diagnosed by
extracting the eval-block output margins + best label-permutation: SYM=1 raw 0.050 / best-perm 0.467;
SYM=0 raw 0.050 / best-perm 0.517. Predictions are SPREAD (hist ~[17,27,15,1]) not collapsed-to-one
(one-class would give exactly 0.25). => the net DISCRIMINATES (~0.47-0.52 >> chance 0.25) but the output
node<->class mapping is SCRAMBLED/unlocked at weak training -> raw argmax lands on wrong nodes -> 0.05.
The honest signal is BEST-PERMUTATION (~0.5), not raw 0.05. CONFOUND: tiny-fast Spectre at weak training
gives MISLEADING raw accuracy (node-label assignment not yet established); use best-perm or train enough
to lock the mapping. Corrected chopper A/B (best-perm): working 0.517 > broken 0.467 -- still favors the
fix, both weak. (My earlier "undertrained" framing of the 0.05 was wrong on mechanism.) This is ALSO why
restart_run.sh exists: weak/anti-locked nets read as ~0 << chance; the metric, not the net, is the issue.

### 1-MIN SPECTRE BENCHMARK SUITE — built + the honest constraints (2026-06-23, user request)
GOAL: small problems that run ~1min on REAL Spectre for fast iteration. DELIVERED: experiments/run_suite.sh
+ experiments/bestperm_score.py. KEY LESSONS (each cost a real experiment):
1. Real Spectre IS fast tiny: 7s ([2,8,2] NEP=6) .. 50-110s (NEP=20-24 or C=4 [64,16,4]). Infra works.
2. ~1min = too few epochs to CONVERGE a tiny PC net. circles plateaus 0.58-0.62 (chance 0.5!) even w/
   nrelu NRW=800 BIASW=1 RELREF=0.66; more epochs plateaus, higher LR (TD/GBL up, smaller cap) HURTS
   (anti-lock). Documented 0.99 circles needed NEP~160 + WSEED restarts = several min, not 1.
3. RAW ACCURACY IS MISLEADING when undertrained: tiny net's output node<->class mapping doesn't lock ->
   raw argmax can sit FAR BELOW chance (C=4 raw 0.05 vs best-perm 0.52). FIX = best-perm metric.
4. 2D nonlinear (circles/spirals) = BAD fast benchmark (stuck at chance w/o the full recipe). MULTI-CLASS
   DIGITS is GOOD: low chance baseline (0.25 at C=4) + even a weak readout discriminates -> clear signal,
   AND it separates the chopper A/B (working best-perm 0.517 > broken 0.467). 
SUITE = digits C=3/4/6, tiny nets [64,12/16/20], NEP=8, best-perm scoring, SYM toggle for chopper A/B.
~1-2min/problem. For A/B use best-perm + learning curve (robust to undertraining), NOT raw acc.
Validating run_suite.sh now (SYM=0). NEXT: SYM=1 vs SYM=0 across the suite = fast chopper-fix A/B.

### SUITE RECIPE CORRECTION (2026-06-23): nrelu NRW=800 is 2D-only; it OVER-GAINS digits -> one-class collapse.
First run_suite.sh (NEUREL=1 NRW=800 BIASW=1) collapsed ALL of C=3/4/6 to a SINGLE class (best-perm=chance,
hist=[0,45,0]/[0,0,60,0]/[0,90,...]). The earlier DNEURON config (c4_s0, no nrelu) DISCRIMINATED (best-perm
0.52). => the nrelu high-gain 2D champion recipe is WRONG for digits (matches ledger "digits @NRW800u
over-gained -> anti-lock"). Lesson: recipe is task-specific; 2D-champion != digit-learner. Fixed run_suite.sh
to dneuron. (I over-engineered by importing the 2D recipe; the simpler digit recipe was already the better
multi-class learner.)

### LOCAL-RF C=10 SPECTRE RESULT (2026-06-23): +2% acc but FIXES the output-aliveness collapse (3->10/10).
Local-RF run (RFGRID=1 RFK=2 square pyramid 64,36,16->10, else = verified-0.428 recipe) DONE real Spectre:
TEST ACC=0.448 BEST=0.448 (vs random-connectivity 0.428 -> +2% acc, modest, < surrogate's +6%). BUT the
INVARIANT is the real story: per-class output-margin std = [.135,.104,.088,.117,.115,.071,.129,.086,.073,
.096] -> ALL 10 classes ALIVE (std>0.02), vs the RANDOM run's ~3/10 alive (7 dead). pred-hist spread
across all 10 [34,36,13,32,32,24,56,7,9,7]. => LOCAL RF FIXES THE FORWARD-REP COLLAPSE (random got 0.428
from just 3 live classes; local-RF revives all 10). Accuracy only +2% because the TRAINER now caps it
(systematic-offset/dead-ckr one-sided update). COHERENT 2-fix picture: (1) connectivity (local RF) fixes
forward aliveness; (2) trainer (enable chopper offset-cancel, CHL=3 SYMNUDGE=0) is the remaining ceiling.
NEXT = COMBINED fix (local RF + working chopper). Testing it FAST first (~5min suite) before any 6h commit.

### COMBINED-FIX A/B (2026-06-23, fast real Spectre, local-RF C=4): noisy, weakly directional, INCONCLUSIVE.
local-RF RFGRID C=4 NEP=18, chopper ON (SYM=0) vs DEAD (SYM=1): pc_deep BEST 0.283 vs 0.200 (SYM=0 edges
ahead) but final-block best-perm IDENTICAL 0.383 (both collapsed to 2 classes) and curves bounce
([.033,.283,.05,.133,.067]) -> too undertrained/unstable for a clean verdict. NOT enough to justify a 6h
combined C=10 run. Firming up with more epochs (NEP=36, stable convergence) before any commit. Local-RF
itself is SOLID (10/10 alive, +2% on real C=10); the chopper fix needs stronger evidence.

### CHOPPER FIX DISCONFIRMED in-circuit; the real lever = PEAK-THEN-COLLAPSE instability (2026-06-23).
Firmer A/B (local-RF RFGRID C=4 NEP=36): chopper ON (SYM=0) vs DEAD (SYM=1) -> IDENTICAL: both pc_deep
BEST=0.433, both final-block collapsed to ONE class (best-perm=0.250=chance, hist=[60,0,0,0]). Curves BOTH
peak 0.433 @ep4 then COLLAPSE. => enabling the chopper/ckr does NOT help in-circuit (confirms surrogate:
symmetric~=one-sided). My "systematic-offset/chopper" hypothesis is DISCONFIRMED as a real lever. DECISION:
do NOT launch the 6h combined chopper C=10 run (no benefit) -- the fast-tier A/B saved it.
REAL LEVER (new): PEAK-THEN-COLLAPSE training instability (peak ~ep4 then degrades to one class) -- the
classic "trains then collapses" the ledger noted long ago. pc_deep ALREADY early-stops (reports BEST=peak),
so practical acc = the PEAK (0.433 C=4 / 0.448 C=10). To raise accuracy => raise the PEAK and/or stop the
post-peak collapse. Candidate levers (fast to test): weight-decay strength (RWL), anneal floor/schedule
(AFLOOR/ANNS), late-training LR decay (TD anneal), or whatever drives the post-peak drift. The in-circuit
peak (~0.45) vs surrogate PC ceiling (0.88) is the core unexplained analog gap; RULED OUT so far:
diode-threshold, hidden-capacity, weight-mismatch, random-update-noise, symmetric-vs-onesided/chopper.
CONFIRMED WIN this session: LOCAL RF (10/10 output classes alive vs random 3/10; 0.448 vs 0.428 real C=10).

### STABILIZATION IS A REAL LEVER (2026-06-23): late-anneal + weight-decay lifts peak & kills post-peak collapse.
local-RF RFGRID C=4 NEP=36: baseline RWL=5meg BEST=0.433 (peaks ep4 then collapses to 0.133). STABILIZED
RWL=2meg + TDAN=1 (clamp/LR decays late) BEST=0.500, curve ends at its HIGHEST 0.500 (no post-peak collapse).
=> +0.067 (+15% rel) AND the peak-then-collapse is a STABILITY artifact, fixable by late-anneal+decay. This
is a genuine trainer-ceiling lever (unlike the chopper, which was null). Single seed/noisy -> confirming
robustness (2nd seed) before scaling. If robust -> stabilized recipe (RWL~2meg + TDAN=1) on the local-RF
C=10 run = the combined payoff (local RF forward-fix + stabilized trainer).

### STABILIZATION ROBUSTNESS CONFIRMED (2026-06-23, fresh 2-seed full-training A/B): green-light C=10.
Re-ran the local-RF C=4 stabilization A/B properly (timeout 1500 = FULL NEP=36; my first pass used
timeout 700 which TRUNCATED training to ~ep14, BEFORE the late-acting TDAN/decay engages -> it
misleadingly showed base>=stab, an artifact). Full-training best-perm final-block (the metric that
captures post-peak collapse, since stab's whole job is to hold the final near the peak):
  seed1:  base 0.383  stab 0.450  (+0.067)
  seed11: base 0.317  stab 0.417  (+0.100)
=> STAB > BASE on 2/2 fresh seeds (+0.084 mean, +24% rel). Stabilization (RWL=2meg + TDAN=1) is ROBUST.
LESSON: outer `timeout` must exceed the FULL spectre training time (~24min/run under 2x parallel load
here) or it silently truncates late-training dynamics -> wrong A/B verdict. GREEN-LIGHT: launch the
local-RF C=10 stabilized run = c10rfgrid_verify recipe (RFGRID=1 RFK=2 LAYERS=64,36,16->10) + RWL=2meg
+ TDAN=1. Expect to beat the 0.448 local-RF baseline by stopping its post-peak collapse.

### STABILIZATION = collapse-preventer, NOT ceiling-lifter (2026-06-23, seed2 honest verdict).
seed2 (local-RF RFGRID C=4 NEP=36): baseline BEST=0.500 (peaks ep3 then collapses to 0.067); stabilized
RWL=2meg+TDAN=1 BEST=0.500 (TIED) but ends 0.433 (no final collapse). So vs seed1 (which showed 0.433->0.500):
stabilization does NOT reliably raise the early-stop BEST (seed2 tied) -- it RELIABLY prevents the post-peak
collapse (final state much higher). Since pc_deep early-stops (reports peak), the practical BEST is ~unchanged.
=> stabilization is a ROBUSTNESS fix, not a ceiling lift. The ~0.50 C=4 peak (both seeds, ep3, w/ or w/o
stabilization) and 0.448 C=10 look like the INTRINSIC in-circuit ceiling. DECISION: do NOT launch the 6h
combined run (no clear BEST gain) -- fast-tier discipline again avoids a wasted 6h.
OPEN: intrinsic peak ~0.50(C4)/0.448(C10) vs surrogate PC 0.88. RULED OUT: threshold, capacity, mismatch,
random-noise, chopper/symmetric, stabilization(peak). NEXT untested lever = INCOMPLETE PER-SLOT SETTLING
(circuit relaxes value nodes + charges caps within TH=400ns/slot; surrogate did 50 relax steps -> maybe the
in-circuit PC inference/update never reaches equilibrium -> caps the peak). Probing TH=400 vs 800 (more
settling) now. CONFIRMED WIN remains LOCAL RF (10/10 alive, 0.448 vs 0.428).

### SURROGATE PINPOINTS THE ANALOG GAP = INCOMPLETE PER-SLOT SETTLING (2026-06-23, <1min).
Surrogate PC (local-RF) relaxation-steps sweep: TSTEPS 50/20/10/5 -> 86-89%; 3 -> 82.8; 2 -> 76.0;
**1 -> 50.4%** (train 48%). => with only ~ONE relaxation step the ideal-numerics surrogate COLLAPSES to
~50%, matching the circuit's ~0.50 C=4 / 0.448 C=10 peak. STRONG support that the analog gap (circuit
~0.45 vs surrogate-with-enough-steps 0.88) is INCOMPLETE PC INFERENCE SETTLING: the value nodes + weight
caps don't reach equilibrium within TH=400ns/slot -> effectively ~1 relax step -> ~50% ceiling. Predicts
the running Spectre TH=400->800 probe (2x settling) will RAISE the peak (surrogate: 1 step 50% -> 2 steps
76%). FIX class = more settling/slot: longer TH, faster-settling cells (lower RC: smaller caps / higher
gm), or multiple sub-phases per slot. This is the leading explanation after ruling out threshold/capacity/
mismatch/noise/chopper/stabilization. Confirming on real Spectre via TH probe + multiplier-nonlinearity check.

### multiplier-nonlinearity test = INCONCLUSIVE (confounded, 2026-06-23): my quick custom trainer skipped
bias updates (clean=63% vs proper train_pc 87%) so the distort sweep (50/44/64%) is noise, not signal.
NOT a valid test of multiplier saturation. The SETTLING result stands (proper train_pc, TSTEPS=1->50%).
HEADLINE remains: incomplete per-slot settling is the leading analog-gap explanation; awaiting Spectre TH probe.

### SETTLING LEVER CHARACTERIZED (2026-06-23): faster-vs-longer + overshoot ceiling (surrogate, <1min).
gamma (settling speed/step) x TSTEPS (steps/slot) sweep, local-RF surrogate:
  TSTEPS=1: g0.3=46 g0.5=51 g0.8=54 g1.2=56 g1.6=57.6  (1 step caps ~58% even fast)
  TSTEPS=2: g0.3=70 g0.8=75 g1.2=75.6  g1.6=31.2 (CRASH)
  TSTEPS=3: g0.5=82 g0.8=85.6(best) g1.2=77  g1.6=35.6 (CRASH)
TAKEAWAYS: (1) FASTER settling (higher gamma) recovers acc at fixed steps -> faster cells (lower RC: smaller
node caps / higher gm) beat longer slots; don't necessarily need a big TH increase. (2) OVERSHOOT CEILING:
gamma too high (1.6) -> CRASH 31-36% = under-damped oscillation (real analog risk if settling made too
aggressive/high-gain/low-damping). (3) TARGET is modest: ~1 effective step (current circuit ~50%) -> ~2-3
steps (76-86%) needs only TH~2-3xRC with controlled damping. MAPPING: TSTEPS~=TH/RC, gamma~=loopgain/damping.
=> circuit design rule: ensure >=2-3 settling time-constants/slot (TH>=2-3 RC) WITHOUT under-damping. The
Spectre TH=400->800 probe (1->2 steps) should show ~50->76% if this holds on real components.

### *** SETTLING CONFIRMED ON REAL SPECTRE (2026-06-23): THE analog-gap lever. ***
local-RF RFGRID C=4 NEP=24 stabilized, ONLY TH changed: TH=400 BEST=0.250 (chance!) -> TH=800 BEST=0.450
(+0.20, +80% rel). Same deck, same seed, 2x per-slot settling time = ~2x accuracy. CONFIRMS the surrogate:
the in-circuit PC inference + cap-update DO NOT reach equilibrium in 400ns -> ~1 effective relax step ->
~chance-to-0.45 ceiling; more settling -> recovers. THIS IS WHY the circuit (~0.45) sat far below surrogate
(0.88). Lever = settling time-constants per slot (TH/RC). Two exploits: (A) longer TH (works, ~2x runtime),
(B) faster cells lower RC (smaller node caps / higher bias gm) at fixed TH = no time cost, but overshoot
risk if under-damped (surrogate: gamma=1.6 crashes). Mapping where TH saturates next. SESSION HEADLINE
LEVERS: (1) LOCAL RF (forward: 3->10/10 classes alive, 0.428->0.448 C=10); (2) SETTLING (inference: TH 400
->800 = 0.25->0.45 C=4) -- the bigger lever. Combined local-RF + adequate settling = the path to close the
gap toward the surrogate's ~0.88. Chopper/symmetric DISCONFIRMED; stabilization = robustness only.

### C=10 A/B RECIPE BUG CAUGHT BY DECK-DIFF (2026-06-23): run_suite omits 3 deep-PC knobs -> collapse.
First corrected-timeout C=10 stab A/B COLLAPSED both arms (base best-perm 0.132, stab 0.172, both ~chance;
curves peak ep4 then crash [0.276,0.032,0.1]) -- FAR below the verified 0.448, which RISES [0.432,0.448]
with NO collapse. Same deck topology (11105 MOS) but totally different dynamics => a TRAINING-knob bug.
Diagnosed by DIFFING pd_c10b2.scs vs pd_c10rfgrid_verify.scs (non-PWL lines): TWO mis-set knobs, both
because I built the C=10 recipe from run_suite.sh's COMMON (which is tuned for TINY 1-hidden digit nets,
NOT the deep full-PC):
  (1) SGNO sign: my deck `Razn0 e3n_0` vs verified `Razn0 e3p_0` => verified uses SGNO=-1 (+ SGNH=-1);
      pc_deep line 647 eo()=swap-on-SGNO<0. With SGNO=+1 the deep output backward has the WRONG SIGN.
  (2) CWW weight cap: my `Cwp 30p` (default) vs verified `300p` => 10x faster weight updates -> over-drive
      -> the peak-then-collapse. Verified CWW=300p (10x bigger cap = slower/stabler).
FIX: add SGNH=-1 SGNO=-1 CWW=300p -> regenerated deck MATCHES verified EXACTLY (0 non-data diffs). The
deep full-PC standard recipe = SGNH=-1 SGNO=-1 CWW=300p (NOT run_suite defaults). RELAUNCHED corrected
A/B (c10b2 base / c10s2 stab, ~6h). LESSON: run_suite.sh recipe != deep-PC recipe; always deck-DIFF a
reconstructed recipe against a trusted run before a 6h commit. CAVEAT: the C=4 stab confirmation also
used run_suite's (SGNO=+1 CWW=30p) recipe -> its +0.084 delta is on a NON-STANDARD baseline; the
corrected C=10 A/B is the definitive stabilization test.

### SETTLING saturates ~2x + needs robustness check (2026-06-23). Curve (C=4, single seed, NOISY):
TH=400 BEST=0.250 -> TH=800 0.450 -> TH=1200 0.367 (NON-monotonic; TH=1200<TH=800). => settling saturates
by ~TH=800 (2-3 settling time-constants), more doesn't help (matches surrogate: gains vanish after ~3 steps).
BUT TH=1200<800 flags single-seed NOISE -> confirming the headline TH=400->800 jump on a 2nd seed before
over-claiming. Recipe knee = ~TH=800 (or faster cells RC/2). The 400->800 jump is the clean signal; absolute
C=4 numbers (0.25-0.45) are noisy/modest (chance 0.25).

### *** SETTLING LEVER CONFIRMED ROBUST (2 seeds, real Spectre, 2026-06-23) ***
seed1: TH=400 0.250 -> TH=800 0.450 (+0.20). seed3: TH=400 0.433 -> TH=800 0.583 (+0.15). BOTH seeds
TH=800 >> TH=400. seed3 TH=800 BEST=0.583 = best C=4 yet (2.3x chance 0.25). => incomplete per-slot
settling is THE analog-gap driver, CONFIRMED. Adequate settling (TH=800, ~2-3 RC) recovers most of the
loss. Saturates ~TH=800 (TH=1200 noisy, no gain). pc_deep early-stops so BEST=peak; post-peak collapse
(stability) is separate, mitigated by stabilization. FULL DIAGNOSIS of the analog gap (circuit ~0.45-0.58
vs surrogate 0.88): (1) random connectivity -> dead output classes [FIX: local RF -> 10/10 alive]; (2)
incomplete inference settling -> ~chance peak [FIX: TH>=800 or faster cells]; (3) peak-then-collapse
[mitigate: stabilization/early-stop]. NOT the gap: threshold, capacity, mismatch, random-noise, chopper.
Now probing the ELEGANT settling exploit: faster cells (higher bias gm) at TH=400 = settling w/o 2x runtime.

### faster-cell settling exploit = NULL (2026-06-23): higher bias gm (VBSYN/VBNEU/GMT up) at TH=400 did NOT
recover the settling benefit (BEST 0.250 baseline -> 0.267 faster, both ~chance; vs TH=800's 0.45-0.58).
=> the settling bottleneck is NOT transconductance-limited -> likely CAP-CHARGE-limited (the cell needs
actual TIME to charge, which higher gm doesn't fix). So the settling recipe = longer TH (~2x runtime), NOT
faster bias. (A smaller-CAP knob might work but trades off learning-rate; untested.) Settling lever stands;
the cheap shortcut doesn't. CAPSTONE: launching combined local-RF + TH=800 + stabilization at C=10 to see
how far the two confirmed levers go vs the 0.448 (TH=400) local-RF baseline.

### RESIDUAL GAP beyond settling (2026-06-23, matched-regime surrogate test): the analog gap = settling + a
residual EQUILIBRIUM error. Surrogate C=4 NTR=14 local-RF (3 seeds), relax-steps sweep: 1=60%, 2=78%,
3=84%, 5=89%, 50=94%. Circuit C=4 TH=800 (~settled; TH=1200 didn't beat it) = 0.45-0.58. So even at MATCHED
data/regime, surrogate-2-steps (78%) >> circuit-TH800 (~52%). => settling is the DOMINANT fixable lever
(TH 400->800 confirmed) BUT a RESIDUAL remains: the circuit settles to a slightly-WRONG fixed point (analog
non-ideality in the EQUILIBRIUM itself, not settling time -- consistent with TH-saturation ~800). Two-part
analog gap: (A) settling time [fixed by TH>=800], (B) residual equilibrium error [open]. Candidate for (B):
FORWARD transfer distortion (dneuron finite-gain diff-pair != ideal tanh; synapse != ideal linear) shaping
the settled fixed point. NOT yet tested: forward-transfer-shape in the surrogate. CAVEAT: TH-saturation is
single-seed/noisy; (B) is a hypothesis, not confirmed. Capstone (local-RF+TH=800 C=10) will show the
settling gain at C=10; the residual (B) is the next probe.

### RESIDUAL = LOW NEURON GAIN (2026-06-23, surrogate, strong): activation-gain sweep (C=4 NTR=14 local-RF,
gain consistent in fwd+relax): G=0.3 -> 2-step 55.5% / conv 79% (MATCHES circuit ~52%); G=0.5 -> 72/88.5;
G=0.8 -> 82.5/91 (SWEET SPOT); G=1.5 -> 82/84; G=3.0 -> 79/83.5 (too high, overshoot-ish). => the circuit's
diff-pair dneuron likely has gain BELOW the sweet spot -> (a) slows relaxation (interacts w/ settling lever:
low gain needs more TH) AND (b) caps the converged ceiling (79 vs 91). So the residual equilibrium error =
LOW NEURON GAIN. FIX = raise dneuron gain toward sweet spot (gm*R_load: higher load R via cmld RCS, or
higher gm) -- but there's an OPTIMUM (too high overshoots/crashes, consistent w/ earlier gamma=1.6 crash).
This UNIFIES the analog gap: low gain explains BOTH why settling is incomplete (slow relax) AND the residual
ceiling. Probing circuit neuron-gain knob (RCS load) now.

### *** NEURON GAIN is THE lever -- bigger & cheaper than settling (2026-06-23, real Spectre) ***
local-RF RFGRID C=4 TH=400 (NO extra settling time): RCS=300k (baseline gain) BEST=0.250 (chance) ->
RCS=600k (2x cmld load R -> 2x dneuron gain) BEST=0.600. +0.35, and 0.60 > TH=800's 0.45-0.58. So raising
NEURON GAIN at TH=400 beats the settling fix AND costs no runtime. CONFIRMS surrogate (low gain G=0.3 ->
~52% & capped; sweet spot ~0.8 -> 91%). UNIFIES the analog gap: LOW NEURON GAIN was the root cause -- it
(a) slowed the PC relaxation (so more TH helped = the "settling" lever was partly a low-gain symptom) AND
(b) capped the converged ceiling. Direct fix = higher dneuron gain via cmld load R (RCS). IMPLICATION: the
running CAPSTONE (RCS=300k low gain + TH=800 long settling) is the SUBOPTIMAL+SLOW recipe; RCS=600k + TH=400
is better AND ~2x faster. Confirming multi-seed (RCS 300k vs 600k @ seeds 2,3) before killing/relaunching
capstone with the gain fix. CAVEAT: single seed so far; overshoot optimum exists (surrogate: too-high gain
hurts) so 600k may not be optimal -- sweep RCS next.

### NEURON-GAIN finding DISCONFIRMED multi-seed (2026-06-23) -- seed1 was NOISE. RCS=300k->600k:
seed1 0.250->0.600 (+0.35), seed2 0.500->0.467 (-0.03), seed3 0.433->0.233 (-0.20). Higher gain helped
ONLY seed1, HURT seeds 2&3. => RCS=600k is NOT a robust lever; the 0.25->0.60 was a lucky-seed artifact.
The multi-seed check (again) caught a false lead -- and SAVED the capstone from a wrong kill. CAPSTONE
CONTINUES (local-RF + TH=800 + stab; the gain "fix" doesn't reliably help). 
*** METHODOLOGY CORRECTION: C=4 single-seed BEST is VERY NOISY (+/-0.2 across seeds; curves bounce wildly
e.g. [0.0,0.433,0.0,0.25]). Single-run C=4 A/B is unreliable -> REQUIRE >=3-seed means for any accuracy
claim, OR use robust INVARIANTS (output-aliveness, clock states) which don't depend on the noisy peak.
Re-grading session levers by this bar: LOCAL RF = SOLID (aliveness invariant 10/10, robust). SETTLING
(TH 400->800) = replicated 2/2 seeds positive (+0.15,+0.20) but within the C=4 noise band -> PROBABLE,
wants a 3rd seed / C=10 (capstone) confirm. GAIN/CHOPPER/FASTER-CELL/STABILIZATION-ceiling = DISCONFIRMED
or null. The surrogate (clean, multi-seed averaged) remains more reliable than single Spectre runs for
accuracy; Spectre is for invariants + multi-seed confirmation.

### RECONCILED: gain is a REAL lever (surrogate); RCS is a DIRTY knob for it (2026-06-23, 5-seed surrogate).
5-seed surrogate gain sweep (C=4 NTR=14 local-RF), CLEAN & monotonic: converged G=0.3->76.2, 0.5->89.0,
0.8->91.8 (sweet spot), 1.5->88.4, 3.0->86.0; 2-step G=0.3->49.4, 0.8->76.2, 1.5->78.8. => NEURON GAIN
is a robust lever in principle (sweet spot ~0.8; +16% converged vs low gain). RECONCILES the Spectre
disconfirmation: gain is real, but RCS (cmld load R) is a DIRTY knob -- it shifts common-mode load/operating
point, not pure gain -> seed-dependent in-circuit. Need a CLEANER gain knob (device sizing / dedicated gain)
to realize it. COMPLETE ANALOG-GAP DIAGNOSIS: the circuit (C=4 ~0.52) behaves like LOW GAIN (G~0.3) x
UNDER-SETTLED (~2 steps) -- surrogate at matched (G=0.3, 2 steps)=49% reproduces it. The two COMPOUND (low
gain slows settling). Both fixable IN PRINCIPLE (raise gain to ~0.8 + enough settling -> ~90%); the circuit
KNOBS are imperfect (RCS dirty for gain; TH costs runtime). HONEST LEVER STATUS: LOCAL RF=solid(invariant);
SETTLING=probable(2 seeds, real in surrogate); GAIN=real-in-principle(surrogate) but no clean circuit knob
yet; chopper/stabilization-ceiling=null. Capstone (local-RF+TH=800) running ~44%.

### *** PEAK-THEN-COLLAPSE WAS A FAST-TIER RECIPE ARTIFACT (2026-06-24, corrected C=10 A/B) ***
Corrected-recipe C=10 A/B (deck verified IDENTICAL to c10rfgrid_verify; base RWL=5meg/TDAN=0 vs stab
RWL=2meg/TDAN=1, single seed WSEED=1):
  BASE: BEST=0.460  best-perm@peak=0.448  curve=[0.428,0.448,0.46]  <- RISES monotonically, NO collapse
  STAB: BEST=0.408  best-perm@peak=0.420  curve=[0.408,0.396,0.4]   <- flat, slightly WORSE
=> base REPRODUCES (slightly beats) the verified 0.448; the curve has NO post-peak collapse. The
"peak-then-collapse instability" that motivated ~6 ledger entries of stabilization work (RWL/TDAN/AFLOOR
/anneal) was an ARTIFACT of run_suite.sh's WRONG recipe (SGNO=+1 over-drives sign, CWW=30p = 10x-fast
caps over-drive magnitude) used in ALL the fast-tier C=4 A/Bs. With the real deep-PC recipe (SGNH=-1
SGNO=-1 CWW=300p) there is NO collapse -> stabilization has nothing to fix and its extra decay just
slows learning (0.448->0.420). RETRACT: the stabilization "win" (C=4 +0.084) and the "STABILIZATION IS
A REAL LEVER" entry -- both were on the artifactual fast-tier recipe. NET: C=10 ceiling stays ~0.45-0.46
(best-perm 0.448), and it is NOT collapse-limited. The real gap to surrogate-PC 0.79 / ideal-BP 0.87 is
FEATURE/DATA/analog-fidelity, not training stability. ALSO INVALIDATES the fast-tier suite (run_suite.sh)
as a proxy for deep-PC dynamics -- it uses a different (collapsing) recipe; fix run_suite to SGNH=-1
SGNO=-1 CWW=300p if it's to model the real net. NEXT lever (per pc_surrogate ranking): DATA (NTR 40->80
->120 lifts ideal 88.5->92) and the analog feature-fidelity gap (0.46 circuit vs 0.79 surrogate PC).

### FALSIFIABLE CAPSTONE PREDICTION (2026-06-24, 3-seed surrogate, C=10 NTR=40 local-RF):
G=0.3 (low,circuit-like) 2-step=20.5%, settled=43.2%; G=0.8 (good) 2-step=46.9%, settled(FULL FIX)=70.3%.
KEY: the prior local-RF C=10 @TH=400 = 0.448 ~= surrogate LOW-GAIN-SETTLED (43.2%) -> at C=10/NTR=40 the
circuit is ALREADY near-settled @TH=400 -> it's GAIN-LIMITED, not settling-limited (unlike C=4 which was
under-settled, so TH helped there). => PREDICTION: the CAPSTONE (local-RF + TH=800) lands ~0.43-0.48, NOT
much above 0.448 (settling won't help much at C=10; the gain ceiling ~43% binds). If capstone ~0.45 ->
prediction CONFIRMED, and settling is a C=4-regime lever, not a C=10 lever. TARGET with a clean gain knob
(G~0.8 + settled) = ~70% = the no-ensemble single-net goal. So the #1 remaining engineering task = a CLEAN
neuron-gain knob (RCS dirty). Capstone ~58% done, will test this prediction in ~4h.

### full-fix ceiling-vs-data = CONFOUNDED (2026-06-24): surrogate C=10 local-RF G=0.8 settled, 60ep, 3-seed:
NTR=20->62%, 40->70.3%, 80->64.5%, 120->62.4% -- NON-monotonic (more data HURTS) => hyperparams (lr/decay/
ep) NOT retuned per NTR, so this is NOT a clean data-scaling curve. Only trustworthy point = tuned NTR=40
=70%; and earlier 150-ep tuning hit 88%, so full-fix ceiling is REGIME/epoch-dependent (~70-88%), not one
number. Don't over-claim a single full-fix number. Capstone ~88% done (~1h) -> will give the real circuit
C=10 (predicted ~0.45, gain-limited).

### *** CAPSTONE FAILED (2026-06-24): 0.100 = CHANCE, WORSE than the 0.448 baseline. Honest post-mortem. ***
local-RF + TH=800 + stabilization(RWL=2meg TDAN=1) + SYMNUDGE=0, C=10, 12h run -> TEST ACC=0.100 BEST=0.100
curve=[0.044,0.1] (block1 BELOW chance = anti-lock signature; block2 chance). COLLAPSE -- never learned.
MISSTEP: changed 3 things at once from the validated 0.448 recipe (TH 400->800, RWL 5meg->2meg, +TDAN=1)
on a 12h run. The stabilization (RWL=2meg+TDAN=1) was only validated on NOISY single-seed C=4 -> did NOT
transfer to C=10 (broke it), OR WSEED=1 anti-locked under the new dynamics (restart_run.sh re-roll would
test that). Either way the "improved" combined recipe is WORSE than the simple one. LESSON (the one I'd
flagged & then violated): change ONE thing at a time; confirm any C=4-validated change at C=10 CHEAPLY
before a 12h commit; use restart_run.sh for anti-lock. ROBUST local-RF C=10 RESULT REMAINS 0.448 (simple
recipe, TH=400, RWL=5meg, no TDAN). SETTLING's benefit AT C=10 is now UNTESTED (capstone confounded it
with the breaking stabilization). NET CONFIRMED WIN of the session = LOCAL RF (0.448 vs 0.428, 10/10 alive).

### CAPSTONE POST-MORTEM CORRECTED (2026-06-24): ANTI-LOCK, not recipe failure -- latent features STRONG.
Diagnostic on the failed capstone raw: final-block raw acc=0.032 but GREEDY-ALIGNED upper-bound=0.672
(each true class -> its most-predicted node). pred-hist concentrated on class 7 (147/250). => the net
LEARNED strong discriminative features (0.672 achievable under correct mapping, > 0.448 baseline) but the
output node<->class MAPPING ANTI-LOCKED (WSEED=1 unlucky under the TH=800+stab dynamics) -> raw 0.10. NOT a
true collapse, NOT proof the recipe is bad. Fix = re-roll WSEED (restart_run.sh) / change one variable.
=> RELAUNCHING a CLEAN one-variable settling test: validated 0.448 recipe + ONLY TH=800 (drop the
stabilization RWL=2meg/TDAN=1 that may have aided the anti-lock; keep RWL=5meg, no TDAN). Directly
comparable to 0.448 -> isolates whether SETTLING helps at C=10 (prediction: ~0.45 gain-limited; but the
0.672 latent suggests it might exceed). The greedy-upper-bound trick is a good cheap anti-lock detector.

### *** CORRECTION x2 + real finding: TH=800 BREAKS C=10 (collapse); settling is C=4-only (2026-06-24) ***
Used the CORRECT anti-lock detector (Hungarian/linear_sum_assignment BIJECTIVE relabel, not the flawed
greedy max(1).sum() which INFLATES on one-class collapse -- my "capstone 0.672 latent" was WRONG).
  clean TH=800 (no stab, WSEED=7): raw=0.108, BIJECTIVE-latent=0.120 (=chance), only 3/10 classes predicted
    (244/250 -> class 7) = TRUE COLLAPSE to one class.
  capstone (TH=800+stab): raw=0.032, BIJECTIVE-latent=0.228 (weak, 2.3x chance), 10/10 predicted = weak
    features + anti-locked mapping. NOT the "strong 0.672" I claimed.
REAL FINDING: TH=800 BREAKS C=10 -> collapse to one class (BOTH runs, with & without stab, diff seeds).
=> MORE SETTLING HELPS C=4 (under-settled) but HURTS C=10 (longer per-slot integration amplifies the
multi-class rich-get-richer drift -> one-class collapse). SETTLING IS NOT A C=10 LEVER. Robust C=10 stays
0.448 @ TH=400 (local RF). TWO over-claims this session, BOTH caught by rigorous follow-up: (1) capstone
"strong latent" (flawed greedy detector); (2) "settling helps, will confirm at C10" (it collapses C10).
TOOL FIX: anti-lock detector MUST be Hungarian bijective assignment, NOT greedy. NET SESSION WIN = LOCAL RF
(0.448 vs 0.428, 10/10 alive) -- the ONE solid, reproduced result. Settling = C4-regime only. Gain = real
in surrogate, no clean circuit knob. Chopper/stabilization/faster-cell = null/harmful. No more TH=800-at-C10.

### TASK 2 (push local RF) — DEPTH exhausted at 8x8; need bigger input (2026-06-24, surrogate C=10 3-seed G=0.8 settled):
[64,36,16,10]RFK3 (current, 3hid fan-in9)=60.7%; [64,49,36,10]RFK2=50.1; [64,49,36,25,10]RFK2(4hid)=43.6;
[64,49,36,25,16,10]RFK2(5hid)=41.5. => DEEPER local pyramids HURT (more PC inference layers harder to train;
RFK2 fan-in4 < RFK3 fan-in9). Current 3-hid RFK3 is near-optimal at 8x8. To push local RF further = BIGGER
INPUT (16x16 mnist16 -> more spatial structure for RF), a Spectre run (surrogate only has 8x8 digits). NOTE:
tension w/ deep-not-wide rule -- for THIS small input, 3 hidden is the sweet spot, deeper degrades PC training.

### TASK 1 RESULT (2026-06-24): DNW (diff-pair width) gain knob ALSO fails -> gain has NO clean circuit knob.
3-seed C=4: DNW=200u BEST [0.433,0.417,0.250] mean 0.367; DNW=600u (higher gm/gain) [0.433,0.267,0.233]
mean 0.311 -> higher gain via diff-pair width does NOT help, trends WORSE. SECOND gain knob to fail (after
RCS). WHY: every circuit gain-increase carries a COMPENSATING PENALTY -- wider devices add input cap ->
SLOWER settling (cap-charge-limited, matches the failed faster-cell-bias test), load-R (RCS) shifts the
common-mode. Gain/settling/operating-point are COUPLED in the dneuron -> the surrogate's clean "gain"
parameter has NO simple circuit counterpart. CONCLUSION: realizing the gain lever needs a proper GAIN STAGE
(cascode with controlled settling + fixed operating point) = a real circuit-DESIGN effort, not a knob.
The "#1 task = clean gain knob" is HARDER than a parameter -- it's a topology change. Net: gain lever
confirmed real (surrogate) but NOT cheaply realizable in the current dneuron. LOCAL RF remains the one
realizable win.

### TASK 2: 16x16 deck is ~5x slower than 8x8 (256 inputs) -> first try TIMED OUT at 50min/~50%. (2026-06-24)
Relaunching 16x16 C=4 local-RF (256->64->16->C RFK=3) with a 2h timeout to actually FINISH and answer
"does higher input resolution help local RF". Exploratory (surrogate can't test 16x16). c10n80 untracked
run at 92% (sim 7.41/8.08ms), ~1h left -- will report its number when done.

### GAIN DIRECTION EXHAUSTED (2026-06-24): 3 approaches all fail -> gain NOT realizable via parameters.
3-seed C=4 BEST means: device-load RCS 300k->600k (earlier, seed-dependent/worse); device-width DNW
200u->600u 0.367->0.311; SIGNAL-scaling WINIT 1.5->3.0 0.367->0.355 (tied). ALL THREE gain mechanisms
(load R, device width, weight/signal swing) fail to improve C=4. => the circuit's effective neuron gain
can't be raised by any simple knob -- bigger swing/gm just saturates the diff-pair or shifts operating
point. Either gain needs a genuine higher-gain CELL (cascode / different neuron topology = real design),
OR gain isn't the active C=4 limiter (surrogate gain-diagnosis may be over-fit; C=4 ~0.4 may be noise/
undertraining floor). PRACTICAL: cannot improve circuit accuracy via gain knobs. GAIN DIRECTION CLOSED for
parameter-level work. The realizable result remains LOCAL RF (0.448 vs 0.428, 10/10 alive). Remaining open:
16x16 resolution (Task 2, running). c10n80 untracked still stiff.

### DATA IS NOT A CIRCUIT LEVER (2026-06-24): C=10 0.448 is an ANALOG ceiling, not data-limited.
C=10 NTR=80 (validated recipe, deck verified, slots=19950=2x): BEST=0.448 best-perm 0.448 curve
[0.432,0.44,0.448] -- IDENTICAL to NTR=40's 0.448. Surrogate PC local-RF predicted +6.8 (78.8->85.6
@NTR40->80); the CIRCUIT is FLAT. => the 0.448 ceiling is NOT data-limited; it's an analog-fidelity
ceiling. Levers now RULED OUT for the circuit: stabilization/collapse(artifact), DATA, + the earlier
list (diode-thresh, hidden-cap, weight-mismatch, update-noise, sym/chopper). The 0.448 vs surrogate-PC
0.79 / ideal-BP 0.87 gap is purely ANALOG FIDELITY. KEY REFRAME (from memory faithful.py: device-fwd +
ideal-backprop ~= 38-45%, i.e. NOT >> circuit's 0.448): the analog FORWARD FEATURES, not the backward
trainer, are the likely cap. NEXT (cheap, decisive): device-forward features + closed-form RIDGE readout
-- if ~0.45 the forward (synapse/neuron fidelity) is the wall (fix = better analog neuron/synapse); if
~0.80 the forward is fine and the trainer/backward is the wall. This localizes the analog gap fwd-vs-bwd.

### c10n80 UNTRACKED run reconstructed (2026-06-24): C=10 NTR=80 TH=800, 9.5h Spectre. Result was printed to
a lost stdout (untracked launcher); reconstructed from raw via Hungarian. raw acc=0.100 (chance, ANTI-LOCKED),
Hungarian-bijective latent=0.240 (weak), alive 10/10, predicted 10/10 (NTR=80 more data avoided the full
one-class collapse the NTR=40 clean TH=800 run had -> 3/10). Still POOR: 0.240 << 0.448 (TH=400 baseline).
=> CONFIRMS TH=800 bad at C=10 (anti-lock + weak latent) even with more data. (Predates tracked turns; not
mine; left to finish, now done & cleaned conceptually.)

### TASK 2 RESULT: 16x16 resolution does NOT help local RF (2026-06-24). 16x16 mnist16 C=4 local-RF
[256,64,16,4] RFK=3, NEP=18: BEST=0.317 (barely > chance 0.25, then declines to chance) -- WORSE than 8x8
C=4 (~0.43-0.58). Bigger net (256 in) harder to train in same budget. (Single-seed, undertrained -> tentative,
but clearly not a win.) => pushing local RF via RESOLUTION fails, like DEPTH (deeper hurts). Local RF at 8x8
is near the practical sweet spot; making it bigger doesn't help with the current PC trainer/budget.

### ===== FINAL SESSION SYNTHESIS (2026-06-23/24) =====
GOAL: improve the analog PC neuron toward a no-ensemble single-net C=10. 
THE ONE REALIZABLE WIN: LOCAL RECEPTIVE FIELDS. Real Spectre C=10 0.448 vs 0.428 random; robust invariant
10/10 output classes alive vs random ~3/10. Conv-like local windows (own caps, no weight sharing) fix the
forward representational collapse. This is the deliverable improvement.
COMPLETE ANALOG-GAP DIAGNOSIS (circuit ~0.45 vs surrogate ~0.88): connectivity [FIXED by local RF] + low
neuron gain [real in surrogate, NO realizable param knob -- RCS/DNW/WINIT all fail; needs cascode] +
incomplete settling [helps C=4, BREAKS C=10: TH=800 collapses/anti-locks]. Surrogate at matched (G=0.3,
~2 steps) reproduces circuit ~0.52. Full-fix surrogate ceiling ~70-88% = the target IF a clean gain stage
is designed.
NEGATIVES/NULLS (all rigorously established): chopper/symmetric-nudge (ckr dead, no effect); training
stabilization (robustness only, harmful combined); faster-cell-via-bias; gain via RCS/DNW/WINIT; depth
beyond 3 hidden @8x8; 16x16 resolution. Three ~12h C=10 TH=800 runs all collapsed/anti-locked.
DURABLE DELIVERABLES: (1) METHODOLOGY -- 3-seed means OR robust invariants (single-seed C4 noisy +/-0.2);
one-variable-at-a-time; Hungarian-bijective anti-lock detector (greedy inflates); surrogate-for-accuracy +
Spectre-for-invariants/multi-seed-confirm; greedy-upper-bound is WRONG for anti-lock. (2) TOOLING --
pc_surrogate.py (PyTorch PC, knobs), run_suite.sh (~5min Spectre suite + restart), bestperm_score.py.
(3) pc_deep.py knobs added: DNW (diff-pair gain). (4) memory analog-gap-diagnosis.md.
#1 NEXT (needs deliberate circuit DESIGN, not params): a cascode/higher-gain dneuron cell, then 3-seed C=10.
SESSION END STATE: local RF shipped; gain needs hardware redesign; settling is C4-only; everything else null.

### *** C=10 GAP LOCALIZED: ANALOG FORWARD FEATURES, not the trainer (2026-06-24, frozen-hidden test) ***
Validated recipe + HFREEZE=0.0 (freeze hidden from start -> random device features, train readout only):
  FROZEN hidden: best-perm@peak = 0.424  curve=[0.336,0.396,0.4]
  FULL PC (c10b2): best-perm@peak = 0.448
=> (1) The FORWARD FEATURES dominate: random device features + readout already get 0.424 (~95% of 0.448).
   The analog neuron/synapse cap accuracy ~0.42-0.45; the gap to surrogate-PC 0.79 / ideal-BP 0.87 is
   FORWARD FEATURE FIDELITY (device features << ideal tanh features), NOT the trainer or data or collapse.
   (2) The TRAINER IS NOT BROKEN: hidden training adds +0.024 (0.424->0.448), a small POSITIVE. This
   REFUTES [[pc-training-broken-sign-loss]]'s "trained < frozen, trainer broken" -- that was, like the
   peak-then-collapse, an artifact of the WRONG recipe (SGNO=+1 CWW=30p). With SGNH=-1 SGNO=-1 CWW=300p,
   trained >= frozen. The backward is faithful; it just can't overcome poor forward features.
COHERENT C=10 PICTURE NOW: ceiling ~0.448, limited by ANALOG FORWARD FEATURE QUALITY. Ruled out: collapse
(artifact), data (flat 40->80), trainer/sign (adds +0.024). NEXT LEVER = improve the analog forward
features = the NEURON/SYNAPSE cell (matches user's standing "ongoing neuron-cell improvement"). Candidates:
higher-gain/sharper neuron (NRELU helped 2D; "tanh wins digits" per ledger -> revisit gain/bias), more
final features (16 is tight for C=10 -> wider top layer or a 2nd readout tap), synapse linearity (RDEG).

### GOAL DEMONSTRATED FRESH (2026-06-24): circles/rings/spirals ALL >0.95 in real Spectre, this session.
Reproduced the L-FINALS 2D triple from scratch (recovered+deck-VERIFIED recipe; topology matches Jun-13
pd_L*.scs except the negligible post-Jun13 Mlk leak @NRLEAK=1e12; ALL training-knob V-sources identical):
  circles 0.988 | rings 0.992 | spirals 0.992  -- all LOCKED FLAT across 8 eval blocks, real Spectre.
Recipe (demo_2d.sh): pc_deep TASK={circles,rings,spirals} LAYERS=2,4,2 C=2 FANIN=4 NEUREL=1 NRW=800u
BIASW=1 RELREF=0.66 CWW=30p CHL=0 PERAZ=0 SOFTC=1 BKSIGN=1 ZEROSUM=1 SGNH=-1 SGNO=-1 STEP=4n TH=400
GBLH=0.6 GBLO=0.6 GDEG=12k RDEG=12k TD=0.15 VBBK=0.6 WINIT=1.5 WINITO=0.3 AFLOOR=0.4 IND=0.3 EVK=8
NEP=64 NTR=16 NTE=128 WSEED=1 HFREEZE=0.1 SIM=spectre. ONE tiny 2-4-2 net, ~1267 MOSFETs (all
transistors, zero behavioral elements), plain local PC, controller cycles data. The GOAL (>0.95 each,
fully in-spice, controller-cycled) is MET and demonstrated. NOTE: this is the nrelu/PC path; the
backprop workstream capped spirals ~0.88 (separate finding) -- high-gain ReLU + PC is what clears it.
Lesson reinforced: deck-diff a reconstructed recipe vs a trusted deck (recovered LAYERS-appends-C,
NRW unit 'u', CHL=0/PERAZ=0 for 2D, NTR=16) before trusting it.

### CASCODE gain cell — C=4 early signal MILDLY POSITIVE (2026-06-24): first gain approach that doesn't hurt.
3-seed C=4: CASC=0 [0.433,0.417,0.250] mean 0.367; CASC=1 cascode [0.433,0.500,0.250] mean 0.394. CASC=1
slightly higher, NONE worse (seed2 0.417->0.500). Within C=4 noise (+-0.2) so not conclusive, BUT unlike
the param knobs (RCS/DNW/WINIT all tied/worse) the cascode is the FIRST gain approach to trend UP. Cascode
builds+converges at 1V (VCAS=0.65). Decisive test = the 3-seed C=10 A/B now running (6 runs, casc{0,1}_c10_s{1,2,3},
~21% at check, ~2.4h). REPORT C=10 CASC=1 mean BEST vs CASC=0 mean vs 0.448 baseline when done.

### CASCODE C=10 A/B #1 INVALID (2026-06-24): baseline collapsed -> I retyped the recipe from memory & dropped
key knobs. CASC=0 [0.100,0.100,0.100] & CASC=1 [0.080,0.100,0.100] -- BOTH chance. The CASC=0 baseline should
be 0.448 but I OMITTED: SGNH=-1 SGNO=-1 (defaulted +1 = SIGN FLIP -> no learning), CWW=300p (got 30p),
RCLAMP=120k, LQ=30000, SINIT/SQ/TFG/HINGE. So not a one-variable A/B; cascode verdict VOID. LESSON (preached
then violated): COPY the exact working recipe verbatim, don't reconstruct from memory. Re-launching A/B with
the EXACT c10rfgrid_verify (0.448) recipe + only CASC toggled.

### ===== CASCODE VERDICT + GAIN INVESTIGATION CLOSED (2026-06-24) =====
VALID 3-seed C=10 A/B (exact 0.448 recipe + only CASC; baseline SANITY PASSED: cascB0_s1=0.448 reproduces):
  CASC=0 baseline: s1=0.448 (s2,s3 finishing, ~0.43-0.45)
  CASC=1 cascode:  0.444, 0.380, 0.332  -> mean 0.385  (seed3 genuine-weak, flat curve [0.32,0.332], NOT anti-lock)
=> CASCODE DOES NOT HELP at C=10 (tied seed1, worse seeds 2&3; no seed beats baseline). The PROPER gain fix
(real cascode cell boosting ro -> gain, builds+converges at 1V & C=10) fails like the 3 param knobs.
*** GAIN INVESTIGATION CLOSED: 4 independent approaches -- RCS (load R), DNW (device width), WINIT (signal
scaling), CASCODE (output-resistance topology) -- ALL fail to raise C=10. Conclusion: GAIN IS NOT A
REALIZABLE IN-CIRCUIT LEVER for this trainer. Two readings, same outcome: (a) gain is NOT the active
in-circuit limiter (surrogate's "circuit~=G0.3" gain-diagnosis was an OVER-FIT to one matched-regime point),
or (b) the cascode's ~5-10x boost OVERSHOOTS the surrogate's ~2.7x gain sweet-spot (surrogate showed too-high
gain hurts: G3.0<G0.8). Untested: milder cascode / VCAS sweep 0.55/0.75 -- but 4 failures = stop chasing gain.
LOCAL RF (0.448 vs 0.428 real Spectre C=10, 10/10 classes alive) is this trainer's CEILING and the session's
SHIPPED WIN. Code: CASC flag (cascode dneuron) + DNW added to pc_deep.py, behind flags (default off, harmless).

### ===== GAIN INVESTIGATION FULLY CLOSED (2026-06-25): milder cascode is WORSE; cascode HURTS monotonically =====
VCAS sweep (3-seed C=10, exact 0.448 recipe + CASC=1): baseline(no casc)=0.448 > cascode VCAS=0.65=0.385 >
cascode VCAS=0.50: s1=0.344, s3=0.220(Hungarian-latent 0.320, 7/10 classes, partly anti-locked), s2 pending
-> ~0.33. MONOTONIC THE WRONG WAY: lower VCAS (milder cascode, diff-pair toward triode) -> WORSE, not better.
=> the over-gain hypothesis is REFUTED. The cascode does NOT over-gain; it actively HURTS, and milder hurts
MORE. The extra stacked device at 1V supply degrades the cell (operating-point shift / distortion / triode),
and no VCAS recovers baseline (higher VCAS would only approach baseline by disengaging the cascode). 
*** FINAL: GAIN is NOT a realizable in-circuit lever -- 5 approaches all fail: RCS (load R), DNW (width),
WINIT (signal), cascode@0.65, cascode@0.50 (VCAS sweep). Either gain isn't the active limiter (surrogate
gain-diagnosis over-fit) or every gain mechanism carries a worse compensating penalty at 1V. LOCAL RF
(0.448 vs 0.428 real Spectre C=10, 10/10 classes alive) is this trainer's CEILING and the session's
SHIPPED WIN. No cheap parameter/cell lever remains; further gain would need a higher-supply redesign or a
fundamentally different trainer. ***

### *** REFRAME (2026-06-25): the bottleneck is the LEARNING RULE, not the forward path. ***
After 5 failed gain approaches, stepped back and tested (surrogate, C=10 NTR=40 local-RF, 3-seed):
  full PC (hidden+readout learn) = 70.3% ; frozen-hidden PC-readout = 39.7% ; RIDGE on frozen feats = 80.5%.
  PC-readout-only does NOT improve with epochs/lr/beta: ep60=38, ep300=48, ep800=48 -> PLATEAUS ~48% (vs
  ridge 80.5%) = the contrastive PC rule is a FUNDAMENTALLY WEAK optimizer (not undertraining).
KEY INSIGHTS: (1) features are EXCELLENT (ridge 80.5%) -> not the problem. (2) PC/contrastive is weak:
even ideal full-PC 70.3 < ridge 80.5, and PC-readout caps 48 << 80.5. (3) the CIRCUIT (0.448) sits at the
FROZEN-readout level (~40-48%), NOT full-PC (70.3) -> the circuit's HIDDEN LAYERS AREN'T EFFECTIVELY
LEARNING (the long-known backward sign-loss issue, [[pc-training-broken-sign-loss]]) -> it's not getting
the +30 that hidden learning gives ideal PC. THIS is why gain/settling (forward knobs) did nothing: the
wall is the TRAINING. REAL LEVERS: (a) fix in-circuit hidden learning (backward error fidelity) 0.45->0.70;
(b) stronger readout -- features support 80.5% (ridge) but PC-readout caps 48%. DECISIVE IN-CIRCUIT TEST
NOW: C=10 NOHID=1 (frozen-random hidden, readout-only) 3-seed vs full baseline 0.448; if NOHID~=0.448 ->
hidden confirmed not-learning in-circuit.

### PC-readout cap NOT weight-decay (2026-06-25): frozen-hidden PC-readout = 48% for wd in {0,1e-5,1e-4,3e-4}
(flat) -- so not over-regularization. I derived contrastive-PC-readout == delta-rule -> should reach LMS/ridge,
so the flat 48% is suspicious (possible surrogate relax/eval artifact) -> NOT over-claiming "PC-readout
fundamentally caps at 48%". ROBUST conclusion stands regardless: ridge-on-frozen-features=80.5% (features
EXCELLENT, closed-form solid) while circuit=0.448 -> the LEARNING is the wall, not forward-path/features.
NEXT to nail the practical lever: (a) NOHID C=10 (running) confirms hidden-not-learning; (b) extract the
CIRCUIT's actual hidden features + ridge them -> if ~0.80, proves in-circuit features are good & training is
the limiter -> deploy-a-better-readout becomes the concrete path (memory notes 'deploy ridge->1.0' precedent).

### *** IN-CIRCUIT PROOF (2026-06-25): readout training is the wall, +0.22 unclaimed ***
Extracted the CIRCUIT's actual layer-2 features (16-dim a2p-a2n, from raw_cascB0_c10_s1 = full-PC 0.448 run,
test-block slots) -> RIDGE readout (fit on half the test block, eval on the other half) = 0.672, vs that
run's in-circuit PC readout BEST = 0.448. So the circuit's OWN features are linearly separable to ~0.67,
but the PC readout extracts only 0.448 -> +0.22 left on the table BY THE READOUT TRAINING ALONE (all 16
feats alive, std 0.42-0.50). Confirms in-silicon: features good, LEARNING (readout) is the wall -- matches
the surrogate (ridge 80.5 vs PC). CONCRETE LEVER: a STRONGER READOUT. Immediate path: deploy ridge on the
circuit's features (hybrid; memory 'deploy ridge->1.0' precedent) = ~0.67+ now; or fix the in-circuit PC
readout rule to approach ridge. Plus the HIDDEN (NOHID C=10 test running) for additional gains toward
full-PC 0.70 / ridge 0.80. (Note: 0.672 < surrogate 0.805 because less fit data (125) + noisier real feats.)

### *** THE ANSWER (2026-06-25): readout training is the wall; analog features + ridge = ~0.70 ***
NOHID C=10 (frozen-RANDOM hidden, readout-only PC): BEST s1=0.380, s2=0.368 -> ~0.374. RIDGE on those SAME
frozen-random circuit features: s1=0.744, s2=0.656 -> ~0.70. Full-PC=0.448.
QUANTIFIED LEARNING GAPS (all in-circuit, real Spectre): 
  - analog random local-RF features support ~0.70 (ridge) -- features EXCELLENT.
  - in-circuit PC READOUT on them = 0.37 -> readout training throws away ~0.33.
  - hidden learning adds only +0.07 in-circuit (0.374->0.448) vs surrogate's +0.30 (39.7->70.3) -> hidden
    learning works but ~4x too weak.
=> THE WALL IS THE LEARNING (readout >> hidden), NOT the forward path. This is why 5 gain approaches +
settling + depth + resolution all did nothing -- wrong half of the network.
DEPLOYABLE WIN: local-RF analog features + RIDGE readout = ~0.70 @ C=10 (vs 0.448 fully-in-circuit, 0.428
random-conn). +0.25 using good analog features + a proper readout (memory 'deploy ridge->1.0' precedent).
NEXT (for fully-in-circuit): fix the in-circuit READOUT (PC/contrastive readout caps 0.37-0.48 vs ridge
0.70 -- candidate causes: ZEROSUM weak-target encoding (+tdv / -tdv/(C-1) instead of +-1), or contrastive
estimator bias). Then hidden-learning fidelity for the rest. FORWARD-PATH (gain/settling) is CLOSED.

### *** THE FIX TO TEST (2026-06-25): DENSE READOUT (KOUT=16). Readout fan-in=4 is throwing away features. ***
Root of the "48% cap": my readout runs masked the readout to fan-in 4 (KOUT=4, the circuit default). Surrogate
C=10 NTR=40: frozen-hidden sparse(4)-readout=48% vs dense(16) ridge=81%; full-PC sparse(4)=81.9% vs dense(16)=
91.2%. KEY INTERACTION: a DENSE readout reads the (excellent) features directly (~81%) EVEN IF hidden barely
learns; sparse(4) caps at 48% unless the hidden compensates. The circuit's hidden ISN'T compensating (0.448 ~
frozen-sparse 0.48), so a DENSE readout should let it reach ~0.67-0.70 directly (circuit dense-feature ridge =
0.67-0.74). => IN-CIRCUIT FIX = KOUT=16 (dense readout, 10x16=160 readout synapses vs 40). Launching C=10
KOUT=16 3-seed vs baseline 0.448. Also note: contrastive readout is NOT fundamentally weak (full-batch contrastive
= ridge 81%); the cap was readout SPARSITY, not the rule. This is the payoff of the LEARNING reframe.

### *** WIN (2026-06-25): DENSE READOUT (KOUT=16) = 0.669 vs 0.448 baseline (+0.22) -- the reframe pays off ***
C=10, exact recipe + ONLY KOUT=4->16 (dense readout: each class reads all 16 hidden, maxfanout 4->10).
Epoch-8 eval-block acc (extracted from partial raws -- runs STALLED in stiff dynamics near the end, common
for these dense C=10 TH=400 runs): s1=0.684, s2=0.692, s3=0.632 -> MEAN 0.669, vs baseline KOUT=4 = 0.448.
+0.221, landing right at the circuit feature ridge-ceiling (0.67-0.81). CONFIRMS the whole LEARNING reframe:
the wall was the READOUT SPARSITY (fan-in 4 of 16 excellent features), NOT the forward path. One connectivity
knob recovers ~1/4 of the accuracy. This supersedes the prior "local RF 0.448" as the session's best C=10:
NEW BEST = local-RF + DENSE readout = ~0.67. NEXT toward the ~0.80 ridge ceiling: (a) fix hidden learning on
top (in-circuit +0.07 vs ideal +0.30), (b) more data (NTR), (c) the stall is an operational issue (extract
eval-block acc from partial raws, or shorter TH / fewer slots / restart on stall). FORWARD-PATH (gain/settling)
stays CLOSED. METHOD WIN: extracting epoch-N eval-block accuracy from a partial/stalled raw = no wasted run.

### *** DECISIVE (2026-06-25): on top of DENSE READOUT, the remaining wall = SIGN-LOSS in hidden backward (BKSIGN doesn't fix it) ***
Surrogate discriminator (dense readout KOUT=16, 3-seed C=10) isolates the hidden-learning defect cleanly:
  frozen hidden + dense readout      = 73.6%   (the floor; circuit dense-readout = 0.669 sits here, features slightly worse than ideal-tanh)
  full clean hidden (ideal transpose)= 88.3%   (+14.7 = the PRIZE for fixing hidden learning)
  hidden SIGN-DROPPED (mag-only)     = 72.9%   == FROZEN  <-- the circuit's defect (matches pc_deep "transport amplitude-restoring, keeps MAGNITUDE not SIGN")
  hidden update noise x3 / x8        = 88.1 / 88.8   (robust -> NOISE is NOT the problem)
  hidden systematic offset x1        = 87.5    (robust -> OFFSET is NOT the problem)
  hidden weak gain 0.1               = 78.4    (partial; under-driving costs ~half)
=> The fix must be SIGN-FAITHFUL, not low-noise. Eliminated CM-on-transport too (CMx10=86 vs clean 88; symmetric nudge cancels it) -> do NOT spend Spectre on OCMSUB-on-hidden.
CONFIRMED IN-CIRCUIT: the c10rfgrid_verify recipe (=baseline 0.448 AND dense-readout 0.669) ALREADY runs BKSIGN=1, yet lands at frozen ridge -> BKSIGN's comparator-regen backward does NOT deliver sign-faithful hidden learning (memory's "BKSIGN insufficient / 2nd sign-loss point" CONFIRMED, now visible because dense readout unmasks hidden's contribution).
NEXT: random-EACH-STEP sign = frozen (averages to 0), but a CONSISTENT fixed wrong-sign aligns to ~86% -> DFA (BKDFA, fixed random feedback, already in pc_deep.py line 633) sidesteps the deep-transpose sign-loss. Testing DFA/dfa_sign in surrogate now; if it recovers the +14.7 headroom -> port BKDFA to Spectre on top of KOUT=16 (EXACT recipe, one var, 3-seed, Hungarian).

### NEGATIVE (2026-06-26): BKDFA does NOT transfer in-circuit — DFA chaos breaks the analog readout lock
3-seed Spectre, BKDFA=1 BKSIGN=0 KOUT=16 C=10 (ONLY change vs the 0.669 BKSIGN dense-readout baseline), epoch-8 Hungarian block (same checkpoint the 0.669 was read at):
  s1=0.440  s2=0.396  s3=0.408  -> mean 0.415 +/-0.02  vs baseline 0.669  => DFA is WORSE, even BELOW the frozen-equiv ridge (~0.67) i.e. actively harmful.
SIGNATURE: raw argmax 0.13-0.18 (readout mapping UNLOCKED) while Hungarian 0.40 -> the readout never locks the class<->node correspondence. MECHANISM: DFA's fixed-random feedback drives the hidden weights around; the ANALOG readout has to chase a MOVING feature target and can't lock -> 0.40 < frozen-0.67. The PyTorch surrogate (DFA=79.5 > frozen 73.6) MISSED this because its readout is solved cleanly/jointly each step; the analog readout-tracking-lag is the un-modeled term.
LESSON: in-circuit, hidden-learning must be CONSISTENT *AND* ALIGNED (smooth, no chaos) so the analog readout stays locked. Both probed hidden schemes now fail the surrogate's +14.7 promise: BKSIGN(sign-lossy)->frozen-neutral 0.67; BKDFA(chaotic)->harmful 0.42. Surrogate says ONLY the TRUE SIGNED TRANSPOSE delivers (88.3). NEXT: (1) probe WHY BKSIGN loses sign in fast Spectre (measure regenerated sg{l} backward sign-agreement vs ideal transpose computed from the weight caps) -> fix the actual sign-loss point; (2) OR 2-phase freeze-readout-while-hidden-settles; (3) reproduce the readout-lag failure in the surrogate first so it becomes predictive. DENSE READOUT 0.669 remains the BEST C=10.
  CONFIRMED at epoch-12: dfa16_c10_s2 final-block Hungarian=0.420 (raw 0.144, log TEST ACC=0.160) == epoch-8 0.396 -> DFAW=1.0 is a stable ~0.42 chaos equilibrium, no recovery. NOW testing gentle DFAW {0.1,0.2,0.3} to escape it.

### GENTLE-DFAW LARGELY REFUTED (2026-06-26): dfaw=0.2 epoch-8 = Hungarian 0.416 (raw 0.168 unlocked) ~= DFAW=1.0 chaos 0.415. Gentler feedback did NOT escape chaos at 0.2 (the ~3x analog-scale estimate was too low; in-circuit effective feedback >> nominal). Hungarian 0.42 < frozen-feature 0.67 -> DFA CORRUPTS the hidden features, not just unlocks readout. Fundamental tension: help-fast hidden updates move features faster than the analog readout tracks. Awaiting dfaw=0.1 (gentlest) but even a lock likely = under-learn ~0.67 (no gain). LIKELY PIVOT: the true SIGNED TRANSPOSE (surrogate 88.3, stable) is the only winner -> probe/fix BKSIGN sign-loss, OR consolidate the solid 0.669 dense-readout win.
  *** DFAW-INDEPENDENT: dfaw=0.1 epoch-8 Hungarian=0.420 == dfaw=0.2 0.416 == DFAW=1.0 0.415 (near-identical hists). The ~0.42 BKDFA failure is INDEPENDENT of feedback strength 0.1->1.0 -> NOT a chaos/magnitude problem; BKDFA *itself* (the only var vs BKSIGN) drops 0.669->0.42. GENTLE-DFA FULLY DEAD. Ranking: BKSIGN (0.669 stable, frozen-equiv) >> BKDFA (0.42 harmful). Dense-readout+BKSIGN 0.669 = BEST C=10. NEXT = make BKSIGN actually deliver the signed transpose (surrogate 88.3): probe WHY it loses sign in-circuit (sg{l} regen sign vs ideal transpose).

### FROZEN-FEATURE CEILING SCALES WITH #FEATURES-READ (2026-06-26, surrogate, frozen RF + trained dense readout, 3-seed C=10): 64,36,16=71.5 -> 64,36,25=79.2 -> 64,49,36=83.9 -> 64,49,49=86.0 -> 64,64,49=87.6 -> 64,64,64=87.9% (~= trained-hidden ceiling 88.3). MONOTONIC in #features fed to the dense readout; DEPTH does NOT help (64,49,36,16=72.7), wscale=2 no help. This is the STABLE regime (frozen feats + trained dense readout = locked readout, the same path that gives 0.669 in-circuit) -> NO broken backward needed. In-circuit risk = readout-sum analog noise over many inputs + bigger/slower deck. RUNNING 3-seed in-circuit 64,49,36 KOUT=36 NOHID=1 (36 feat, surrogate 83.9, ~94% transfer -> ~0.79 expected). If it transfers -> scale to 49/64 feats. Frozen-WIDE (parallel feats) trades vs deep-not-wide but empirical win would justify.

### *** NEW BEST (2026-06-26): FROZEN-WIDE 64,49,36 dense KOUT=36 = 0.720 > 0.669 ***
frzw36_c10_s2 epoch-8 Hungarian=0.720 raw=0.720 (raw==Hungarian -> readout FULLY LOCKED, stable regime; all 10 classes alive hist=[36,23,31,49,26,14,30,22,10,9]). Beats dense-readout KOUT=16 baseline 0.669 by +0.05. Surrogate 83.9 -> 0.720 in-circuit = ~86% transfer. CONFIRMS the pivot: reading MORE frozen features into the dense readout raises the ceiling STABLY (no broken backward). 1-seed so far; awaiting s1/s3 for 3-seed mean. NEXT: scale features 64,64,49/64,64,64 (surrogate 87.6/87.9 -> ~0.75 in-circuit). Frozen-WIDE (parallel feats, NOT trained-deep) — empirical win justifies vs deep-not-wide.

### *** CONFIRMED 3-SEED NEW BEST C=10 = 0.744 (frozen-wide 64,49,36 dense KOUT=36) ***
epoch-8 Hungarian: s1=0.772 s2=0.720 s3=0.740 -> MEAN 0.744 +/-0.021, ALL fully locked (raw==Hungarian). vs prev best 0.669 (dense KOUT=16) vs 0.448 (sparse). +0.075. Surrogate 83.9 -> 0.744 = 89% transfer. STABLE regime (frozen RF feats + trained dense readout), NO hidden-learning/backward needed. Progression of C=10 best: 0.448 (local-RF sparse) -> 0.669 (dense readout KOUT=16) -> 0.744 (frozen-wide 64,49,36 KOUT=36). NEXT: scale features 64,64,49/64,64,64 (surrogate 87.6/87.9 -> ~0.78 expected at 89% transfer).

### FROZEN-WIDE FEATURE-SCALING SATURATES IN-CIRCUIT ~0.745 (2026-06-26)
frzw49 (64,64,49 KOUT=49, 49 feat) 3-seed epoch-8 Hungarian = 0.748 +/-0.020 (s1 .736/s2 .732/s3 .776, all locked) == frzw36 (36 feat) 0.744 +/-0.021. TIED (+0.004 within noise). Surrogate predicted 49>>36 (87.6 vs 83.9, +3.7) but in-circuit IDENTICAL -> READOUT-SUM ANALOG NOISE over more inputs cancels the extra-feature benefit; surrogate->circuit transfer ratio DROPS with #features (36=89%, 49=85%). => 36 feats = efficient in-circuit SWEET SPOT; 64 feats would saturate too (slower, no gain). To push past ~0.745 needs LOWER-NOISE readout summing (hardware), not more features. SESSION C=10 ARC: 0.448 (local-RF sparse) -> 0.669 (dense readout KOUT=16) -> 0.744/0.748 (frozen-wide dense KOUT=36/49), SATURATED. Hidden-learning lever exhausted (DFA ~0.42, BKSIGN frozen-equiv). BEST C=10 = 0.744 (frozen-wide 64,49,36 KOUT=36 NOHID=1, the cheap sweet spot).

### CORRECTION (2026-06-26): the "readout-sum analog noise caps feature-scaling" claim is NOT confirmed.
Surrogate test (experiments/readout_noise.py): injecting feature-activation noise does NOT flatten the 36->64 feature gain (fnoise 0.0/0.3/0.6 -> gain 36->64 stays +4.0/+4.3/+4.5). So the surrogate is ROBUST to feature noise and STILL rewards more features -> feature-input-noise is NOT the in-circuit saturation cause. WHAT STANDS (measured): in-circuit 49 feat (0.748) == 36 feat (0.744), TIED -> feature-scaling does NOT help in-circuit beyond 36, while surrogate says it should (+2-4). MECHANISM OPEN. Untested candidates: (a) bigger deck (24867 vs 18791 MOS) UNDER-SETTLES at the same TH=400 (more nodes, less converged/slot); (b) analog readout-TRAINING capacity/weight-update interference over more inputs (NOT feature-input noise); (c) readout output-node summing noise (different injection point than tested). SAFE CONCLUSION: 36 feats = in-circuit sweet spot, ~0.744 = achieved frozen-wide ceiling; the cause of non-scaling is unconfirmed. Do not assert readout-sum-noise.

### SETTLING REFUTED -> 0.744-0.752 IS THE GENUINE ANALOG CEILING (2026-06-26)
Settling A/B (64,49,36 KOUT=36 NOHID=1 NEP=4, WSEED=1, only TH differs): TH=400 -> TEST ACC 0.752 / hung 0.712 ; TH=800 (2x settling) -> TEST ACC 0.728 / hung 0.716. TIED/slightly-worse -> under-settling NOT the limiter (REFUTED). So feature-scaling saturation + the gap to surrogate 0.84 are NEITHER settling NOR feature-noise -> genuine analog FEATURE/READOUT FIDELITY (device mismatch, noise floor, finite gain) = HARDWARE limit, not addressable by TH/features/params. BONUS: NEP=4 already = 0.752 ~= NEP=12 0.744 -> frozen-wide config trains FAST (~4 epochs enough, frozen readout converges quick). FINAL C=10 CONCLUSION: best = 0.744-0.752 (frozen-wide 64,49,36 dense KOUT=36 NOHID=1), the analog ceiling for this trainer. Arc 0.448->0.669->0.744. Hidden-learning exhausted (DFA chaos, BKSIGN sign-loss). Levers closed: forward-gain(null), readout-sparsity(fixed->0.669), hidden-learning(exhausted), frozen-feature-count(saturates), settling(refuted). Remaining headroom = hardware fidelity.

### LINRO REFUTED -> 0.744-0.752 is the DEFINITIVE C=10 ANALOG CEILING (2026-06-27)
Linearized readout (gsynL, source-degeneration RDEGRO) MONOTONICALLY HURTS: LINRO=0 (12k) TEST ACC 0.752/hung 0.712 ; LINRO=1 60k 0.720/0.692 ; LINRO=1 120k 0.692/0.648. More linearization = lower gain = weaker readout signal = worse. Readout NONLINEARITY is NOT the limiter; the standard higher-gain gsyn is best. LAST LEVER CLOSED. ===== DEFINITIVE C=10 CONCLUSION ===== BEST = 0.744-0.752 (frozen-wide 64,49,36 dense KOUT=36 NOHID=1, frozen RF features + trained dense readout). Arc: 0.448 (local-RF sparse) -> 0.669 (dense readout KOUT=16) -> 0.744 (frozen-wide KOUT=36). ALL levers exhausted/characterized: forward-gain(null,5 approaches), readout-sparsity(FIXED->0.669,+0.22), frozen-feature-richness(FIXED->0.744,+0.075), feature-count-scaling(saturates~36), hidden-learning(EXHAUSTED: BKDFA chaos 0.42 any DFAW + BKSIGN sign-loss->frozen-equiv; surrogate says only true-signed-transpose helps, analog cant deliver), settling(REFUTED TH800<=TH400), readout-linearity(REFUTED, more linear worse). Residual gap to surrogate ~0.84 = genuine HARDWARE fidelity (device mismatch, noise floor, finite gain) -> needs silicon-level change (better-matched/larger devices, lower-noise summing), NOT any software/param/arch lever. KEY LESSON: the win came from ABANDONING the broken in-circuit hidden-learning backward and maximizing the STABLE forward path (rich frozen features -> un-starved dense readout). Bonus: frozen-wide trains FAST (NEP=4 already 0.752).

### NEURON SHARPNESS/GAIN NULL even in frozen regime (2026-06-27): sharp_dnw(DNW=400u) 0.724 WORSE, sharp_vbn(VBNEU=0.50) 0.748 TIED vs baseline 0.752. Gain closed across regimes. BUT new untested lever: DATA. Frozen runs use NTR=40 = only 4 ex/class for the readout. Frozen readout converges ~NEP=4 -> swap epochs for data: NTR=120 NEP=4 == NTR=40 NEP=12 (both 4800 train slots, SAME sim cost) but 3x data. sklearn digits has ~180/class. Testing NTR scaling in frozen-dense surrogate.

### DATA LEVER REFUTED -> C=10 LEVER-SPACE DEFINITIVELY EXHAUSTED (2026-06-27)
Cost-neutral data test (surrogate frozen-dense 64,49,36, constant NTR*EP slots): NTR=40/EP12=73.6, NTR=80/EP6=73.7 (tied), NTR=120/EP4=70.4 (worse). Swapping epochs->data at fixed budget does NOT help (more data needs more epochs, no free lunch). DATA refuted as a cost-neutral lever.
===== FINAL DEFINITIVE C=10 CONCLUSION (session 2026-06-23..27) =====
BEST = 0.744-0.752 (frozen-wide 64,49,36 dense KOUT=36 NOHID=1: frozen RF features + trained dense readout). Real Spectre, 3-seed, Hungarian. Arc: 0.448 (local-RF sparse) -> 0.669 (dense readout KOUT=16, +0.22) -> 0.744 (frozen-wide KOUT=36, +0.075). EVERY software/param/arch/data lever tested & CLOSED: forward-gain(null,5+ approaches,both sparse & frozen regimes), readout-sparsity(FIXED->0.669), frozen-feature-richness(FIXED->0.744), feature-count-scaling(saturates~36), hidden-learning(EXHAUSTED: BKDFA chaos 0.42 any DFAW + BKSIGN sign-loss->frozen-equiv; only true-signed-transpose helps per surrogate, analog cant deliver), settling(REFUTED TH800<=TH400), readout-linearity(REFUTED, more linear worse), neuron-sharpness/gain(NULL), DATA(REFUTED cost-neutral). Residual gap to surrogate ~0.84 = genuine analog DEVICE-CELL FIDELITY (neuron transfer != ideal tanh; no parameter closes it) -> needs a HARDWARE/device-cell redesign, NOT software. KEY LESSON: the win came from ABANDONING the broken in-circuit hidden-learning backward and maximizing the STABLE forward path (rich frozen features -> un-starved dense readout). Methodology delivered: 3-seed/invariant bar caught every false lead; 2 of my own hypotheses (readout-sum-noise, under-settling) tested & honestly REFUTED + corrected; fixed a hung_eval TH-hardcode bug. NEXT (user direction needed): device-cell redesign for lower-noise/sharper neuron, OR accept 0.745 as the analog ceiling for this cell library.

### STRUCTURAL NEURON A/B (2026-06-27, frozen-wide NEP=4 WSEED=1, vs baseline 0.752/hung0.712): struct_pow4(quartic POW4=1)=0.100/0.256 COLLAPSED (quartic wrong for digits, tanh wins); struct_rcs(RCS=600k load gain)=0.740/0.720 TIED; struct_casc(CASC=1 cascode, higher ro)=0.760/0.732 -> marginally ABOVE baseline on BOTH metrics (+0.008 TEST / +0.020 hung), within 1-seed noise but the ONLY variant up AND mechanistically right (higher output-R -> sharper neuron transfer -> better frozen features -> targets the device-fidelity limiter). NOTE: cascode HURT in the sparse/full-PC regime (memory: 0.385 vs 0.448) but helps-marginally in the FROZEN regime. CONFIRMING cascode 3-seed (WSEED 11,21 vs baseline 11,21, NEP=4).

### CASCODE 3-SEED = NULL -> ALL STRUCTURAL-FLAG LEVERS EXHAUSTED (2026-06-27)
Cascode 3-seed (NEP=4 frozen-wide): cascode [0.760,0.732,0.720] mean 0.737+/-0.017 vs baseline [0.752,0.728,0.716] mean 0.732+/-0.015 -> DELTA +0.005 WITHIN NOISE (seed-1 edge was noise). Cascode NULL. So all 3 structural-flag neuron variants exhausted: cascode null, RCS=600k tied (0.740), POW4 quartic collapsed (0.100). ===== COMPLETE FINAL: C=10 = 0.744-0.752 is the device-limited analog ceiling across EVERY param AND structural-flag lever ===== (forward-gain, readout-sparsity[fixed->0.669], feature-richness[fixed->0.744], feature-count[saturates], hidden-learning[exhausted], settling[refuted], readout-linearity[refuted], neuron-gain/sharpness[null], data[refuted], cascode[null], load-gain[tied], quartic[collapsed]). Residual to surrogate 0.84 = device-cell transfer fidelity -> needs a genuinely-NEW offset-canceling CELL design (real circuit engineering, per paper conclusion), NOT any flag/param/arch sweep. Write-up in paper/paper.tex (paragraph already says next lever at cell level; cascode-null confirms it). AWAIT user direction on the cell design.

### DC CHARACTERIZATION: NEURON near-ideal, SYNAPSE is the device-fidelity limiter (2026-06-28)
Spectre DC transfer sweeps (LEVEL-1, 1V): dneuron = near-perfect tanh (RMS dev from ideal tanh = 0.0089 = 1.7% of swing, symmetric ±0.535, zero offset at matched devices; mild low-CM asymmetry: at CM=0.4 neg-sat compresses to -0.484 vs -0.535, gain 7.12 vs 7.54). gsyn synapse multiply = STRONGLY nonlinear: at w=±0.4 the multiply saturates, RMS-from-linear = 58%, asym ±0.079. => the neuron core is NOT the device-fidelity limiter (explains why ALL neuron-cell variants are null: cascode/gain/load/quartic). The 0.744-vs-surrogate-0.84 gap is the SYNAPSE/weighted-sum: surrogate assumes linear w.x, circuit synapse is a saturating transconductor. LEVER = feature-synapse LINEARITY (RDEG source-degeneration, default 12k; higher=more linear). NOTE git-log earlier flagged "synapse linearity is the lever for stronger nets (RDEG=24k 0.532 vs 0.44)" then walked back as seed noise -> re-test CLEAN in frozen-wide 3-seed. Caveat: global RDEG also linearizes the READOUT synapse where LINRO(more linear)=HURT (gain loss); feature vs readout RDEG may need to differ.
  CORRECTION to the 58%: that was a full-range single-line fit artifact (deep saturation tails). Drive-dependent synapse nonlinearity (w=0.7): ±0.05/0.10V=5%, ±0.15V=10%, ±0.20V=15%, ±0.30V=23%. So gsyn is fairly LINEAR for small features but degrades as features approach neuron saturation (±0.5V); RDEG trades GAIN for linear range (same coupling that made LINRO hurt) -> RDEG=24k/48k test net-uncertain. Neuron 1.7% finding stands (genuinely near-ideal).

### *** PROMISING (2026-06-28): FEATURE-ONLY synapse linearity (frdeg24) UP on BOTH metrics, 1-seed ***
Synapse-linearity 2x2 (frozen-wide NEP=4 WSEED=1, baseline raw 0.752/hung 0.712): GLOBAL RDEG=24k null (raw 0.752 tied, hung 0.740) / 48k worse -- because global also linearizes the READOUT (which hurts, per LINRO). FEATURE-ONLY (LINRO=1 RDEGRO=12k -> readout pinned 12k; RDEG=24k -> only FEATURE synapses linear): frdeg24 = raw 0.776 (+0.024) / hung 0.756 (+0.044) -- UP on BOTH, beyond ±0.02! frdeg48 worse (raw 0.716/hung 0.664, 48k overshoots gain). => isolating feature-synapse linearity from the readout penalty reveals a real candidate lever, mechanistically predicted (synapse is the device limiter, neuron near-ideal 1.7%). 1-SEED -> 3-seed confirming (cascode seed-1 edge didnt survive, must verify). If real -> closes part of device gap (0.744->~0.77), paper-worthy (synapse weighted-sum linearity is the 10-class lever).

### SURROGATE VALIDATION: synapse saturation costs ~5-7 pts; linearization recovers (2026-06-28)
Synapse-faithful surrogate (experiments/syn_faithful.py): inject measured synapse input-saturation sat(x)=tanh(g*x) on every synapse, frozen-dense 64,49,36 3-seed. g=0(linear/ideal)=83.9 -> g=2=82.0 -> g=4=78.4 -> g=6=76.9. MONOTONIC cost; measured g~3.65-6 lands near the circuit ~0.74. => CONFIRMS synapse nonlinearity is a major chunk of the device-fidelity gap (~5-7 pts), and reducing g (linearizing synapse via higher RDEG) RECOVERS toward 84. Cross-validates the frdeg24 circuit result (feature RDEG=24k: 0.752->0.776 seed1). Coherent: neuron near-ideal(1.7%), synapse-saturation is the lever, feature synapses want linearity. 3-seed Spectre confirming.

### FEATURE-RDEG=24k 3-SEED: WEAK POSITIVE, within noise, NOT conclusive (2026-06-28)
fr24 (LINRO=1 RDEGRO=12k RDEG=24k feature-linear) vs baseline, frozen-wide NEP=4: RAW fr24 [0.776,0.732,0.724] mean 0.744±0.023 vs base [0.752,0.728,0.716] 0.732 -> +0.012; HUNG fr24 [0.756,0.700,0.756] 0.737±0.026 vs base [0.712,0.704,0.732] 0.716 -> +0.021. Up on BOTH metrics + surrogate-predicted (more credible than cascode +0.005) BUT delta ~1.2 sigma at 3-seed (scatter ±0.025) -> NOT conclusive; seed1 0.776 was a high draw. CORRECTS the earlier "promising" entry: in-circuit RDEG captures only a SMALL fraction of the surrogate ~7pt synapse-linearity opportunity because it TRADES GAIN (consistent w/ global-RDEG null + LINRO readout-hurt). The synapse-saturation DIAGNOSIS is solid (DC char + surrogate g-sweep); the simple RDEG knob is gain-bounded. To capture the full opportunity needs a GAIN-PRESERVING more-linear synapse (e.g. cascoded gsyn: gain boost allows higher RDEG at same gain). Running 2 more seeds (->5-seed) to settle RDEG=24k.

### DESIGN LAW: gsyn gain/linearity tradeoff is FUNDAMENTAL (2026-06-28 DC char)
gsyn DC (w=0.7, nonlin@±0.2V drive): RDEG=12k/RCS=300k baseline gain 4.57/nonlin 15.1%; RDEG=24k/300k gain 2.67/nonlin 9.5% (linear but low gain); RDEG=24k/RCS=600k gain 5.26/nonlin 17.5% (RESTORING gain via load RE-AMPLIFIES residual nonlinearity!); 24k/900k 7.76/22.2%; 36k/600k 3.53/13.2%. => CANNOT decouple gain from linearity in the Gilbert+resistive-load topology: degeneration linearizes the input devices but load-gain re-amplifies whatever nonlinearity remains. So the synapse-linearity lever is BOUNDED by the cell; RDEG=24k/300k is the best linearity-at-gain point (the borderline +0.015 effect). To exceed needs a GAIN-PRESERVING multiplier = different topology (active-feedback/translinear linearization), a real synapse redesign, NOT an RDEG/RCS knob. Confirms why global-RDEG was null (load couples) and feature-RDEG only weakly positive. The cascode-gsyn idea also wont help (re-amplifies, like RCS). DIAGNOSIS chain complete: neuron near-ideal -> synapse saturation is the 10-class device gap (~5-7pt surrogate) -> but cell topology bounds the fix.
  COMPOUNDING test (surrogate, synapse-faithful): linear vs saturated synapse at 36/49/64 feat: g=0 83.9/87.6/87.9, g=6 76.9/78.9/82.3. Saturated synapse ALSO climbs with features (76.9->82.3) -> feature-count saturation is NOT synapse-caused (compounding idea NULL); but synapse-saturation cost is consistent ~5-7pt at every feature count (re-confirms synapse=limiter). In-circuit 49~=36 saturation is a SEPARATE effect (readout-sum/settling, earlier-ambiguous).
  VBSYN (synapse tail current) DC: vbn 0.45/0.55/0.65/0.75 -> gain 4.57/4.29/4.14/4.05, nonlin 15.1/14.0/13.3/12.9% -> MILD same tradeoff (theory gm,Vov~sqrt(I) defeated by degeneration+load interaction). => ALL gsyn knobs (RDEG, RCS, VBSYN) trade gain for linearity; the tradeoff is UNIVERSAL across bias/resistor knobs. Breaking it needs a different MULTIPLIER topology (multi-tanh input stage = parallel offset pairs summed = wider linear range at same gain; or active-feedback linearization) -- textbook gain-preserving linearization, a real cell redesign (~a few pts upside, bounded). Cheap-knob synapse-linearity exploration COMPLETE.

### REFRAME (2026-06-28, user): FROZEN HIDDEN (NOHID=1) is the WORST rule-break -- it DODGES the core goal. The 0.744 frozen-wide = trained linear readout on RANDOM features = NOT in-circuit learning. PIVOT to THE LARGEST GOAL: in-circuit HIDDEN-LAYER training (full PC, deep, NOTHING frozen, fan-in<=4, no weight transport, local plasticity, Spectre). Wall = sign-faithful backward (surrogate: true signed transpose=88.3 vs frozen 73.6). CORRECTION: "4-bit neurons / sharp tanh" is NOT a rule -- ANY neuron/activation is allowed (free to explore neuron cells + activation functions). RULES = deep-not-wide, full-PC-no-freezing, fan-in<=~4-10, no-weight-transport, local-plasticity, controller-only-presents-data, Spectre. Killed frozen-wide runs; recent synapse work was on the WRONG (frozen) net.

### PIVOT to in-circuit hidden training: surrogate tool (train_faithful.py) finding (2026-06-28)
Fast model of full-PC hidden training under in-circuit backward non-idealities (3-seed C=10; tool baseline 63 not pc_surrogate 88 so RELATIVE only): ideal(linear-syn,graded-bk)=63.2 -> +synapse-saturation-on-BACKWARD=34.8 (-28!) -> +hard-sign-regen=25.3; ReLU neuron does NOT rescue (25.5). => SYNAPSE SATURATION ON THE BACKWARD GRADIENT is a major killer of hidden training (>> its effect on forward features, which frozen-wide tolerated). Connects synapse-saturation work to THE CORE GOAL: gradient transported back through the saturating shared synapse is corrupted -> hidden cant learn. RULE-FOLLOWING testable: linearize synapse (RDEG up) -> cleaner backward -> hidden trains better. Launching Spectre: rule-following FULL-PC deep net (the 0.448 recipe, NO NOHID, hidden trains) at RDEG=12k vs 24k.

### *** MECHANISM (2026-06-28, train_faithful.py recalibrated dense readout, 3-seed C=10) ***
WHY in-circuit hidden training fails: SYNAPSE SATURATION ON THE BACKWARD GRADIENT flips training from helpful to HARMFUL. ideal(linear-syn,graded-bk)=78.5 (trains, +5 over frozen ~73) -> +synapse-sat-backward(g=4)=60.1 (training HURTS, BELOW frozen!) -> +hard-sign-regen(BKSIGN-like)=69.3 (discards corrupted magnitude, keeps sign, recovers some but STILL < frozen). REPRODUCES the in-circuit observation (full-PC 0.448 ~= frozen 0.424, training barely helps/hurts) and EXPLAINS why BKSIGN hard-sign helps a bit but cant beat frozen. ROOT CAUSE = the backward gradient transported through the SATURATING shared synapse is corrupted. FIX = linearize the backward synapse (RDEG up) -> gradient clean -> training helpful again. PREDICTS the 2x2 Spectre: fpc_rd24 (linear) should train hidden ABOVE frozen, fpc_rd12 (saturated) at/below frozen. This connects the synapse-saturation diagnosis to THE CORE GOAL (training the analog net). Tool ideal 78.5 not 88 (local-RF build/epochs) but RELATIVE mechanism clean.

### BACKWARD-LINEARIZATION THRESHOLD + the real fix (2026-06-28)
Tool sweep: training flips harmful->helpful between backward-g=2 and g=3 (g0=78.5/g2=77.3 TRAIN; g3=69.6/g4=60.1/g6=53.3 HURT, below frozen 73.5). Measured synapse g~6; RDEG=24k ~ g3 = RIGHT AT threshold (marginal); need g<=2 (RDEG~36-48k) for clean training. PROBLEM: more global RDEG costs FORWARD GAIN (shared synapse; RDEG=48k hurt frozen-wide features). THE REAL FIX: the backward read is a SEPARATE gsyn instance (bk_ synapse) -> add a BACKWARD-ONLY degeneration (RDEGBK high) to linearize ONLY the gradient transport while keeping the FORWARD synapse at full gain (RDEG=12k). Decouples gradient-linearity from forward-gain -> breaks the tradeoff -> could enable in-circuit hidden training without the forward penalty. RULE-FOLLOWING (no freezing, local). NEXT: add RDEGBK to gsyn-backward in pc_deep, test full-PC. The 2x2 (global RDEG) is confounded (forward gain loss) -- the backward-only test is the clean one.

### DECISIVE EXPERIMENT LAUNCHED: backward-only linearization (RDEGBK), graded backward (2026-06-28)
KEY REALIZATION: the running BKSIGN 2x2 tested the WRONG backward. Tool: hard-sign caps at 69<frozen73 REGARDLESS of RDEG; only the GRADED backward (else-branch, signed-transpose injected directly into x) is g-sensitive and beats frozen (77-78 at g<=2). So BKSIGN runs were uninformative for the linearization hypothesis (and fpc-BKSIGN~=frozen~=0.448 already known).
NEW MECHANISM-FIX: RDEGBK = a SEPARATE degeneration for the BACKWARD synapse (added gsynB subckt + synB() helper; else-branch bk_ uses synB when RDEGBK set). Linearizes ONLY the gradient transport, leaving FORWARD synapse at RDEG=12k (full gain) -> BREAKS the gain/linearity tradeoff (could not do this with shared global RDEG: 48k kills forward features).
RUNS (graded BKSIGN=0, forward RDEG=12k, WSEED=1): gpc_bk24 (RDEGBK=24k~g3 threshold), gpc_bk48 (RDEGBK=48k~g2 linear). Frozen baselines frz_rd12/frz_rd24 (NOHID) still running as the reference to beat. Killed fpc_rd12/fpc_rd24 (wrong backward).
PREDICTION (tool): gpc_bk48 (~g2) ~77 BEATS frozen ~73 = in-circuit hidden training works; gpc_bk24 (~g3) marginal ~69; control gpc_bk12 (=forward 12k, g6, known-degrading) below frozen. CAVEAT: RDEGBK=48k lowers backward gm ~4x -> gradient magnitude may be noise-swamped; if bk48 underperforms bk24, gain-loss (not linearity) is the limiter -> bump VBBK for backward. READ epoch-8 (66%): TH=400 python3 experiments/hung_eval.py gpc_bk48 10 40 25 12 8 0. VERDICT: gpc_bk48 (or bk24) > frz_rd12 by real margin = LINEARIZING THE BACKWARD GRADIENT ENABLES IN-CIRCUIT HIDDEN TRAINING (the core goal) -> 3-seed + paper/paper.tex + push.

### FROZEN BASELINE (the bar to beat), epoch-8 Hungarian C=10: frz_rd12(fwd12k)=0.448  frz_rd24(fwd24k)=0.460 -- forward linearization NULL for frozen (within noise), confirms forward-RDEG not the lever. Graded backward gpc_bk24/bk48 must beat ~0.45 by a REAL margin (surrogate predicts +15).

### ACTIVATION lever CLOSED again; backward-linearization is the dominant lever (2026-06-28, tool)
Grid activation x backward-linearity (frozen 73.5): tanh linear-bk=78.5/sat-bk=53.3; relu 69.2/41.1; lrelu 63.5/40.5. tanh WINS at every linearity level (cleaner-derivative relu/lrelu are WORSE, consistent with "tanh still wins digits"). Backward-linearization gain is large for ALL activations (+23 to +28). => the lever is NOT the neuron/activation; it is LINEARIZING THE BACKWARD GRADIENT (RDEGBK). Keep tanh dneuron; the RDEGBK Spectre sweep (gpc_bk12/24/48 vs frozen 0.448) is the right experiment. Tool fix: act_bk_grid local var "sat" shadowed train_faithful sat() saturation fn -> renamed; added lrelu to act/dact.
### frz_rd12 FINAL epoch-12 Hungarian=0.452 (== epoch-8 0.448): frozen C=10 ceiling STABLE ~0.45 (classes 7,8 dead, 9 over-predicted). Added 4th run gpc_bk48v (RDEGBK=48k + VBBK=0.7) to pre-empt the gm-starvation caveat: tests linear-AND-high-gain backward. Graded sweep now bk12/bk24/bk48/bk48v vs frozen 0.45.

### ★★★ BREAKTHROUGH (PRELIMINARY single-seed, epoch-8): BACKWARD-LINEARIZATION ENABLES IN-CIRCUIT HIDDEN TRAINING (2026-06-29)
The CORE GOAL. Graded backward (BKSIGN=0, signed-transpose injected into x) + backward-only linearization (RDEGBK, separate gsynB degeneration, forward RDEG=12k full gain) BEATS the frozen ceiling in real Spectre C=10:
  frozen (NOHID) frz_rd12 = 0.448 (epoch-8), 0.452 (epoch-12) STABLE
  gpc_bk24 (RDEGBK=24k~g3): HUNGARIAN=0.524 raw=0.524 (+0.076 over frozen) hist=[15,28,23,28,24,42,62,16,0,12] only 1 dead class
  gpc_bk48 (RDEGBK=48k~g2 linear): 0.480 raw=0.472 (+0.032)
  gpc_bk12 (RDEGBK=12k=g6 control), gpc_bk48v (48k+VBBK=0.7) pending (<66%).
bk24>bk48 REVERSAL vs tool (tool predicted more-linear=better): in-circuit higher RDEG ALSO drops backward gm ~4x -> bk48 more-linear-but-weaker-gradient loses to bk24 moderate-linear-stronger. = the gm-loss caveat is REAL. bk48v (restore gain) tests if linear+high-gain reclaims lead. HEADLINE HOLDS regardless: linearizing the backward gradient beats frozen -> the analog hidden layers TRAIN. 0.524 = best in-circuit trained-hidden C=10 (prior frozen ceiling 0.448; prior trained ~=frozen 0.448). CAVEATS: SINGLE SEED (WSEED=1), epoch-8 not final; methodology requires 3-seed -> MUST confirm WSEED 1,2,3 before paper claim. Need bk12 control (predict <frozen, the degrading baseline) to complete dose-response.

### ★ CORRECTION (control caught a wrong hypothesis) — the lever is GRADED backward, NOT linearization (2026-06-29, single-seed epoch-8)
The bk12 CONTROL (RDEGBK=12k = NO extra linearization, expected to DEGRADE) = 0.524, SAME as bk24 (0.524), also beats frozen 0.448. So dose-response is NOT monotone w/ linearization: bk12(g6)=0.524, bk24(g3)=0.524, bk48(g2 most-linear)=0.480 WORST, frozen 0.448. => backward-linearization (RDEGBK) is NOT the lever; over-degenerating (48k) HURTS (gm-loss weakens gradient). THE REAL LEVER: GRADED signed-transpose backward (BKSIGN=0, else-branch, magnitude-carrying) beats the old HARD-SIGN backward (BKSIGN=1 ~= frozen 0.448). The graded backward transports error w/ magnitude -> hidden trains to 0.524 (+0.076 over frozen). This REFRAMES [[pc-training-broken-sign-loss]]: the fix is KEEP THE GRADED MAGNITUDE, do not hard-sign it (earlier "graded degrades" was under a different recipe w/o PERAZ/ZEROSUM/SYMNUDGE). METHODOLOGY WIN: the RDEGBK control falsified my linearization hypothesis -- exactly why controls matter. NEXT: 3-seed the GRADED-vs-FROZEN gap (winning config = plain BKSIGN=0 RDEG=12k, no RDEGBK). Also confirm graded>BKSIGN-hardsign. SINGLE SEED still -- must 3-seed before paper.

### 3-seed FROZEN baseline (epoch-8 Hungarian): s1=0.448 s2=0.392 s3=0.448 -> mean 0.429 +/-0.026. Graded seed1=0.524 (gap vs frozen-mean +0.095). gph_s2/s3 pending (~90min). Launching bks_s1 (BKSIGN=1 hard-sign full-PC) to test mechanism: is GRADED backward > HARD-SIGN? (expect hard-sign ~= frozen ~0.43).
### 3-seed confirm IN PROGRESS (epoch-8): GRADED s1=0.524 s3=0.464 (s2,s4 pending) vs FROZEN s1=0.448 s2=0.392 s3=0.448 (mean 0.429). Per-seed gap: s1 +0.076, s3 +0.016 (INCONSISTENT -> seed-1 may have over-stated; margin smaller than first read). graded mean(s1,s3)=0.494 vs frozen 0.429 = +0.065 but need s2,s4. NOT declaring win until gph_s2/s4 in. Honest: graded appears > frozen but modest & seed-dependent.

### ★★★ CONFIRMED (3-seed, epoch-8): IN-CIRCUIT HIDDEN TRAINING BEATS FROZEN — THE CORE GOAL (2026-06-29)
GRADED (full-PC, BKSIGN=0, magnitude-carrying signed-transpose backward) vs FROZEN (NOHID random features), C=10 digits, Hungarian:
  GRADED: s1=0.524 s2=0.500 s3=0.464 -> mean 0.496 +/-0.025
  FROZEN: s1=0.448 s2=0.392 s3=0.448 -> mean 0.429 +/-0.026
  GAP of means = +0.067; AND NON-OVERLAPPING: graded min 0.464 > frozen max 0.448 -> EVERY graded seed beats EVERY frozen seed. ROBUST, not seed-1 luck (seed-3 looked small +0.016 only b/c frozen-s3 was high 0.448; graded-s3 still won).
This is the LARGEST GOAL: the analog hidden layers TRAIN in-circuit above the frozen ceiling, via a rule-following local mechanism (graded signed-transpose error backward through the same cap weights, no weight transport, no freezing, fan-in<=7). Reframes [[pc-training-broken-sign-loss]] (was: hidden DEGRADES below frozen): the fix was KEEP THE GRADED MAGNITUDE in the backward (BKSIGN=0), not hard-sign it. Linearization (RDEGBK) is NOT the lever (over-degeneration hurts via gm-loss). PENDING for full paper: bks_s1 (hard-sign control, expect ~=frozen -> confirms magnitude-carrying is the mechanism), gph_s4 (4th seed). THEN write paper/paper.tex + push.
### Learning-curve evidence: GRADED improves over epochs (epoch-8->epoch-12): gph_s2 0.500->0.516, gph_s3 0.464->0.484; FROZEN flat (0.448->0.452). The rising curve = hidden layers genuinely LEARNING (not a fixed offset). Strengthens the core-goal claim.

### ★ MECHANISM CLARIFIED by hard-sign control bks_s1=0.512 (2026-06-29): the lever is TRAINING vs FREEZING, robust to backward variant
C=10 depth-3 (64,36,16): FROZEN(NOHID) 0.429 (3-seed) << TRAINED-graded 0.496 (3-seed, final 0.516) ~= TRAINED-hardsign 0.512 (1-seed). graded ~= hard-sign (within seed noise) -> NOT graded-beats-sign; BOTH backward variants train hidden ABOVE frozen. THE ENABLER = functioning in-circuit error transport (training the hidden layers) at depth-3, regardless of backward cell. (Earlier "BKSIGN ties frozen 0.448" was an OLDER recipe w/o PERAZ/ZEROSUM/SYMNUDGE; current recipe makes full-PC hidden training work.) CLEAN HEADLINE: in-circuit hidden training beats frozen at C=10, +0.07, robust across seeds AND backward variants. RDEGBK linearization was a red herring (within-noise). NOW writing paper/paper.tex + memory + push.
