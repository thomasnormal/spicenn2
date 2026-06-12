# Equilibrium-Propagation fork — milestone result (in ngspice)

Energy-based learning for the analog NN chip: make the network a physical system that
relaxes to an energy minimum, and learn from a purely local two-phase rule. If it holds,
the explicit backward pass, weight transport (reading Wᵀ), and the depth gradient problem
all dissolve into physics. This fork tested whether that is real **for our device family**.

**Verdict: all three de-risking milestones pass.** A reciprocal network built from our own
triode MOSFETs relaxes to a unique stable equilibrium, the triode is a clean programmable
reciprocal conductance, and the two-phase EP local update trains toy tasks in-circuit —
no transpose, no backward pass. The energy-based path is real for this device, in the
near-linear (small-swing) regime. The open frontier is in-range nonlinearity (XOR), which
collides with the small-Vds reciprocity constraint — see Limitations.

## PREDICTIVE CODING — a deep net that learns XOR entirely as analog dynamics (`gen_pcL.py`)
The project goal: a network where *both inference and weight changes are the analog circuit relaxing*,
every weight updating only from the local voltages at its own two terminals — no backprop, no digital
gradient, no controller optimizer. **Achieved with Predictive Coding: a 2-layer current-mode network
trains XOR in ONE continuous ngspice `.tran`, 20/20 seeds (H=4 hidden), genuine learning** (e.g. 2/4
→ 4/4 by epoch 20, then stable).

How it meets every clause:
- **Inference & learning are one relaxation:** activities settle while the weight caps integrate,
  simultaneously, in the same `.tran`.
- **Weights are charge on caps,** *shared* across the forward-prediction synapse and the backward
  error-feedback synapse — so the transpose needed for credit assignment is physical wiring (no
  weight transport).
- **Strictly local update:** each W_ih cap integrates ε_h·x_in, each W_ho cap integrates ε_o·a_h —
  the PC rule dW/dt ∝ (error)·(presynaptic activity), realized by the chopper-free `gprod` cell.
- **No optimizer/backprop:** the controller only writes the PWL input/label schedule; the output is
  hard-clamped to the label and the error neuron ε = (label − prediction) is an explicit node (`esub`).

Why PC and not EP: EP's update is a *difference of two phases*, which dragged in the whole offset /
chopper / CDS apparatus. PC has *explicit error neurons*, so the update is a **single local product**
— no two-phase, no offset-cancellation. Hard-clamping the output (vs EP's soft nudge) makes the error
a full-strength signal you can read directly.

Cells: `gsyn` (constant-tail Gilbert synapse → weight-independent common-mode), `cmld` (CMFB load),
`esub` (error = x−μ, diff-difference amp), `dneuron` (nonlinearity), `gprod` (local-product weight
integrator). Recipe: `H=4 KMAP=0.3 WL=1000u RNODE=20k VBSYN=0.45 VBNEU=0.35 GBLH=GBLO=0.85 TH=250
NEP=160 SGNH=1 SGNO=-1` + `.options cshunt=1e-14 gmin=1e-10 reltol=2e-3` (the last fixes `.tran`
timestep-too-small aborts; without it some seeds fail to *simulate*, not to learn).
The single-layer warm-up (`gen_pc1.py`, OR/AND 5/5) proved the loop first; numpy `pc_check.py`
validated the algorithm (XOR 20/20, continuous-update best). Honest debugging note: an early
"μ_o stuck at 0" that sent me down CMFB rabbit holes was substantially a *readout bug* (debug code
indexing the wrong `wrdata` columns) — the cascade had been propagating.

## FULLY-ANALOG UPDATE — transistor cells compute the gradient; CDS makes it train (`ep_dtcm.py PHYSUPD=1`)
The last digital piece — the weight update — is now **computed in silicon**. Each weight has a
transistor Gilbert cell that multiplies its two local activities φ_s·φ_d and integrates the product
onto the weight cap; the forward pass stays the analog equilibrium (`.op`). The only thing the
controller does is present data, read, and apply the step (η, momentum, freeze, restart — the
optimizer bookkeeping a tiny on-chip digital block would do).

**The problem and the fix.** A naive physical cell gives a *noisy* gradient — cos≈0.6, per-weight
**sign agreement only ~60%** — and **cannot train** (SGD ascends; sign-SGD needs ~80%). The culprit
is the time-domain **choppers** used to form the two-phase (nudge−free) difference: they inject
switch charge *and* fail to cancel the cell's own offset (input-chopper `gupd2` leaves the offset
in → 60%; output-chopper `gupd3` cancels offset but the 4-switch charge-injection makes it *worse*,
cos 0.31). The fix is **correlated double sampling (CDS, `CELLU=cds`)** — a standard analog
offset-cancel readout: run a **chopper-free** product cell on the free activities, then on the
nudge activities, and take the **read difference**. The identical cell offset cancels, with zero
chopper noise. Fidelity jumps to **cos≈0.85 / sign≈80%** (at `PHISC≈0.35`, which keeps the Gilbert
in its linear range — larger swing saturates and hurts).

**Result: at that fidelity the fully-physical update trains XOR — 28/28 fresh seeds to stable 4/4**
(8/8 on seeds 1–8, 20/20 on seeds 20–39; only 3 needed a restart). Recipe: `PHYSUPD=1 CELLU=cds
PHISC=0.35 CWU=400p THU=600 GBNU=1.1 ETA=8 MOM=0.6 ITERS=130` + cosine anneal + freeze + restart
(USIGN=−1). The multiply (the costly, parallel, per-weight part of the gradient) is done by
transistors; CDS — the free/nudge subtraction — is a differential read of two cap states, an analog
technique, not digital arithmetic. **This is fully-analog in-circuit weight-update learning that
works.** The earlier "research barrier" was not analog-fundamental; it was chopper artifacts, and
CDS removes them.

**It generalizes to real data (`phys_digits.py`).** On the sklearn-digits net (NHID=16, 114 weights,
2 classes, local 2×2 patches) the CDS gradient fidelity is even *higher* than XOR — **cos 0.984,
sign 96%** (pixel inputs are continuous, so less Gilbert saturation than XOR's binary ±). Training
that classifier with the **physical** CDS update (controller streams examples through the persistent
ngspice forward; one product-cell-bank run on all free activities + one on all nudge activities per
epoch; weights live on caps) climbs **50% → 82.5%** test accuracy. So the fully-analog update is not
XOR-specific: it computes faithful gradients and trains a real multi-class classifier, on the
scalable persistent-controller path. (Accuracy is modest because the net/data/epochs are small for
ngspice, not because of the update — the gradient is cos-0.98 faithful.)

## END-TO-END continuous-time EP — physical weights + physical update in one `.tran` (`gen_e2e.py`)
The whole training loop runs inside a **single ngspice `.tran`** with **no numpy in the loop**:
weights are differential voltages on **caps** (they gate the `dsyn` synapse tails — the weight
literally *is* the gate voltage), per-synapse 4-quadrant cells physically correlate the local
pre/post activities and chopper the two-phase (nudge−free) difference back onto the caps, a PWL
phase clock drives free/nudge, and the output is nudged toward the target by switch transistors.
The controller only *writes the PWL schedule* (inputs, target, clock) — it never computes a
gradient or a weight. ~400–700 MOSFETs depending on width. See `ep_e2e.png`.

**What is demonstrated (all in-circuit, one `.tran`):**
1. **A minimal 1-weight EP regression trains from scratch** — the weight cap converges so the
   output tracks the target, **both signs** (Fig A). The closed fast(node)/slow(weight) loop is
   stable and the EP fixed point is `w·x = target`.
2. **The continuous net REPRESENTS XOR** — loaded with the trained weights, the `.tran`
   relaxation classifies all four patterns **4/4** (Fig B: out = [−39,+39,+68,−36] mV vs target
   [−,+,+,−]). The circuit, the diff-pair neurons, the reciprocal hidden↔output coupling, and the
   weight-cap→gain mapping are all correct.
3. **The physical update HOLDS the XOR solution in-circuit** — starting at the solution with the
   update running, accuracy stays **4/4 over 40 epochs** (Fig C); the in-circuit update direction
   is correct (verified by sign scan: `SO=−1, SH=+1` for the differential-only cell).

**Simplifications / improvements over the stepped controller:**
- **One `.tran`, no Python loop** — the stepped `ng_live` driver collapses to a PWL schedule.
- **Differential-only update cell (`gupd2`)** — pushes current onto one weight cap and pulls an
  *equal* current from the other → **zero common-mode injection**. The naive common-injecting
  cell (`gupd`) lets the weight-cap CM run to the rail and the differential collapses to 0; the
  differential-only cell removes that runaway at its source.
- **In-circuit LR anneal** (ramp the update tail bias `Vgbn` as a PWL) and **neuron-bias
  diversity** (`NDIV`) — both pure-circuit knobs.

**Honest open frontier — from-scratch XOR does NOT yet converge in continuous time.** Pursuing it
revealed a *chain of failure modes*, each fix exposing the next — the signature of a problem where
many analog conditions must hold at once:
1. **Common-mode runaway** (cell `gupd`): the weight-cap CM rails, differential collapses to 0.
   → fixed by the **differential-only cell `gupd2`** (zero common injection).
2. **Fixed-point decay**: even at the solution the margin slowly shrinks (Fig C orange). Traced to
   **chopper charge-injection** — the drift scales as 1/C_weight (≈30 mV over 40 ep at C=40p vs
   ≈8 mV at 160p) and a saturation CMFB sink stabilizes it. **C_weight=160p + CMFB=120u → flat
   fixed point** (drift ≈ −2 mV, i.e. holds/slightly grows). So the solution can be made a genuine
   attractor.
3. **Bias-domination**: with the decay nulled (progress now *retained*) and a high learning rate,
   the *easy* bias-weight gradient accumulates and dominates before the input→hidden features form
   — the net converges to a **degenerate, input-independent constant output** (all four patterns
   ≈ equal). The hidden layer never differentiates.
Bias-LR throttle (`GBNB`) helps: most seeds then stay non-degenerate and reach **3/4 — but those
are *linear cuts*** (out ≈ tracks x1, or NAND), the ceiling a linear separator hits on XOR. The
hidden layer does not form the nonlinear XOR partition.

**Decisive negative result:** adding *capacity + diversity* (H=10, strong `NDIV`) — the lever that
fixed reliability in the discrete screen — makes the continuous net **worse**: all seeds go
degenerate. More hidden units feeding the reciprocal output→hidden loop *reinforce* the constant-
output (bias-only) attractor. So this is **not** a capacity problem; the continuous dynamics +
reciprocal coupling + the physical update create a genuinely harder optimization landscape than the
discrete screen, with a robust degenerate attractor.

I then *tested* the obvious architectural fix — **breaking the reciprocal symmetry** (scaling the
backward output→hidden synapse to 0.3× via `BWSCALE`, so the hidden state is driven more by the
input than by the degenerate output feedback). It reduced degeneracy on some seeds but **still
capped at 3/4** (e.g. out = [−,+,+,+] = OR, not XOR). So the 3/4 barrier survives *every* principled
lever I tried — differential-only cell, decay-null, bias-throttle, capacity, diversity, and the
symmetry break. The hidden layer simply does not form the nonlinear XOR partition under continuous
EP dynamics here. This is a real research-level barrier, not a tuning gap; a heavier change
(explicit hidden-feature curriculum, or a different relaxation/update scheme) would be needed.

So the end-to-end *machinery* is real and the update direction is validated in-circuit, but
reliable from-scratch XOR currently lives in the discrete screen below (170/170), **not** in the
fully-continuous chip. Fixes that DID work and are kept: differential-only cell (`gupd2`),
charge-injection-nulled weight caps (`CW=160p`+CMFB), in-circuit LR anneal, bias-LR throttle.

## RELIABLE XOR — fully-differential transistor EP + single-NMOS update (`ep_dtcm.py`)
The open frontier above is now closed. The fully-differential **current-mode** EP network
(crossed diff-pair synapses + diff-pair neurons, ~218 MOSFETs, offset-free) trains XOR
**reliably across seeds** — the metric the earlier single-ended nets failed.

**Result: 170/170 fresh seeds reach a stable, frozen 4/4-correct XOR solution** (two
independent ranges: seeds 100–199 = 100/100, seeds 200–269 = 70/70; *stable-final* accuracy,
i.e. the network still classifies all four patterns after training stops). **69/70 solve on
the very first attempt;** restart was invoked for 1 seed.

The reliability came from the *update rule and the training controller*, not a fancier neuron:
1. **Single-NMOS `(Δv)²` update (`UPDATE=sqdiff`)** — the square-law rule from one saturation
   NMOS (`I∝(φ_i−φ_j)²`) is *more* reliable than the exact Gilbert product: its self-terms
   regularize, so the net reaches XOR on ~100% of seeds (best-acc). The simpler physical cell
   is the more robust one.
2. **Freeze-on-convergence** — the controller stops nudging once all four patterns are correct
   with margin (the natural physical stopping rule: stop nudging when the chip has solved the
   task). This converts "finds the solution" into "holds it" — without it, ~10–15 % of seeds
   drift off the solution after grazing it (post-peak drift); with it, stable-final ≡ solved.
3. **Restart-on-stall** — if an attempt plateaus unsolved (no best-loss improvement for `STALLW`
   iters), the controller re-initializes the weight caps and retries. Needed ~1 % of the time;
   a safety net for the rare thin/stuck init, not a crutch.
4. Minor stabilizers: cosine LR anneal (`ANNEAL=cos`), small weight decay (`DECAY`), width 12.

**Neuron finding (the thing explicitly explored):** the base differential diff-pair neuron is
*sufficient*; it did not need redesign. A 2-stage higher-gain neuron (`NEU=two`) is marginally
better but much slower in sim; **naive high gain via a huge load resistor (`RLNEU=1.6M`) is
catastrophic (0/12)** — gain must come from cascaded stages, not from starving the load. So the
honest lever for reliability was the local update rule + the controller, not neuron gain.

Recipe: `NHID=12 UPDATE=sqdiff ANNEAL=cos AFLOOR=0.05 DECAY=0.02 MOM=0.3 ITERS=220 FREEZE=1
MARGIN=0.15 HOLD=2 NRESTART=4 STALLW=80` (USIGN=−1, WLOAD=3000u, RB=2e5, TD=0.10). The update
primitive here is a numpy *rule-screen*; the *physical* single-NMOS squarer cell that computes
it is validated separately in Xyce (`nmos_cell.cir`, below).

## (a) RISK 1 — the reciprocal network relaxes to a unique stable equilibrium
`relax1.cir`: 3 inputs → 2 hidden → 1 output, fully-connected **symmetric two-terminal
conductances** (ideal resistors here, to isolate the relaxation question), diode-clamp
neurons, a capacitor per free node. `.tran` to 300 ns.

- Settles cleanly (τ≈RC≈10 ns), **no oscillation, no runaway**.
- **Unique fixed point:** four very different initial conditions — (0,0,0), (1,1,1),
  (1,0,1), (0.2,0.9,0.1) — all converge to the *same* equilibrium
  (h1,h2,o1)=(0.5400, 0.4288, 0.4793). A symmetric conductance network has a convex
  co-content energy, and the circuit physically minimises it. Confirmed.

## (b) RISK 2 — the programmable triode conductance is reciprocal and gate-settable
`rec2.cir`: one NSYN NMOS (`LEVEL=1 VTO=0.2 KP=50u LAMBDA=0`) in triode as a two-terminal
conductance; common mode held fixed, both terminals driven symmetrically; current read with
a series ammeter.

- **Reciprocal:** branch current is antisymmetric in Vab to numerical precision
  (asym = 0.00%) for every gate voltage and |Vab| ≤ 0.3 V. (An earlier 10–25% asymmetry was
  a *test artifact* — letting the common mode drift changes the gate-to-source reference;
  the device itself is symmetric.)
- **Programmable:** small-signal G is set linearly by the gate, matching
  G = KP·(W/L)·(Vg−Vt−Vs):

  | Vg | 0.9 | 1.2 | 1.5 |
  |---|---|---|---|
  | G | 20 µS | 50 µS | 80 µS |
  | R | 50 kΩ | 20 kΩ | 12.5 kΩ |

- **Honest caveat (verified analytically):** energy-reciprocity needs the *Jacobian* to be
  symmetric, i.e. ∂I/∂Vd + ∂I/∂Vs = 0. For the triode this sum equals −KP(W/L)·Vds — exact
  only at small Vds; the Vds²/2 term breaks it at O(Vds). **Keep neuron swings small.** The
  brief's warning is correct and it drove the training design below.

## (c) RISK 3 — the two-phase EP local update trains a toy task
`ep_train.py`: 2 inputs (+ complements) → 1 output, **signed weights as differential triode
conductance pairs** (G⁺ to xᵢ, G⁻ to x̄ᵢ), a leak to fix the operating point. Small signal
swing (logic 0/1 = 0.4/0.6 V) to stay in triode → near-reciprocal. Equilibrium found with
`.op` (RISK 1 proved the fixed point is unique, so the DC solution *is* the relaxed state).

Per pattern: **FREE phase** (inputs clamped, output floats → v*), **NUDGE phase** (output
tied to the target through a weak conductance β → v^β). Local update for each conductance k:

    ΔG_k ∝ (1/β)[ (Δv_k*)² − (Δv_k^β)² ]      Δv_k = voltage across k

realised as a gate-voltage nudge. Each conductance updates only from the squared voltage
**across itself** in the two phases — no transpose, no backward pass. (The update arithmetic
runs on the controller — sanctioned for the principle demo; the all-transistor update cell is
the next build, not this milestone.)

Result (`ep_findings.png`):

| task | loss start → end | corners | learned outputs o* | targets |
|---|---|---|---|---|
| OR  | 0.0112 → 0.0049 | **4/4** | 0.467 / 0.516 / 0.516 / 0.570 | 0.4/0.6/0.6/0.6 |
| AND | 0.0088 → 0.0048 | **4/4** | 0.427 / 0.488 / 0.488 / 0.558 | 0.4/0.4/0.4/0.6 |

At init the free output is input-*independent* (≈0.488 for all patterns, weights ≈0 by
symmetry); training reorganises the conductances so o tracks the target logic. Converges by
~5 iterations.

**Decisive control — the EP signal is a true gradient, not the nudge dragging outputs:**
flipping the update sign turns descent into ascent. The loss *rises* (0.0112 → 0.0202) and
the network learns the exact **inverse** (anti-OR: 0.516/0.467/0.467/0.421, 0/4). The sign of
the local (Δv)² difference carries the correct gradient direction. This is the EP theorem
working in silicon.

## Limitations / honest framing (the real findings)
1. **Near-linear regime only — XOR does not train yet (attempted, `ep_xor.py`).** To keep the
   triode reciprocal (small Vds) the swing is small, so the neuron nonlinearity barely engages
   → only **linearly-separable** tasks are demonstrated (OR, AND). I built the nonlinear case:
   2→2→1 with **physically differential hidden units** (complement from swapped-gate mirror
   conductances, no behavioral sources) and tight diode clamps for in-range saturation, then
   swept sign/swing/clamp/η. **It does not converge.** The most telling run found the XOR
   partition transiently — {00,11} pushed one way, {01,10} the other (the one
   nonlinearly-separable split) — but with ~0.03 V amplitude, then stalled; other settings
   collapse to input-independent outputs. So the hidden nonlinearity is on the right track but
   far too weak: a symmetric diode clamp around a leak-pinned 0.5 operating point, held to a
   small swing for reciprocity, cannot form strong enough features. **This is the central open
   finding of the fork** — the in-range-nonlinearity vs small-Vds-reciprocity tension is real
   and currently binding. The lever is a stronger, asymmetric neuron (the backprop fork's
   ReLU-like keystone) that saturates within a Vds the triode can still tolerate — a real
   design subproject, not a tuning knob.
2. **Soft margins.** Outputs cluster near 0.5 (the leak pulls there and the swing is small) —
   correctly classified but thin margins. Tunable (weaker leak, larger β/η), not yet pushed.
3. **Update is in software** (the local rule, not a backward pass). The all-transistor
   (Δv)²-difference cell + drifting signed differential conductances are the next build.
4. **Reciprocity is approximate** (O(Vds) Jacobian asymmetry) and it trains anyway —
   consistent with EP's known robustness, and with "the math doesn't have to be exact as long
   as it trains."

## Follow-up — idea 2 validated: the reciprocal cell that unblocks depth (`rec3.cir`, `rec3b.cir`)
The binding constraint above (small-Vds reciprocity) is **removed** by building the conductance
as a **CMOS transmission gate** instead of a single triode. Correct reciprocity metric: the
energy needs Jacobian symmetry `∂I/∂Va + ∂I/∂Vb = 0`, which in common-mode/differential
coordinates is `∂I/∂CM = 0` (current independent of the terminals' common mode). RISK 2 held CM
fixed, so its "antisymmetry pass" never tested this; a single NMOS actually has
`∂I/∂CM = −KP(W/L)·Vds` (its source reference rides with CM). Two *identical* NMOS in
anti-parallel do **not** help (the MOS auto-selects the lower terminal as source → just 2× one
device). The fix is complementary: as CM rises an NMOS conductance falls and a PMOS rises;
with matched slopes `KP_n(W/L)_n = KP_p(W/L)_p` they cancel.

Measured (NMOS W=200/L=100 + PMOS W=250/L=100, `PSYN` = PMOS triode VTO=−0.2 KP=40u LAMBDA=0):

| Vds | single NMOS ∂I/∂CM | G swing over CM | **T-gate ∂I/∂CM** | G swing |
|---|---|---|---|---|
| 0.1 | −10 µA/V | 80% | **0.00 µA/V** | **0%** |
| 0.3 | −30 µA/V | 80% | **0.00 µA/V** | **0%** |
| 0.5 | −50 µA/V | 80% | **0.00 µA/V** | **0%** |

- **Reciprocity holds at full swing** (Vds = 0.5), not just small Vds → you can now run large
  signals → strong neuron nonlinearity → depth is no longer blocked by the synapse.
- **Programmable and still reciprocal:** one control Vc sets G linearly (60 → 100 → 140 µS over
  Vc = 0 → 0.4) with `∂I/∂CM` staying 0.00 — the canceling slopes are gate-independent.
- **Mismatch tolerant:** ±10% PMOS width mismatch leaves only ∓3 µA/V (10× better than the
  single NMOS, ~1.3% G variation vs 80%); residual scales linearly with mismatch and signed
  differential pairing should cancel the common part further.
- **Honest caveats:** the exact 0.00 is the idealized LEVEL-1 model (LAMBDA=0, no mobility
  degradation / body effect) — real silicon has higher-order CM terms this won't fully cancel;
  the cell costs a complementary device per conductance and needs a small negative gate range
  for the PMOS. Within the project's device-model fidelity, the cancellation is real.

**Consequence:** ideas 1 (one-port nonlinear neuron) + 2 (this cell) together remove both halves
of the XOR blocker — strong in-range nonlinearity *and* exact reciprocity at full swing. The
natural next experiment is to rerun XOR with T-gate synapses + a one-port neuron at full swing.
Figure: `ep_reciprocal_cell.png`.

## XOR redux with the unblocks — and the deeper wall it exposed (`ep_xor2.py`, `ep_active.py`)
Rebuilt the nonlinear net with T-gate synapses + one-port (clamp) neurons at full swing. **It
still does not train XOR**, and chasing why produced the most important finding of the fork.

- **Forward representability search (`ep_xor_search.py`):** over 300 random weight draws the
  **max XOR output-separation is 0.002** (and max single-hidden XOR-sep 0.006). The architecture
  *cannot represent XOR for any weights* — so this is not a learning-dynamics problem, it's a
  representational one.
- **Root cause:** a reciprocal conductance node computes a **normalized weighted average**
  `v = ΣGₖvₖ/ΣGₖ`, i.e. a *contraction toward the centroid*, not a weighted sum. The
  pre-activation never leaves the neuron's linear band, so the nonlinearity stays dormant and the
  whole net is effectively linear. **Reciprocity (idea 2) was necessary but not sufficient — the
  missing ingredient is gain.** (This is also why OR/AND had only soft margins, and it is the EP
  twin of the surrogate's "stacked lossy analog neurons destroy signal" depth limit.)
- **Fix identified and confirmed (`ep_active.py`, `ep_gain_diagnosis.png`):** in a Hopfield/EP
  network the *synapses* must be reciprocal but the *neurons* may be **active high-gain
  amplifiers** — gain in the neuron does **not** break the symmetric-synapse energy. Holding
  reciprocal averaging synapses fixed and sweeping neuron gain:

  | neuron gain | 1 | 2 | 4 | 8 | 16 |
  |---|---|---|---|---|---|
  | max XOR output-sep | **0.000** | 0.244 | 0.565 | 0.635 | **1.000** |

  At gain 1 (passive) separation is exactly 0 (the wall); at gain 16 the best draw is
  `o* = [1,0,0,1]` — **perfect XOR**. (Active neuron is a behavioral `tanh`/clip scaffold here,
  clearly labeled; the transistor version is the backprop fork's gain/keystone cell.)

**Unified takeaway for "trains in depth":** the buildable recipe is **reciprocal T-gate synapses
+ active high-gain saturating neurons**. Reciprocity (now solved) gives the valid energy and the
transport-free local update; neuron gain gives the feature formation and signal restoration that
passive crossbars lack. This is the standard Hopfield structure and it is the same active-gain
element the backprop fork already builds — so depth-capable analog EP is a *neuron-design*
problem, not a learning-rule problem. Next experiment: transistor high-gain neuron + real
two-phase EP training (not just representability) on XOR.

## Full transistor XOR training attempt — representable, but EP gradient is invalid (the real wall)
Built the validated recipe end-to-end (`ep_xor_train.py`): reciprocal T-gate synapses + **active
high-gain inverter neurons** (`neuron_char.cir`: low-Vt CMOS inverter, gain≈14 under load, clean
0–1 swing, complement output), in a **recurrent** 2-2-1 net with bidirectional weight-shared
hidden↔output coupling so the output nudge can propagate back (the EP signal path).

- **XOR is representable with the real transistor net.** Random weight search reaches output
  XOR-separation 0.817; e.g. `a_o = [0.073, 0.912, 0.874, 0.079]` — XOR. The device + recipe can
  express the function. `.op` converges (the high-gain recurrent net settles).
- **But two-phase EP does not train it** — it oscillates / collapses to a uniform mean-predictor
  (loss stalls ~0.08–0.12, never the ~0.06 XOR floor) across learning rates, momentum, seeds,
  nudge strengths, and late-weight averaging.
- **Root cause, established by a finite-difference gradient check (`ep_gradcheck.py`):** the EP
  gradient estimate barely correlates with the *true* loss gradient — cos(EP, FD) ≈ 0 (e.g.
  0.44, 0.02, −0.02, 0.01), and this holds for **both** the `(Δv)²` rule *and* the Hopfield
  activation-correlation rule, and at every neuron gain tested. The active inverter neuron makes
  the network **not a gradient system**: a conductance between an *activation* node `a_j` and a
  *state* node `u_i`, with an active neuron in the loop, has no energy whose minimum the two-phase
  difference can read. (The huge FD gradients, up to ~7, also show the high-gain recurrent
  equilibrium is near-singular.) The passive OR/AND net *was* a valid gradient system (cos would
  be high); adding active gain to get features **broke** the gradient validity.

**This is the precise statement of the reciprocity-vs-gain wall.** Two self-consistent regimes:
- **(A) passive resistive form** (synapses between state nodes, two-terminal neurons): valid
  gradient, `(Δv)²` EP works → but gain < 1, only linearly-separable tasks (OR/AND demonstrated).
- **(B) active-gain hybrid** (conductance synapses + amplifier neurons): represents XOR → but no
  valid energy, EP gradient invalid (FD-confirmed), does not train.

**The buildable path that should resolve it (next milestone, not yet built):** the proper
**Hopfield/current-mode** form — synapses inject current `T_ij·a_j` into `u_i` (a transconductor,
i.e. the backprop fork's trans2 cell) with **forward/backward transconductor pairs sharing the
weight** to make the coupling matrix symmetric, active gain neurons, and the **activation-
correlation update** `Δ ∝ a_i·a_j` (its cos-with-FD trended consistently positive in the check,
unlike `(Δv)²`). That form has a valid Hopfield energy *with* gain — the one regime that has both.
It reuses the backprop fork's transconductance synapse, symmetrized. **Status: XOR training in
ngspice is NOT yet achieved; the blocker is identified and FD-validated, and the next architecture
is specified.**

## RESOLVED — current-mode EP trains XOR in ngspice (`ep_cm.py`, `ep_deep.py`, `tc_cell.cir`)
The wall above was beaten. The fix is exactly idea 1: **current-mode synapses**. Found by using
the finite-difference gradient check (idea 9) as a design instrument — it isolated every bug.

**The diagnostic chain (each step a single decisive measurement):**
1. The active-neuron + *conductance* hybrid gave cos(EP,FD)≈0 → not a gradient system.
2. Dropping in an ideal *increasing* neuron + nudging the *state* node lifted cos to ~0.7–0.96 —
   so the original inverter neuron (g'<0) had **flipped the Lyapunov sign**, and the nudge must
   act on the neuron *state*, not its driven output. But conductance coupling still averages
   (contraction), so the gradient was noisy and XOR optimization collapsed to a mean-predictor.
3. Switching synapses to **current-mode** (a VCCS injecting `w·(a_j−0.5)` into the node, so the
   node integrates a true SUM and the leak sets the gain) gave **cos(corr,FD) = −0.96…−0.99**
   (sign = convention) and output swing −0.9…+2.4 (no compression). The standard **correlation**
   update `Δw ∝ (1/β)[φ_iφ_j |^β − φ_iφ_j |*]` is the right rule here (it's Hopfield/EP-standard).

**XOR trains, reliably, entirely in ngspice** (both relaxation phases are SPICE `.op`s; only the
local weight update runs on the controller):

| hidden width | seeds solving XOR (of 5) |
|---|---|
| 3 | 3/5 |
| 4 | 4/5 |
| **6** | **5/5** (three at loss = 0 exactly: `u_o=[0.15,0.85,0.85,0.15]`) |

Loss falls 0.58 → 0 with clean convergence (`ep_xor_trained.png`). **Width buys reliability** —
the surrogate's "spend transistors on width" reproduced for EP. This is the milestone: XOR, the
first nonlinearly-separable task, trained by two-phase EP in SPICE with a local update.

**The synapse is buildable (`tc_cell.cir`).** A 5-transistor OTA transconductor (our NNR/PNR
devices) gives output current `gm·(a_in−0.5)` into the node: signed (crosses zero at 0.5),
gm programmable by the weight/tail voltage (46→59 µS), and **high Zout** (current flat at −13.5 µA
across the output range = true current source). The current-mode synapse is realizable in the
device family; the differential w⁺/w⁻ pair symmetrizes the single-ended OTA.

**Scalability (`ep_deep.py`, layer-based [2,H,…,1]).** Single hidden layer is rock-solid
(cos −0.89, min −0.999). Depth generalizes structurally and *can* train — `[2,8,8,1]` reaches
4/4 XOR on some seeds — but **2 hidden layers is unreliable** (random-config cos collapses toward
0; ~1/4 seeds solve). This is the known EP-at-depth difficulty (the nudge weakens through layers),
and it is exactly what the depth-specific ideas target: **centered ±β nudging** (idea 5),
**orthogonal/isometric init + per-neuron normalization** (the surrogate's depth cure), and
**skip couplings** (idea 6). The method is right; depth needs those add-ons — the honest next step.

**Which of the 10 ideas carried it:** #1 current-mode (decisive architectural fix), #9 FD-fidelity
as a design instrument (caught every bug in one measurement each), #7 width/symmetry-breaking
(reliability), #2 shared-gate symmetric coupling (valid energy, transport-free). #5 centered
nudging and the isometry/skip ideas are the validated next levers for depth.

**Honest status:** XOR is trained by EP in ngspice with the scalable current-mode method, and the
synapse is a validated transistor cell — but the trained net still uses behavioral neurons and the
weight update runs on the controller. Remaining for "100% transistor, fully autonomous": integrate
the OTA synapse + a non-inverting transistor neuron into the trained net, and build the local
correlation-update cell (a multiplier + charge pump). All are validated-buildable components now.

## FULLY-TRANSISTOR EP trains XOR — zero behavioral sources (`ep_tcm.py`)
The behavioral cells were replaced with real transistors and the whole net trains XOR in ngspice.

- **Cells:** synapse = 5-transistor OTA transconductor (the validated `tc_cell.cir`), signed via a
  differential `w⁺/w⁻` OTA pair driven by the neuron's `a`/`ac`; neuron = 3-transistor diff-pair
  amplifier (`neuron2.cir`, gain ≈7, increasing, with complement); state node = a leak resistor;
  symmetric hidden↔output coupling shares the stored weight (transport-free). Weights are gate
  voltages; only the local correlation update runs on the controller.
- **Deck audit:** `grep` confirms **0 behavioral/dependent (B/E/G/H) sources** and **366 MOSFETs**
  (7 neurons×3 + 69 OTAs×5) — comparable to the backprop fork's 357, genuinely transistor-level.
- **The full-transistor net is a valid gradient system:** FD check gives **cos(EP,FD) = −0.83
  (min −0.94)** despite a real analog hurdle (see below).
- **It trains XOR and converges stably.** With a gentle optimizer (η=0.008, mom=0.3) seed 2 reaches
  **loss 3e-5, 4/4, `u_o=[0.432,0.565,0.561,0.432]`** = the XOR targets, and *stays* there.
  **All 5/5 seeds reach 4/4** (seed 0 hits loss = 0 exactly). A too-high learning rate oscillates
  (finds XOR early then drifts); the gentle rate converges monotonically. Figure: `ep_xor_transistors.png`.

**The analog hurdle, named honestly:** the single-ended OTA pair has a **systematic source/sink
offset** (the `w⁺` branch sinks via NMOS, `w⁻` sources via PMOS — different devices don't cancel),
so at zero weight the nodes sit at ~0.77, not 0.5. The working fix here is a **strong leak**
holding each state node near its reference so the differential signal survives; the cost is **thin
output margins** (u_o swings ~0.43–0.57 about the 0.5 threshold) and a too-high learning rate
oscillates (gentle η converges). The clean chip-level fix is a **fully-differential** state-node
pair (common-mode rejected) or a virtual-ground summing node — standard analog, the next refinement.

**This closes the milestone in the strong sense:** XOR — a nonlinearly-separable task — trained by
two-phase Equilibrium Propagation in a **366-MOSFET, zero-behavioral** ngspice netlist, with a
reciprocal/transport-free local update, FD-validated. The energy-based path is real for this device
all the way down to transistors.

## Refinement — fully-DIFFERENTIAL: simpler, more robust, more scalable (`ep_dtcm.py`)
The single-ended net's weakness was a **systematic offset** (w⁺ sinks via NMOS, w⁻ sources via
PMOS — they don't cancel, nodes sat at 0.77), which forced a strong leak, thin margins, and
learning-rate sensitivity, and which *accumulates with fan-in* (the scaling killer). Going
fully-differential fixes all of it at once.

- **The differential signal IS the signed representation** → the complement is free (no separate
  `ac` path), and common-mode/offset is **rejected**. The synapse becomes two **crossed NMOS diff
  pairs** (`dsyn`, 6T) — the mirror is gone. Validated (`dsyn_test.cir`): signed transfer, and
  **exactly 0 output at zero weight** (offset eliminated). State nodes use diode-PMOS loads that
  hold the common mode (the one new subtlety — the summed sink current droops the CM, so loads must
  source the fan-in current; sized once, the CM holds at ~0.49).
- **Fewer devices:** **218 MOSFETs** vs the single-ended **366** (~40% fewer), 0 behavioral sources.
- **Better-conditioned gradient:** FD check **cos(EP,FD) = +0.88 (min +0.86)** — tighter and more
  consistent than single-ended's 0.83.
- **Much bigger margins:** trained `dout = [−115, +118, +118, −115] mV` (±~5× the single-ended
  ±30 mV about threshold) — because there's no offset to fight, the signal swings freely.
- **Trains XOR reliably and scales with width:** stable 4/4 on **5/8 seeds at width 6, 7/8 at
  width 8** (width buys reliability, as the surrogate predicted). Figure: `ep_xor_differential.png`.

**Net effect of the refinement:** simpler synapse (6T vs 10T), ~40% fewer transistors, offset-free
(the property that matters for scale and mismatch-robustness), cleaner gradient, and ~5× margins —
all still zero-behavioral, all in ngspice. The remaining scale lever is **common-mode feedback**
(replacing the fixed diode load) so the operating point is fan-in-independent; and a cascode tail
would tighten the residual CM rejection (~26 mV). Both are standard analog, the next refinements.

## Multi-class + fan-in scaling: sum-mod-3 (`ep_mc.py`, flexible in N/C/width)
Extended the differential net to **C one-hot differential outputs** (prediction = argmax over the
C output differentials, per-output nudge toward a one-hot target). Task = **(sum of N bits) mod C**
— the multi-class cousin of parity (C=2 is parity). Code is parametric in **N inputs, C classes,
hidden width**; convergence sped up with a `.nodeset` (state nodes ≈0.5 → no gmin stepping, ~10×
faster `.op`); per-process file tags make concurrent sweeps safe.

- **Fan-in stays scalable:** each state node's diode load is sized to its **fan-in** (W ∝ fan-in),
  so the common mode is **fan-in-independent** — measured **CM = 0.498 at both N=3 (fan-in 7) and
  N=4 (fan-in 8)**. This is the mechanism that lets the differential design scale in input
  dimension without the CM drooping (the lever flagged in the previous refinement).
- **Multi-output gradient is valid:** FD check **cos(EP,FD) = +0.83 (min +0.62)** at N=3, C=3.
- **It learns the modular multi-class task:** sum-mod-3 reaches **acc 0.75–0.875 at N=3** and
  **0.875 at N=4** (16 patterns), vs **chance 0.33**. All zero-behavioral, argmax over 3
  differential outputs.
- **Honest limits:** it plateaus below 100% — the modular target is genuinely hard (e.g. `000` and
  `111` both map to class 0, a sharply non-monotonic "collapse the extremes" mapping), and the
  optimizer still oscillates (best > final). The surrogate's **competition / common-mode
  normalization** (the one shared amp) is **not yet added** — expected benefit is modest at C=3 but
  grows with C, so testing it wants larger C (the natural next step).

**Takeaway:** multi-class EP works on the differential silicon, and the fan-in-scaled loads keep the
common mode flat as N grows — so input dimension scales cleanly. The open levers are the
competition amp (calibration / large-C accuracy) and a better optimizer (centered nudging) for the
hard modular targets.

## Real dataset on a LIVE ngspice process — sklearn digits (`ng_live.py`, `ep_digits.py`)
Switched from per-iteration batch decks to a **persistent ngspice process driven over a pipe**
(`ngspice -p`): load the deck **once**, then stream `alter inputs → relax → read → alter target +
nudge → relax → read`, accumulate the local correlation update over the epoch, and `alter` the
weight sources once. This is the controller loop a real chip wants (present input, let it settle,
read, next) and the path to continual EP — no deck regeneration, no process respawn.

- **Driver (`ng_live.py`):** load-once + `alter`/`op`(or `tran`)/`wrdata` bursts, synchronized by an
  `echo @@K@@` sentinel; values returned via files. Two gotchas fixed: `.nodeset` (state ≈0.5 →
  no gmin stepping, ~10× faster `.op`) and **`destroy all` every step** (else ngspice accumulates a
  plot per op and bloats to GBs/slows to a crawl).
- **Data I/O solved:** sklearn digits, 8×8 → **4×4** (2×2 average-pool), C classes; each of the
  **16 pixels** is a differential input voltage, C one-hot differential outputs (argmax).
- **Throughput is fine:** one relax+read ≈ **18 ms**, a full 230-weight update (alter 460 sources)
  ≈ **17 ms**, so an 80-example epoch ≈ 3 s.
- **Fan-in test on real data:** the 16-pixel fan-in is the largest yet; the fan-in-scaled diode
  loads hold the output **CM at 0.493** — the scaling property holds on real inputs.
- **It trains end-to-end:** digit **0-vs-1** climbs from chance (0.50) to **~0.68**, all on the live
  transistor net, zero behavioral sources.

**Honest caveat:** ~0.7 is modest for near-separable 0/1 (a linear classifier gets ~99%). The
likely cause is the surrogate's prediction surfacing on real data — **independent one-hot outputs
are poorly calibrated for argmax** — plus full-batch-EP optimization headroom. The indicated fixes
are the **competition / common-mode normalization amp** (the surrogate's one shared amp) and a
better optimizer (online updates / centered nudging). This is exactly the competition experiment we
queued, now well-motivated by a real-data result.

### Local receptive fields (2×2 patches) — better accuracy AND smaller fan-in
The first layer was fully-connected (every hidden neuron from all 16 pixels, fan-in 19). Making the
first layer **local 2×2 patches** (each first-layer neuron reads only its 2×2 quadrant;
`PSIZE`/`PSTRIDE`/`FEAT` configurable) helps on both axes at once:

| first layer | hidden fan-in | weights | best test (0-vs-1) |
|---|---|---|---|
| fully-connected | 19 | ~230 | **~0.68** |
| **local 2×2** | **7** | **114** | **~0.825** |

A convolutional-style local receptive field — the right inductive bias for images — lifts accuracy
~0.68 → ~0.83 *and* cuts the first-layer fan-in (16→4 pixels) and weight count, which is exactly the
analog-friendly direction (smaller diode loads, less current to budget, the per-feature signal not
diluted across 16 inputs). (Weight *sharing* across patches — software "conv" — was rejected: it's
non-physical for analog and absent in cortex, where each column learns its own filters; local
connectivity is the half that's real.)

### Ablation — noise, leak, late-averaging, competition, normalization (digits 0-vs-1, local 2×2)
The training peaks then drifts (the late-weight overshoot), so each run reports three readouts:
**final** weights, **best-checkpoint** (peak val), and **late-average** (mean of late-epoch weights).

| config (seed 0) | final | best-ckpt | late-avg |
|---|---|---|---|
| baseline | 0.675 | 0.825 | 0.725 |
| + competition amp | 0.700 | 0.825 | 0.725 |
| + noise+leak (0.015/0.01) | 0.725 | 0.825 | 0.725 |
| + **strong noise+leak (0.05/0.02)** | 0.750 | **0.875** | **0.825** |
| + strong noise + **per-image norm** | 0.825 | **0.900** | 0.775 |

Interactions, the interesting part:
- **The small knobs don't change the *peak* (~0.82); they only change how well the *final* holds it.**
  best-checkpoint reliably captures the peak — the cheap, robust win.
- **Noise is qualitatively different at strength.** Weak noise just trims the drift; **strong noise
  (a) raises the peak** 0.825 → 0.875 (SGLD **escapes the soft attractor**) **and (b) flips the late
  curve from monotonic drift to oscillation**, so **late-averaging finally works** (0.725 → 0.825) —
  the OU/SGLD "average = posterior mean" prediction realized on the energy-based side, matching the
  backprop fork's noise+averaging robustness result. So **noise × averaging are synergistic**:
  averaging only pays off once there's enough noise to put the weights in the sampling regime.
- **Per-image contrast normalization** (stretch each image to full [0,1]) lifts the peak again,
  0.875 → **0.900**, and the final 0.750 → 0.825 — clean input conditioning, stacks with the rest.
- **Competition amp is marginal at C=2** (0.675 → 0.700), exactly as the surrogate predicted (benefit
  grows with C); its real test is larger C.

**Net progression on digits 0-vs-1:** fully-connected ~0.68 → local 2×2 ~0.83 → + strong-noise SGLD
0.875 → + per-image norm **0.90** — all on the live transistor net, zero behavioral sources, with the
readout (best-checkpoint or late-average-in-the-noisy-regime) doing the rest. Knobs are toggles in
`ep_digits.py` (`COMP`, `NOISE`, `LEAK`, `AVGFR`, `NORM`).

## PHYSICAL update cell — the learning is now silicon, not numpy (`gen_cell_scs.py`, Spectre)
The standing gap was that the weight *update* ran on the controller (read activations → numpy
correlation → write weight voltages). Built a transistor cell that computes and applies the EP
update itself, validated in **Spectre** (its convergence on the switch + high-impedance integrator
is solid where ngspice's was not — adopting it was the unblock).

**Cell:** Gilbert multiplier (`φ_i·φ_j`, validated 4-quadrant) → PMOS current mirrors → **chopper**
(NMOS pass gates, sign-flipped by the phase signal) → **differential weight cap** with **CMFB sinks**
(gate = sensed common mode → self-matches the source current, holds CM, lets the differential
integrate) → a CM-sense resistor that doubles as the weight **leak** (the prior). Phase/inputs are
controller voltages (allowed); the *update arithmetic* is the circuit.

**The decisive principle (`ep_cell1.cir`):** phase-gated sign-flipped integration is a built-in
chopper — net ΔVw was **+27.5 mV regardless of an injected multiplier offset of 0, 1 µA, or 5 µA**
(the free phase dips harder but the cycle-net is unchanged). The two-phase difference *is* the
offset canceller, provided the two phases are equal duration.

**Validated in transistors (Spectre):**
- **It integrates** — the weight cap accumulates across cycles (CM held by CMFB, no rail collapse).
- **Correct bidirectional sign** — nudge·>free· → ΔW>0; free·>nudge· → ΔW<0.
- **Offset cancels** — at zero signal (free=nudge) the drift is ~0.6 mV/cycle, **~15× smaller**
  than a typical signal step.
- **Correct increment** — per-cycle ΔW is **linear in `⟨φ_iφ_j⟩_nudge − ⟨φ_iφ_j⟩_free`** and
  crosses zero where they're equal: it physically computes the EP correlation gradient.

**Honest caveats:** validated as a single cell driven by external φ_i/φ_j/phase (closing the loop —
wiring it to real neurons + synapse so the weight learns itself in one `.tran` — is the next step);
residual offset (~0.6 mV/cyc, charge injection / CMFB asymmetry) would accumulate over very long
training (lock-in / dummy switches fix it); gain compresses at large steps; the leak sets the weight
dynamic range. But the core is real: **the local EP weight update is now a transistor circuit.**
Tooling: `gen_cell_scs.py` (Spectre deck gen), `specparse.py` (nutascii reader), `ep_cell1.cir`
(ngspice principle test).

## Do we need the Gilbert? No — a SINGLE NMOS suffices (`gen_nmos_cell.py`, Xyce)
The 7-transistor Gilbert computes the exact product `φ_i·φ_j`. But the resistive-EP update is a
SQUARE `(Δv)²`, and a single MOSFET *is* a squarer. Tested the update *rule* on the real XOR task
(swapping the primitive in the differential trainer, `ep_dtcm.py UPDATE=...`):

| update primitive | multiply cost | XOR seeds solved |
|---|---|---|
| `corr` = φ_i·φ_j (Gilbert) | 7 T | 8/8 |
| `quarter` = ¼[(a+b)²−(a−b)²] | 2 NMOS (exact product) | 8/8 |
| **`sqdiff` = (φ_i−φ_j)²** | **1 NMOS (ideal square)** | **5/5** |
| `sqsum` = (φ_i+φ_j)² | 1 NMOS | 5/5 |
| `rectdiff` (real rectified NMOS) | ~2 | 2/5 |
| `triode` (2-quadrant) | 1 | 1/5 |

So the square-law `(Δv)²` update trains XOR with a *single* squarer; the self-terms in the
expansion are benign. Honest nuance the table shows: a *real* NMOS squares only one polarity
(rectified) with a Vt offset, so the clean full-range version is a 2-NMOS pair.

**Physical cell validated in Xyce** — replaced the Gilbert with **one NMOS** (`Vgs = vg−vs = Δv`,
`I ~ (Δv−Vt)²`) feeding the same chopper + CMFB integrator:
- integrates: dW accumulates ~linearly, CM held by CMFB.
- correct bidirectional sign: nudge²>free² → +82 mV/cyc; free²>nudge² → −76 mV/cyc.
- offset cancels: zero-signal drift +0.7 mV/cyc, ~115× below the signal.

So the update *multiply* drops from 7 transistors to **1** (operating with Δv kept above Vt; a
2-NMOS pair covers the full range). Xyce ran the stiff switch+integrator deck cleanly (`.tran` with
`gmin`/`voltagelimiterflag` + balanced `.ic`/UIC, per SIMULATORS.md). Remaining: close the loop —
drive the actual XOR network's weights with these cells in one `.tran`. Caveat: `ep_dtcm.py prim()`
is a numpy *rule-screen* (behavioral); the *cell* (`nmos_cell.cir`) is the transistor proof.

## Files
- `gen_nmos_cell.py` / `nmos_cell.cir` — single-NMOS squarer update cell (Xyce-validated).
- `gen_cell_scs.py` / `ep_cell.scs` — transistor EP update cell, Gilbert version (Spectre); `specparse.py` — raw reader.
- `ng_live.py` — persistent interactive-ngspice driver (the requested live control loop).
- `ep_digits.py` — sklearn-digits EP on the live process (4×4, C classes, fan-in-scaled, batch EP).
- `ep_mc.py` — multi-class differential EP (flexible N/C/width; sum-mod-C; fan-in-scaled CM).
- `ep_dtcm.py` — fully-differential transistor EP (218 MOSFETs, offset-free, the refined design);
  `dsyn_test.cir` — differential synapse cell (offset-free validation); `ep_xor_differential.png`.
- `ep_tcm.py` — single-ended fully-transistor EP (trains XOR, 366 MOSFETs, 0 B-sources);
  `tc_cell.cir`, `neuron2.cir` — the validated synapse/neuron cells; `ep_xor_transistors.png` — result.
- `ep_cm.py` — behavioral current-mode EP (method validation); `ep_deep.py` — layer-based (depth).
- `tc_cell.cir` — transistor OTA transconductor synapse (validated); `ep_xor_trained.png` — learning curve.
- `ep_xor_train.py` — earlier T-gate + active neuron recurrent net (diagnostic); `neuron_char.cir` — neuron.
- `ep_gradcheck.py` — finite-difference validation of the EP gradient (the decisive diagnostic).
- `ep_xor2.py` — T-gate + one-port neuron nonlinear net; `ep_xor_search.py` — representability search.
- `ep_active.py` — neuron-gain diagnosis; `ep_gain_diagnosis.png` — the gain-vs-XOR-separation curve.
- `rec3.cir`, `rec3b.cir` — reciprocal-cell (T-gate) validation; `ep_reciprocal_cell.png`.
- `relax1.cir` — reciprocal relaxation network (RISK 1).
- `rec2.cir` — triode-conductance reciprocity + programmability sweep (RISK 2).
- `ep_train.py` — two-phase EP trainer (RISK 3); `ep_plot.py` — loss curves + truth tables.
- `ep_findings.png` — loss curves (OR, AND, wrong-sign control) + learned truth tables.
