# HAND-OFF A — Multi-class fully-transistor trainer in ngspice (extend the working XOR deck)

You are continuing a long-running project. This brief is self-contained; read it fully before acting.

## The mission
Build a neural-network trainer where **learning is physical / in-circuit** — gradient computation
AND weight update happen in the analog hardware, never offline. The controller may only present
inputs and targets as voltages. Everything must live in ONE monolithic ngspice deck with **zero
behavioral/dependent (B-) sources** — every operation transistor-level. Weights are stored as
capacitor gate voltages.

**Your specific goal for this fork:** extend the *already working* 2-input XOR trainer to a
**multi-class** classifier (C output classes) with a **competition/normalization stage**, in a
**shallow-and-wide** topology (one hidden layer, more hidden units). Then train a small multi-class
problem in ngspice and show it works for real.

## Who you're working with (match this)
Designs an analog NN accelerator chip. Socratic, highly technical, wants substance and honesty,
dislikes laziness, over-claiming, and preemptive declining. Standing license: **"the math doesn't
have to be exact, as long as it trains."** Always **build + run + verify before integrating**.
Own mistakes plainly without over-apologizing. Transistor count is the currency being spent —
be economical with it.

## Sandbox, tooling, and hard gotchas
- ngspice 42 (root). numpy/matplotlib available (`pip install --break-system-packages`).
- **Filesystem RESETS between sessions EXCEPT `/mnt/user-data/outputs/` which persists.** All prior
  work is there. **First action: `cp /mnt/user-data/outputs/* /home/claude/`.**
- Always wrap ngspice runs AND long python in `timeout`. The bash tool itself times out on
  multi-minute commands — break work into pieces; use `date +%s` for timing (the `time` builtin is
  unavailable).
- ngspice specifics: first netlist line is the TITLE (ignored); `let` rejects relational operators;
  the `.control` interpreter is **too slow for training loops** (do not train inside it — see method
  below); duplicate device names abort the run; PMOS bulk ties to VDD (models use GAMMA=0, no body
  effect); `wrdata` writes (t,val) pairs per vector so real values are at columns `[1::2]`; a `.tran`
  can abort mid-run yet still echo "OK" — always check the data length; `trnoise` on independent
  sources models device noise without a B-source.
- ngspice training time scales ~N^1.8 in MOSFET count (measured). **Start small, then scale.**

## What already exists and WORKS (your starting point)
In `/mnt/user-data/outputs/`:
- **`xor_full.cir`** — the working deterministic trainer deck. 2-input XOR, H=6 hidden ReLU units,
  1 output. ~357 MOSFETs, zero B-sources, trains 4/4 (frozen truth table) in ngspice. THIS IS YOUR
  TEMPLATE. Read it.
- **`gen_xor_full.py`** — the python generator that emits `xor_full.cir`. Currently hardcoded to
  2 inputs / 1 output. THIS IS WHAT YOU GENERALIZE.
- `gen_infer.py` — freezes learned weights and runs inference on the patterns. `score.py` — checks
  the truth table. `checkpoint_curve.py`, `plot_convergence.py` — validation curves.
- `surrogate.py`, `mc_experiment.py` — a fast, ngspice-calibrated NumPy surrogate. **Use it to
  pre-validate any architecture/config before spending SPICE time.** It is a study tool, not the
  learner.
- `SURROGATE_STUDY.md`, `README.md`, `digits_results.md` — full findings and honest caveats.

**Sanity step:** after copying, re-run the existing XOR train+infer to confirm the deck still
trains 4/4 before changing anything.

## How the working deck is built (device-level — read xor_full.cir for exact cards)
- **Signed synapse = triode PAIR.** `Mp` gate = the live weight `gp` (a capacitor voltage), `Mn`
  gate = fixed `VC = 1.8`. Both drains = the input voltage; sources = `colp` / `coln`. The signed
  product is the *difference* (gp − VC)·input. Synapse transconductance `Gs ≈ 1.2`; low synapse
  threshold `VTOSYN = 0.2` so hidden units don't die when a weight goes negative.
- **Keystone (current-difference → voltage).** Diode-NMOS sets a virtual ground at `colp` and at
  `coln`; current mirrors copy `I_colp` and `I_coln` to the output node; a resistor `R` to `vref`
  gives `vout = vref + R·(I_colp − I_coln)`. Hidden: `VREFH = 0.5`, `RTH = 8e3`. Output:
  `VREFO = 0.5`, `RTO = 7e3`. (NMIR diode device W is ~20µm.)
- **Activation (ReLU).** Mirror-based, `hv = VG0 + RH·I_relu`, `VG0 = 0.7`, `RH = 8e3`; it's a
  square-law (quadratic) ReLU that clamps near the rail. High gain sharpens class margins.
- **Backward = transpose-read (`BWD=trans2`), the part that took the most engineering:**
  - δ propagates by reading the *same* weights backward (transpose). δ1_j = δ2·w2_j·ReLU′(z1_j).
  - **Differential error encoding** `d2e_p = 1.2 + δ`, `d2e_n = 1.2 − δ` (a mirror copy with
    output/target roles swapped) — required because a single-ended error drain cut off at the
    (1,0) corner and deadlocked.
  - The error node is **buffered by matched low-Vt source followers** (the transpose drains were
    loading it and causing runaway).
  - **Comparator ReLU′** (z1 vs a 0.5 threshold, near-rail output) gates a **widened stacked OTA
    tail** (a z1-gated tail starved near-threshold units via its (z1−Vth)² weighting).
- **Weight update = OTA.** A transconductor charges/discharges each weight cap by the local
  error·input product (the physical gradient). Continuous-time, in-circuit.

## How training is run in ngspice (the method — not the .control loop)
The generator writes a FLAT deck where each training pattern occupies a time "slot"; inputs and
targets are presented as time-varying sources (PWL/pulse) over a long `.tran`. The forward path,
the transpose backward, and the OTA weight update all run **continuously** as the analog dynamics
evolve — the caps integrate the gradient over the run. `wrdata` logs the cap voltages over time.
Then `gen_infer.py` freezes the final weights and runs inference; `score.py` checks accuracy.
Honest known caveats (documented in README.md): the XOR (1,0) corner trains soft (~0.60); the
learning curve is non-monotonic (it finds a crisp solution early, then drifts to a softer fixed
point — early-stopping / late-weight averaging helps).

## What the surrogate already established (decides your design — see SURROGATE_STUDY.md)
- **A competition stage is mandatory for many classes.** One-vs-all (independent outputs, no
  competition) collapses to chance as classes grow — on real sklearn digits it hit ~10% (= chance)
  at 10 classes, while competition got ~69%. Do NOT build one-vs-all.
- **Best buildable competition: power normalization** `p_c = relu(z_c)^n / Σ_k relu(z_k)^n` with
  n ≈ 4–6, reaching ~75% on a hard 8-class task vs softmax's ~78% and divisive's (n=1) 67%. It is
  all-above-threshold: `z^n` is cascaded square-law MOS stages (z² = one stage, z⁴ = (z²)², z⁶ =
  z²·z⁴) feeding a divisive divide. Divisive (n=1) is the simplest fallback. (If exact softmax is
  ever wanted: a small subthreshold shared-tail differential pair at the OUTPUT ONLY gives true
  exp/Σexp — localized cost, C transistors.)
- **The error rule is δ_c = p_c − t_c** on the normalized outputs — the SAME backward you already
  have, just per class. The hidden δ sums over classes automatically via KCL at each transpose
  column (each hidden unit's transpose reads all C output weights × their δ_c).
- **Shallow-and-wide is the right design.** (Depth is a separate hard problem — see Finding 5 — and
  is the subject of the *other* fork. Don't pursue depth here.) Width buys accuracy and robustness.

## Build plan (concrete)
1. Copy files; re-run XOR to confirm it still trains (sanity).
2. **Generalize `gen_xor_full.py`** from hardcoded 2-in/1-out to parametric (n_inputs, n_hidden,
   n_classes). Keep the device cards and the trans2 backward intact.
3. **Replicate** the output keystone C times (one per class) and the differential error block C
   times. Present one-hot targets as C voltage sources over the slots.
4. **Build the competition cell (the one genuinely new circuit).** Sum the C output currents on a
   shared node (KCL); for divisive, divide each by the sum (current-mode/translinear divider, or a
   shunt encoding 1/Σ); for power-norm, insert square-law stage(s) before the divide. **Validate
   this cell on a standalone test deck first** (feed known currents → check it normalizes).
5. **Wire the hidden transpose to all C errors** — verify each hidden unit's transpose column reads
   every output weight × its δ_c, so the class-sum happens by KCL.
6. Apply the **shared-negative-reference cleanup**: the VC/`coln` reference side is identical across
   all neurons in a layer → compute once per layer (≈ halves synapse transistor count).
7. **Start SMALL**: 3 classes, 3–4 inputs, ~8 hidden. Pre-validate this exact config in
   `mc_experiment.py` first (fast), then build it in SPICE.
8. **Train in ngspice** (long `.tran` over slots), freeze weights, score the multi-class truth
   table. **Success = frozen accuracy clearly above chance (1/C)**, classes separating, with the
   competition stage shown necessary (compare to no-competition ≈ chance). Then scale up
   classes/width as the N^1.8 budget allows.

## Definition of done for this fork
A single monolithic ngspice deck, zero B-sources, that trains a ≥3-class problem fully in-circuit
(physical gradient + physical weight update), reaching frozen accuracy well above chance, with the
competition stage demonstrated to be necessary. Report honestly (soft corners, non-monotonic curves,
seed variance are all fine to state). Cross-check the final number against the surrogate.
