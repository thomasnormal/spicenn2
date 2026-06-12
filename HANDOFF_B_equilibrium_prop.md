# HAND-OFF B — Equilibrium Propagation prototype in ngspice (the energy-based fork)

You are continuing a long-running project. This brief is self-contained; read it fully before acting.

## The mission
Build a neural-network trainer where **learning is physical / in-circuit**. The controller may only
present inputs and targets as voltages. This fork explores a *fundamentally different* learning
mechanism from backprop: **Equilibrium Propagation (EP)** — let the network be a real physical
system that relaxes to an energy minimum, and learn from purely local rules. If it works, three
analog headaches vanish at once: the explicit backward pass, **weight transport** (physically
reading Wᵀ), and the exploding/vanishing-gradient problem of depth — all dissolve into physics.

**Your specific goal for this fork:** demonstrate in ngspice that (a) a small *reciprocal*
relaxation network settles to a stable equilibrium, (b) a programmable reciprocal conductance
element works, and (c) a two-phase EP local update actually trains a toy task. That milestone
decides whether the energy-based fork is real for this device.

## Who you're working with (match this)
Designs an analog NN accelerator chip. Socratic, highly technical, wants substance and honesty,
dislikes laziness, over-claiming, and preemptive declining. Standing license: **"the math doesn't
have to be exact, as long as it trains."** Always **build + run + verify before integrating**. Own
mistakes plainly without over-apologizing. This path is the more speculative/elegant one — be
honest about what does and doesn't converge.

## Sandbox, tooling, and hard gotchas
- ngspice 42 (root). numpy/matplotlib (`pip install --break-system-packages`).
- **Filesystem RESETS between sessions EXCEPT `/mnt/user-data/outputs/` which persists.**
  **First action: `cp /mnt/user-data/outputs/* /home/claude/`** (for device models, the surrogate,
  and the project docs as reference).
- Always `timeout` ngspice and long python. The bash tool times out on multi-minute commands —
  break up; use `date +%s` (no `time` builtin).
- ngspice: first line = TITLE; `let` rejects relational ops; `.control` interpreter is slow;
  duplicate device names abort; PMOS bulk→VDD (GAMMA=0); `wrdata` writes (t,val) pairs (values at
  `[1::2]`); `.tran` can abort yet echo OK (check data length); `trnoise` on independent sources
  models device noise.

## Context from the project (read for device models + constraints)
In `/mnt/user-data/outputs/`: `xor_full.cir` (a working *backprop* trainer — the OTHER fork; NOT a
template here because it's non-reciprocal, but its MOSFET model cards and structure are reference),
`SURROGATE_STUDY.md` / `README.md` (project context, device constants, the user's hard constraints),
`surrogate.py` / `mc_experiment.py` (the backprop surrogate — note: it CANNOT model EP, because it
models the non-reciprocal transconductance synapse; that is exactly why EP must be tested in SPICE).

## THE CRUX — read this before designing anything
EP requires the network dynamics to be a **gradient flow of an energy**: `dx/dt = −∂E/∂x`. That
exists ONLY if the network is **reciprocal** — the coupling i→j equals j→i (a symmetric conductance
matrix) — giving a Lyapunov energy the circuit physically minimizes.

- A **resistive/memristive crossbar has this for free**: a two-terminal resistor between nodes i,j
  is inherently symmetric, so the conductance matrix is symmetric and the network minimizes a
  well-defined co-content energy. Essentially all analog EP/Hopfield hardware is built this way.
- **The existing trans2 synapse is a TRANSCONDUCTANCE** (a gate voltage sets a one-directional
  drain current) — it is **NOT reciprocal**, so there is no clean energy and EP does not apply to
  it. **This is why this fork is a different topology and cannot reuse the backprop deck or the
  existing surrogate.**

**The reciprocal element (concrete, buildable with our device family):**
- A **MOSFET in TRIODE is an approximately symmetric drain-source conductance** controlled by its
  gate. Use it as a **programmable bidirectional conductance**: gate voltage = stored weight,
  drain-source = the reciprocal conductance G(gate). (Same triode device as the backprop synapse,
  but used as a symmetric conductance, NOT a multiplier.) Keep Vds small to stay symmetric
  (channel-length modulation breaks reciprocity at large Vds).
- **Signed weights:** conductances are positive. Use a **differential pair** G+ and G− per weight;
  effective weight = G+ − G− (two triode MOSFETs per signed weight). Same signed trick as the
  backprop synapse, but as conductances.
- **Neurons:** nonlinear clamping elements at the nodes (an activation φ) — e.g. diode clamps or a
  saturating transconductance. The energy includes the neuron nonlinearity term.

## EP mechanics (what to implement)
Energy (co-content) of the resistive net with nonlinear neurons, Hopfield-style:
`E = ½ Σ_ij G_ij (v_i − v_j)² + Σ_i (neuron term)`. The circuit relaxing IS minimizing E.

1. **Free phase:** clamp the inputs (as voltages); let the network RELAX to equilibrium `v*` (run
   `.tran` until it settles). Outputs float.
2. **Nudged phase:** weakly clamp the outputs toward the target — connect each output to its target
   voltage through a *small* conductance β (the nudge strength); relax again to `v^β`.
3. **Local update (the whole point — no transpose, no backward pass):**
   `ΔG_ij ∝ (1/β)[ (v_i^β − v_j^β)² − (v_i* − v_j*)² ]`.
   Purely local: each conductance updates from the difference of (the voltage across it)² between
   the two phases. The two-phase *difference* cancels the non-gradient part of ∂E/∂G, leaving the
   loss gradient. Realize it as a cell that squares each endpoint Δv (square-law MOS), takes the
   two-phase difference, and nudges the stored gate voltage (the conductance). **Design this cell
   only after the relaxation is shown to work.**

Note on error neurons: EP does NOT need explicit error neurons (those are an artifact of predictive
coding's specific prediction-error energy). With a symmetric energy, `−∂E/∂x = −Gx − b + φ′(x)` and
`−Gx` is literally the synaptic current the crossbar already sums — the neuron just integrates its
input. The price you pay instead is the **two-phase nudge** (the free-vs-nudged contrast is what
makes the local update equal the loss gradient; a single clamped phase would just shrink E
trivially).

## Build plan — start TINY, de-risk in this exact order
1. Copy files. Read `SURROGATE_STUDY.md` / `README.md` for device cards and the user's constraints.
2. **RISK 1 (cheapest, most decisive): does a small reciprocal network RELAX to a stable
   equilibrium?** Build a tiny reciprocal crossbar (e.g. 3 inputs, 2 hidden, 1 output, fully
   connected symmetric conductances) with nonlinear neurons; clamp inputs; `.tran`; confirm it
   **settles to a unique stable fixed point** (no oscillation/runaway). If it doesn't settle, the
   approach needs stabilization (damping / a convex-enough energy) before anything else.
3. **RISK 2: does the programmable triode conductance behave** — symmetric, stable, settable by a
   stored gate voltage? Test the element standalone (apply Δv, measure current both directions,
   sweep the gate; confirm I(i→j) ≈ −I(j→i)).
4. **RISK 3: does the two-phase EP update TRAIN a toy task?** Implement free phase + nudged phase +
   the local (Δv)²-difference update; train something trivial (XOR as a 2-class problem, or a
   3-point linearly-separable toy). **Success = the loss decreases and the task is learned.** You
   may first verify the PRINCIPLE with a minimal/coarse update mechanism before building the full
   transistor-level update cell.
5. Only after the principle works in-circuit, design the clean fully-transistor local-update cell
   and the signed differential conductances.

## Definition of done for this fork
Show in ngspice that: (a) a small reciprocal relaxation network settles to a stable equilibrium,
(b) the programmable conductance element is reciprocal and settable, and (c) the two-phase EP local
update reduces a loss / trains a toy task — all in-circuit, no transpose, no backward pass. That is
the decisive milestone: it tells us whether the energy-based path (the one that makes the backward
pass, weight transport, and the depth gradient problem all disappear into physics) is real for this
device. Don't over-build before the relaxation + update principle is demonstrated. If it works,
it's a major result — say so, with the convergence/stability evidence.

## Honest framing to keep
This is the speculative, elegant path. The first milestone is modest but pivotal. Likely failure
modes to watch and report: the relaxation may not converge (stability), the triode may be too
non-reciprocal (Vds-dependent), the nudge β trades gradient bias vs noise, and signed differential
conductances may drift. Any of these is a real finding. Cross-pollinate with the backprop fork
(`HANDOFF_A`) — they share device models and the project's constraints.
