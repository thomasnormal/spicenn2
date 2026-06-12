# Sparse-neuron digit classifier, trained in-circuit (sklearn digits, 4x4)

Real sklearn digits, 8x8 → **4x4** (2x2 avg-pool), 2–3 labels, trained fully in-circuit in
ngspice (physical gradient + OTA weight update), zero B-sources.

## Sparse 2x2 receptive fields (the requested inductive bias)
Each first-layer neuron is wired to **exactly one 2x2 input patch** (4 pixels) + bias, not all
16 inputs — verified in the netlist (neuron 1 → synapses on inputs {1,2,5,6} only). 4
non-overlapping patches × 2 filters = 8 local neurons, then fully-connected to the C classes.
~710 MOSFETs (the dense first layer would add ~190 more → sparsity is ~20% fewer devices *and*
the right prior). `gen_mc.py CONN=patch2x2` + `DATAFILE=...`; `gen_mc_infer.py` mirrors both.

## Reliability fix: per-image contrast normalization (the key result)
First runs were a lottery — some label pairs/seeds collapsed to predicting one class (digit "8"
has more ink than "3" → its activations dominate → output biases to it). **Per-image min–max
normalization** (`NORM=perimage`, standard vision preprocessing) removes that class-magnitude
bias and kills the collapse:

| task | before norm | **with per-image norm** |
|---|---|---|
| 0 vs 1 | 89–92% (worked) | 90–93% |
| 3 vs 8 | 50% (collapsed) | **91–92%, both seeds, no noise** |
| 0,6,9 (3-class) | 33% (collapsed) | **84–88%, both seeds (+ weight noise)** |

(batch trainer, real ngspice infer, chance = 100/C). This made the multi-class digit case
**reliable** — the recurring output-collapse lottery is removed at the data level.

## Runtime data streaming (no data baked into the netlist)
`stream_mc.py` drives ngspice interactively (`-p`): inputs/targets are DC sources that the
controller `alter`s **per example** while advancing the continuous, state-preserving simulation
one slot at a time. The dataset lives entirely controller-side (`example(k)`), so any dataset of
any size streams in without touching the netlist. Demonstrated: streamed (3,8) reaches **93.8%**.

**Streaming reliability — RESOLVED (it was a simulation artifact).** Streaming first looked like
a lottery (≈60–94%) vs batch's tight 91–92%. The cause was NOT the abrupt DC step — it was
**coarse timesteps** (~3 steps/slot) leaving the forward unsettled when the update OTA integrated.
Resolving the per-slot settling (DTFRAC=0.1, ~10 steps/slot) makes abrupt-step streaming reliable:
(3,8) hits **90 / 95 / 97.5%** across seeds, matching batch. Two notes: (1) *ramping* the inputs
(DAC-slew, which a real controller can do) did NOT help and even hurt — a naive ramp also blurs
the one-hot target during the transition; the fix is settling time, not smoothing. (2) On real
**continuous-time** silicon there is no discrete timestep, so a circuit settles per example
naturally — meaning this gap was largely a discretization artifact of the simulator, and batch vs
streaming should be equivalent on hardware. Streaming is now both demonstrated AND reliable.

## Bottom line
- **Sparse 2x2-patch, in-circuit digit classifier: reliable ~90% on 2-class, ~85% on 3-class**
  (batch + per-image normalization). A genuine tiny analog conv-net trained on-chip.
- **Runtime streaming demonstrated** (arbitrary dataset, nothing baked) — works to ~94%, but
  not yet as reliable as batch.
- The one new general lever: **per-image normalization** removes the multi-class collapse
  lottery — the cleanest fix yet for the calibration problem that's recurred throughout.
