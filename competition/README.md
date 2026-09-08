# Circuit-learning competition — runner specification

Submit a circuit; the organizer supplies examples, runs an open source transient
simulator, and measures classification accuracy and electrical energy. This is the
circuit-in, score-out runner used by the [introductory tutorial](../README.md).
The examples learn in the circuit. See the [v0 rules](RULES.md),
[leaderboard](LEADERBOARD.md), and [submission instructions](../submissions/README.md).
Submissions are reviewed and rerun by the organizer; there is no automatic upload service.

Use a [named task](tasks/README.md), such as `--task mnist017-v0`,
to select the full configuration and reject mismatched data or simulator versions.

## Run it

Requirements: Python 3.9+, NumPy, and ngspice or Xyce. No proprietary simulator is needed.
From the repository root:

```bash
python3 competition/prepare_toy.py /tmp/competition-blobs.npz
python3 competition/runner.py competition/examples/blobs.cir \
  /tmp/competition-blobs.npz --epochs 10 --startup 0.02 --output /tmp/competition-ngspice
python3 competition/runner.py competition/examples/blobs.cir \
  /tmp/competition-blobs.npz --epochs 10 --startup 0.02 --simulator xyce --output /tmp/competition-xyce
python3 -m unittest discover -s competition -p 'test_*.py' -v
```

`blobs.cir` is the small delta-rule learner. `mnist_017.cir` expands it to 16 inputs
and three MNIST classes. Both are actual circuit submissions generated without
reading training data. See the tutorial for MNIST download/preparation and the
[measured results](../docs/TUTORIAL_RESULTS.md). `constant_zero.cir` is retained
only as a wiring/energy sanity check; it always predicts zero and does not learn.

If Xyce is outside PATH, supply its executable explicitly:

```bash
python3 competition/runner.py competition/examples/blobs.cir \
  /tmp/competition-blobs.npz --epochs 10 --startup 0.02 --simulator xyce \
  --binary /path/to/Xyce --output /tmp/competition-xyce-custom
```

Use `XYCE_BINARY=/path/to/Xyce` to include that build in tests. Follow your Xyce
installation's instructions for any required shared-library search path.
The simulator integrations were tested here with ngspice 46 and Xyce's local
7.10 development build. Xyce is an exploratory cross-check; v0 scoring uses the
exact ngspice 46 image identified in [release.json](release.json).

Each successful run creates a new output directory with `report.json`, `harness.cir`,
and `simulator.log`. Add `--keep-trace` to retain `trace.txt.gz`; otherwise the raw
trace is temporary. Parsing and energy integration use bounded batches, not a
full in-memory text load. The report records dataset, circuit and harness
SHA-256 hashes, simulator version, schedule, accuracy, confusion matrix, component
counts, and energy by phase and source. Never publish a private evaluation harness:
it contains the evaluation input images even though it omits their labels.

`dataset_sha256` identifies array contents, not ZIP archive bytes. Its versioned
format hashes the four names and shapes in fixed order plus little-endian float64
inputs and int64 labels. Archive compression/metadata and integer storage width
do not affect it. `dataset_archive_sha256` is retained for file-level diagnostics.

The default simulator timeout is 1,800 wall-clock seconds; `--timeout` changes this
local limit, not the simulated schedule. On an abort, timeout, or invalid trace,
the runner exits nonzero and keeps `harness.cir`, `simulator.log`, `failure.json`,
and `trace.partial.txt` if one exists. A failed run never produces a score report.
This includes ngspice failures that print an abort but exit with status zero.

## Circuit interface

The submission is a flat list of R, C, M and D devices, optionally with full-line
`*` or trailing `;` comments. Device syntax is ASCII; comments may contain UTF-8
text such as µF or kΩ. `gnd` is accepted as an alias for `0`. The runner wraps it
in a subcircuit; do not include a SPICE title line or `.end`. No submitted models,
sources, behavioral expressions, initial conditions, `.include`, `.control`,
subcircuits, or commands are accepted. Internal nodes must start `n_`.
MOS multipliers such as `M=2` are not part of this format; use separate devices.
Check syntax without simulation using `python competition/validate.py your.cir`.

| Pins | Harness behavior |
| --- | --- |
| `vdd`, `0` | 3 V supply and ground; supply ramps from zero during startup |
| `vref`, `vweight`, `verror`, `voffset`, `vbias` | Metered bias sources: 0.5, 1.8, 1.2, 1.9, 1.7 V respectively |
| `reset` | Rises to 3 V during startup, then falls to 0 V before training |
| `in0`…`in15` | Row-major 4×4 image; block-average grayscale mapped to 0.3–1.7 V |
| `target0`…`target9` | One-hot 1/0 V training target; all zero during testing |
| `learn` | 3 V for training, 0 V for testing |
| `clock` | One 3 V pulse per example, high during first half of the slot |
| `out0`…`out9` | Analog scores; highest voltage at 90% of the slot wins |

Each output is loaded with 1 pF in parallel with 1 GΩ. Ties within 1 µV are
invalid and count as wrong; confusion-matrix column 10 contains invalid predictions.
There is no Python readout fitting or external weight update. Ground is global;
all other internal nodes are scoped by the wrapper.

Component syntax:

```spice
Rexample in0 n_sum 10k
Cexample n_sum 0 1p
Mexample out0 n_sum 0 0 nch W=10u L=1u
Dexample n_sum 0 diode
```

Only the organizer's `nch`, `pch`, `syn`, and `diode` models are available. `syn` is
the low-threshold NMOS used in the baseline's input/weight cells. Values must
be positive literals: R in [1 Ω, 1 TΩ], C in [0.1 fF, 1 mF], and MOS W/L
in [1 µm, 10 mm]. Maximum 20,000 components and 2 MB per submission. These broad
prototype bounds are not manufacturing constraints or a credible chip-area budget.

Protocol: `mnist4-circuit-learning-v0`. The named MNIST tasks fix startup to 20 ms,
training to 20 passes, each example to 1 ms, maximum simulation timestep to
10 µs, and temperature to 27 °C. All sources and capacitor states start at zero;
startup energy is included. Inputs/targets transition over 1% of a slot.
The reset pin permits circuits to initialize their weight capacitors by drawing
metered energy from the bias rails; free precharged initial conditions are not accepted.
Training uses shuffled passes, followed by shuffled test examples in the
**same transient**, with stored state preserved. Test targets ramp to zero over
the first 1% of the first test slot. Circuits must use `learn` to gate updates;
the harness cannot guarantee a submitted circuit actually stops learning.
Without a named task, custom-run defaults are one training pass and 1 ms startup.
`--epochs 0` supports untrained controls, but is not a ranked track.
Changing timing, model, dataset, seed, epochs or simulator changes the benchmark
configuration: results from different configurations must not share a ranking.

The tutorial flags specify reproducible experiments, not minimum requirements for
learning: blobs can already learn with fewer passes. Twenty milliseconds of startup
gives the reset switches more initialization time; ten or twenty epochs give the
chosen learner repeated training opportunities. Do not omit these flags when
comparing to the published tutorial scores.

The v0 harness uses ngspice's behavioral `pwl(time, ...)` to supply the same voltage
waveforms, with series 0 V current meters. Native PULSE sources force the clock
and startup ramp boundaries. This avoids a native voltage-source PWL breakpoint
failure near four seconds in ngspice 46; removing redundant flat points alone was
not sufficient. Behavioral sources are allowed **only in the organizer's harness**,
not in submissions, and perform no learning or classification. Xyce continues to
use its native PWL voltage sources. The disconnected startup timing source has
zero current and delivers no energy to the circuit.

The runner also removes redundant flat points, saves only needed vectors, and writes
higher-precision ngspice traces. A varying-input learner regression crosses four
seconds without changing the 10 µs maximum timestep. A full ten-digit run also
completes its 5.32-second schedule. See the [run records](../docs/TUTORIAL_RESULTS.md)
for identifiers; previous draft harness hashes necessarily differ.

## Running other people's circuits

Use the [isolated Docker backend](container/README.md), not your unrestricted local
simulator, for third-party submissions. `--docker-image` exposes only the generated
simulator work directory, with no network, a non-root user, and resource limits.
The Python runner still needs to be trusted; do not substitute a PR's modified
runner or execute its generator during organizer scoring. Use a disposable machine
without credentials for additional separation.

## Energy definition

For every externally driven pin k, delivered instantaneous power is
`p_k(t) = -v_k(t) * i(V_k)(t)`, using SPICE's source-current sign convention.
We integrate **each source separately**, then sum:

`E_delivered = sum_k integral max(p_k(t), 0) dt`.

We also report returned and signed net energy. Returned energy does not cancel
energy drawn from another source or at another time in the primary metric. A
future energy-recovery track would need an explicit physical recovery model.
Power is integrated using actual, nonuniform simulator timestamps, interpolated
phase boundaries and exact zero crossings of the piecewise-linear power trace.
Finite trace resolution remains a source of numerical error.

Report startup energy, training energy, total inference energy, inference J/image,
and total run energy. Count power delivered through image, label and control pins,
as well as VDD: counting only VDD lets input-powered circuits appear free.
Transitions, idle periods and the output load are included in their phase.
This is **simulated energy at the circuit boundary**, not computer electricity
used to run SPICE. Ideal external voltage drivers and score readout have no modeled
internal loss; their DAC/ADC/controller costs are outside this metric.

The Level-1 MOS models have explicit capacitance parameters but remain toy models.
They do not establish realistic layout parasitics, device leakage, area, process
variation, or fabricated-chip energy. Before claiming physical efficiency, select
and validate a redistributable technology model plus geometry/parasitic rules.

## Rules and result verification

The [v0 rules](RULES.md) fix the learning task, budgets, ranking, licensing,
and failure policy. The [organizer workflow](MAINTAINERS.md) describes independent
reruns and recording results. Public evaluation labels are a deliberate limitation:
keeping labels out of the SPICE harness does not make MNIST a secret test set.
Future tasks or more realistic device models need a new version and separate scores.
References: [ngspice documentation](https://ngspice.sourceforge.io/docs.html),
[Xyce](https://xyce.sandia.gov/),
[Xyce documentation](https://xyce.sandia.gov/documentation-tutorials/), and the
repository's [earlier simulator comparison](../docs/SIMULATORS.md).
