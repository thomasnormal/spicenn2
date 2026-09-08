# Circuit-learning competition — draft runner

Submit a circuit; the organizer supplies examples, runs an open source transient
simulator, and measures classification accuracy and electrical energy. This is the
circuit-in, score-out runner used by the [introductory tutorial](../README.md).
The examples learn in the circuit. Final competition settings and hosted scoring
are not yet available; the protocol below is a development benchmark.

## Run it

Requirements: Python 3.9+, NumPy, and ngspice or Xyce. No proprietary simulator is needed.
From the repository root:

```bash
python3 competition/prepare_toy.py /tmp/competition-blobs.npz
python3 competition/runner.py competition/examples/blobs.cir \
  /tmp/competition-blobs.npz --epochs 10 --startup 0.02 --output /tmp/competition-ngspice
python3 competition/runner.py competition/examples/blobs.cir \
  /tmp/competition-blobs.npz --epochs 10 --startup 0.02 --simulator xyce --output /tmp/competition-xyce
python3 -m unittest competition.test_runner -v
```

`blobs.cir` is the small delta-rule learner. `mnist_017.cir` expands it to 16 inputs
and three MNIST classes. Both are actual circuit submissions generated without
reading training data. See the tutorial for MNIST download/preparation and the
[measured results](../docs/TUTORIAL_RESULTS.md). `constant_zero.cir` is retained
only as a wiring/energy sanity check; it always predicts zero and does not learn.

The local Xyce installation is outside PATH. On this machine use:

```bash
LD_LIBRARY_PATH=/home/thomas-ahle/xyce_deps/lib \
python3 competition/runner.py competition/examples/blobs.cir \
  /tmp/competition-blobs.npz --epochs 10 --startup 0.02 --simulator xyce \
  --binary /home/thomas-ahle/Xyce/build/src/Xyce --output /tmp/competition-xyce-local
```

Use `XYCE_BINARY` with the same path and library environment to include Xyce in tests.
The simulator integrations were tested here with ngspice 46 and Xyce's local
7.10 development build. These are recorded versions, not pinned release artifacts.

Each run creates a new output directory with `report.json`, `harness.cir`,
`trace.txt`, and `simulator.log`. The report records dataset, circuit and harness
SHA-256 hashes, simulator version, schedule, accuracy, confusion matrix, component
counts, and energy by phase and source. Never publish a private evaluation harness:
it contains the evaluation input images even though it omits their labels.

## Draft interface

The submission is a flat ASCII list of R, C, M and D devices, optionally with
full-line `*` comments. The runner wraps it in a subcircuit. No submitted models,
sources, behavioral expressions, initial conditions, `.include`, `.control`,
subcircuits, or commands are accepted. Internal nodes must start `n_`.

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

Protocol: `mnist4-circuit-learning-draft2`. Startup defaults to 1 ms (the learning
tutorials explicitly use 20 ms), each example to 1 ms, maximum simulation timestep to
10 µs, temperature to 27 °C. All sources and capacitor states start at zero;
startup energy is included. Inputs/targets transition over 1% of a slot.
The reset pin permits circuits to initialize their weight capacitors by drawing
metered energy from the bias rails; free precharged initial conditions are not accepted.
Training defaults to one shuffled pass, followed by shuffled test examples in the
**same transient**, with stored state preserved. Test targets ramp to zero over
the first 1% of the first test slot. Circuits must use `learn` to gate updates;
the harness cannot guarantee a submitted circuit actually stops learning.
`--epochs 0` supports inference experiments, but is not a separate finalized track.
Changing timing, model, dataset, seed, epochs or simulator changes the benchmark
configuration: results from different configurations must not share a ranking.

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

## Before opening submissions

- Agree on the learning task: on-chip training and inference, pretrained inference,
  or separately ranked tracks. Decide whether designs may encode pretrained weights
  in component values; a netlist parser cannot prove learning happened from scratch.
- The tutorial's DIRECT delta-rule learner is adapted and measured on blobs and
  three MNIST classes. Extend validation to harder datasets, different learning
  rules, more seeds and ten classes. The historical scripts' published accuracies
  do not transfer to this protocol automatically.
- Freeze the circuit interface, resolution, supply, output load, training budget,
  latency, seeds, allowed components and technology. Package a pinned simulator
  build/container digest and dependency versions. Xyce and ngspice share the circuit
  format here, but only one pinned backend should decide official scores.
- Tests cover known resistor power, a MOS charging a storage capacitor through a
  target pin, and retention after training on both simulators. Broaden these checks
  to representative learners; repeat candidate scores at smaller timesteps and tighter tolerances to establish
  an acceptance tolerance. Cross-check finalists with the other simulator.
- Publish a development dataset and a leaderboard policy. `prepare_mnist.py` keeps
  the official 60k/10k source splits separate, records source hashes, then selects
  balanced subsets. `--labels` selects a subset and `--normalize zscore` enables
  per-image contrast normalization after block averaging. `--download` obtains the
  original archives from the torchvision mirror and checks their published MD5
  checksums; preparation records SHA-256 hashes too. The repository's
  existing pooled 70k caches are deliberately not used because they lack split
  provenance. Standard MNIST test data is public: keeping its labels out of SPICE
  prevents direct harness leakage but does not make it a secret test set. Plan a
  separately collected final set or explicitly disclose this limitation.
- Rank the accuracy–energy Pareto frontier at a fixed latency/budget, or publish
  minimum-accuracy tiers with lowest J/image winning. Keep training energy visible;
  if combining it with inference, fix an amortization workload in advance.
- Run third-party submissions in disposable, unprivileged isolation with no network,
  secrets or host mounts and with CPU, memory, process, file-size and wall-time caps.
  This prototype validates a narrow format, disables ngspice startup files, and uses
  a temporary directory and timeout; **it is not a security sandbox**. Do not expose
  it directly as an upload-and-run service.
- Publish submission licensing, attribution, resource-limit/failure policy and
  reproducible result artifacts. Submission intake and a hosted leaderboard are
  not implemented here. Development submissions are accepted through pull requests;
  see [submission instructions](../submissions/README.md).

References: [ngspice documentation](https://ngspice.sourceforge.io/docs.html),
[Xyce](https://xyce.sandia.gov/),
[Xyce documentation](https://xyce.sandia.gov/documentation-tutorials/), and the
repository's [earlier simulator comparison](../docs/SIMULATORS.md).
