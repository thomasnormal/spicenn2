# backprop-cmsub

Author: Thomas Dybdahl Ahle and contributors. MIT; see LICENSE.

This is a circuit implementation of the historical CMSUB backprop learner in
`experiments/gen_mc.py` and `experiments/run_backprop.sh`, adapted to the
`mnist017-v0` interface. The submitted artifact is `circuit.cir`; the organizer
does not need to execute Python. `generate.py` uses only the standard library,
reads no dataset or trained-weight file, and reproduces the netlist deterministically.

## Learning circuit

Sixteen input pins feed nine overlapping 2×2 receptive fields. Each hidden unit
has four trainable input weights and one trainable bias, a current-difference
readout, and a bounded differential-pair activation. Three output neurons read
all nine hidden activations plus a bias, producing scores on out0, out1 and out7.
The other digit outputs are unused. There are 75 weight capacitors in total.

Each output-weight capacitor also drives two gates in a four-transistor
differential transpose cell. Buffered positive and negative output-error signals
flow through those shared weights and sum at each hidden unit's backward column.
A nine-resistor star subtracts the mean hidden error (CMSUB) before the hidden
update cells integrate their local input/error products. Both layers learn.

This is analog transpose-read backprop with common-mode subtraction, not an exact
implementation of the mathematical tanh derivative. It preserves the historical
comparator activation gate, a ReLU′-style approximation, and subtracts the hidden
backward mean. It is neither fixed-random feedback alignment nor an equilibrium
propagation circuit.

All capacitors begin at zero. During startup, resistor dividers powered by the
reset pin and low-threshold MOS switches initialize the weight capacitors to
data-independent pseudorandom voltages (Python Random seed 5). Hidden input
weights use 1.8 + 1.2×round(uniform(-0.8,0.8),3) V, hidden biases 2.16 V, and
output weights 1.8 + 1.2×round(uniform(-0.3,0.3),3) V. These are initial states,
not weights obtained by training. The divider rails return to zero when reset
ends and the switches open. Their energy is included in startup measurements.

Each update cell has a learn-controlled tail switch, including the hidden cells.
During evaluation those switches open; the learned capacitor voltages are retained
in the same transient. The unused backward circuitry remains connected and its
energy is counted. There is no external optimizer, weight extraction, fitted
readout, output centering, classification controller, or test-label access.

The historical update input band (0.3–1.0 V) is generated electrically from the
0.3–1.7-V image pins: au = input/2 + 0.15 V. Additional gate biases come from
metered-rail resistor dividers. Two output-load resistors reproduce each hidden
activation's historical 8 kΩ load to 0.7 V by its Thevenin equivalent.

## Reproduce

From the repository root, choosing unused output paths:

```bash
python3 submissions/backprop-cmsub/generate.py /tmp/backprop-regenerated.cir
cmp submissions/backprop-cmsub/circuit.cir /tmp/backprop-regenerated.cir
python3 competition/validate.py submissions/backprop-cmsub
python3 competition/prepare_mnist.py data/mnist-idx /tmp/backprop-data.npz --download --labels 0,1,7 --normalize zscore
python3 competition/runner.py submissions/backprop-cmsub/circuit.cir /tmp/backprop-data.npz --task mnist017-v0 --docker-image sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d --output /tmp/backprop-score
```

For an untrained control, generate with `--freeze-all`. For a fixed-random-hidden,
trained-readout control, generate with `--freeze-hidden`. These change the update
enable wiring, not the initialization seed or forward topology. Custom shorter
checks use `--epochs 2 --startup .02` without a named task. Such checks are not
ranked task results.

## Historical evidence and disclosure

The earlier [backprop study](../../docs/BACKPROP_DEPTH.md) reports eight seed
accuracies from 72.0% to 84.7% on a different MNIST split. Its trained hidden and
output weights were physically updated in ngspice, but evaluation used a separate
forward-only deck with Python-averaged late weight samples. Its random capacitor
states used ideal initial conditions. Those numbers are provenance, not scores
for this submission.

The original recipe is `experiments/run_backprop.sh`, with `D=16 C=3 H=9
CONN=rf3x3 ACT=tanh TW=32 VTB=1.4 VMID=0.5 BHID=0.3 INITH=0.8 INIT=0.3
OWTW=20 CMSUB=1 RCMB=10e3 INLO=0.3 INHI=1.7 VREFH=0.5`, default seed 5,
1,400 training slots at 300 ms each, 500-µF weights, clipped per-image z-score
inputs, 24 training/30 validation/50 test samples per class, and `AVGW=20`
for separate-deck inference. The older preparation shuffled a pooled MNIST cache;
the v0 split and test schedule are different. A later ledger reports 85.3% in
Spectre on that historical three-class configuration; that is also not a v0 score.

The port uses the v0 official train/test split, supplied models including their
capacitances, all-zero startup, physical reset, learn gating, and continuous
evaluation without weight averaging. Weight capacitance is initially scaled from
500 µF to 1.67 µF for the change from 300-ms to 1-ms training slots. The generator's
optional seed/capacitance arguments permit disclosed new design revisions; the
submitted defaults are fixed. No pretrained values or data-dependent connectivity
are encoded. Circuit-boundary simulated energy excludes physical external driver
and readout implementation losses.

## Current verification

The 1,444-component netlist passes the v0 R/C/M/D format validator and reproduces
byte-for-byte from the generator. It contains 1,095 MOSFETs, 273 resistors and
76 capacitors (75 weights plus the backward-mean capacitor).

Initial local ngspice 46 checks use the official v0 dataset, 20-ms startup and
1-ms slots, with a shorter training schedule. All have zero invalid predictions:

| Check | Training passes | Correct | Accuracy |
| --- | --- | --- | --- |
| [Untrained, all updates disabled](results/untrained.json) | 0 | 50/150 | 33.33% |
| [Frozen random hidden, trained readout](results/frozen-hidden-two-epochs.json) | 2 | 72/150 | 48.00% |
| [Both layers trained with CMSUB](results/two-epochs.json) | 2 | 85/150 | 56.67% |

The two trained checks differ only in the 45 hidden update-enable connections;
the initial voltages, topology and output learning are identical. This check
supports a positive contribution from hidden learning for this seed and schedule.
It does not establish a general advantage or the final 20-pass task performance.
The checks used local ngspice, not the reference container. All three reports
are retained as entrant measurements, not organizer-verified leaderboard records.

The full [mnist017-v0 reference-container run](results/mnist017-v0.json) measured
**122/150 correct (81.33%)**, zero invalid predictions, **59.3951 µJ/image**,
14.9601 mJ startup energy and 174.6604 mJ training energy. Per-class correct counts
are 35/50 for zero, 50/50 for one and 37/50 for seven. It used the named task's
20 training passes, fixed shuffle seed and exact reference image, and completed
in approximately 787 wall-clock seconds. The organizer ran and reviewed the
reference-image pair and [recorded it on the leaderboard](../../competition/LEADERBOARD.md).
No frozen-hidden control was run at the
full 20-pass schedule, so the shorter ablation should not be extrapolated to it.

The [5-µs reference-container check](results/mnist017-v0-halfstep.json) also measured
122/150 correct. All 150 predictions matched, and every phase's delivered energy
differed by less than 1%. The read-only report verifier passed circuit, dataset,
task, image, settings, regenerated harnesses, score consistency and numerical
stability. The verifier alone checks report consistency; the organizer's actual
simulator reruns and review establish the recorded result. The circuit SHA-256 is
`030d49fb946df4f78427f9b6cd5ddeba53e943eb90e230de4edd4eb0fde1de2f`.

To reproduce the numerical check and inspect the retained report pair:

```bash
python3 competition/runner.py submissions/backprop-cmsub/circuit.cir /tmp/backprop-data.npz --epochs 20 --startup .02 --step .000005 --docker-image sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d --output /tmp/backprop-halfstep
python3 competition/leaderboard.py verify --task mnist017-v0 --circuit submissions/backprop-cmsub/circuit.cir --report submissions/backprop-cmsub/results/mnist017-v0.json --verification submissions/backprop-cmsub/results/mnist017-v0-halfstep.json --dataset /tmp/backprop-data.npz
```

`python3 -m unittest competition.test_backprop_submission -v` checks regeneration
without data reads, shared learned-weight gates in the backward cells, electrical
update/reset paths, isolated control changes, and report/circuit identities.
