# Physical random features + per-class auto-zero

Author: Thomas Dybdahl Ahle and contributors. MIT; see LICENSE.

This ports the **architecture family** of the strongest historical ten-digit
keystone/PERAZ experiment, not its pretrained state or published score. The
submitted `circuit.cir` has 16 inputs, 96 fixed nonlinear features, and three
learning outputs for `mnist017-v0`. Only the readout learns. It is a local analog
delta-rule learner, not backprop or equilibrium propagation.

## What is electrical here?

Every feature reads four randomly selected pixels and a bias through differential
MOS synapses. Fixed resistor dividers establish data-independent random gate
voltages. Current mirrors and a bounded differential-pair activation produce the
feature voltage; these are physical nonlinearities, not an exact mathematical
`tanh`. All 96 feature circuits remain powered during inference and their energy,
including the divider and input-pin loads, is counted.

The readout uses weight capacitors and input/error-controlled current updates.
Each class's error is compared with a slow local error average: **PERAZ**, or
per-class auto-zero. A resistor and 10-µF capacitor form that tracker. An additional
resistor star couples the class errors (the historical output common-mode circuit).
Passive dividers transform the target pins to approximately 0.467/0.800 V.

Reset-powered dividers charge the readout's 5-µF weight capacitors to small,
data-independent random initial weights. MOS switches initialize auto-zero states
from the metered error rail. Both weight updates and auto-zero tracking are gated
off by `learn` during evaluation. There are no ideal initial conditions, external
features, extracted/averaged weights, fitted output calibration, or test-label
access. `generate.py` reads no dataset and uses only the Python standard library.

## Run

From the repository root, after the [Python setup](../../README.md) and
[reference image download](../../competition/container/README.md), use unused paths:

```bash
python3 submissions/random-feature-peraz/generate.py /tmp/peraz-regenerated.cir
cmp submissions/random-feature-peraz/circuit.cir /tmp/peraz-regenerated.cir
python3 competition/validate.py submissions/random-feature-peraz
python3 competition/prepare_mnist.py data/mnist-idx /tmp/peraz-data.npz --download --labels 0,1,7 --normalize zscore
python3 competition/runner.py submissions/random-feature-peraz/circuit.cir /tmp/peraz-data.npz --task mnist017-v0 --docker-image sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d --output /tmp/peraz-score
```

This is substantially larger and slower than the introductory linear learner.
For a short, **unranked** smoke check, replace `--task mnist017-v0` with
`--epochs 2 --startup .02`. Generate with `--freeze` for an untrained control;
it preserves initialization and forward circuitry but disables the update and
auto-zero tracking switches. The CLI also supports all ten labels; that is a
different circuit and requires its own measurement, not a transferred score.

## Historical provenance and important differences

The earlier [run_iso.sh](../../experiments/run_iso.sh) pipeline used 64 input
pixels, 96 random features computed by **Python** (`tanh` of Gaussian projections),
and an in-SPICE trainable readout. The [research ledger](../../docs/IDEAS_LEDGER.md)
reports **ngspice** results of 74.7% on 600 test images (40 training images/class)
and 75.0% with 80 training images/class; these are not the older Spectre results.
Evaluation used a separate inference deck with 20 late weight samples averaged
by Python. Its pooled/shuffled dataset split, ideal initial states, models,
timing, preprocessing and evaluation procedure differ from v0.

Those are historical research results, **not scores for this submission**.
Replacing that Python front end with electrical feature circuits changes both
the transfer function and energy budget. The present port retains 96 features,
four-input random connectivity, bounded activations, per-class auto-zero and
the compressed-target readout, but uses v0's 16 pixel pins and three-class starter
task. Random weights are clipped Gaussian-derived gate voltages; nonlinear device
currents are not exact linear Gaussian projections. No old accuracy is claimed.
The historical 300-ms/1-ms slot ratio motivates the initial capacitances
(1.5 mF → 5 µF for weights, 3 mF → 10 µF for auto-zero).

## Measurements

The 6,915-component netlist passes the submission format validator. On the exact
starter dataset and reference image, a two-pass unranked check scored **82.67%
(124/150)**, versus **28.67% (43/150)** with zero training passes. Both had zero
invalid predictions. Inference energy was 286.007 µJ/image after two passes and
289.578 µJ/image untrained; startup was 61.4485 mJ in both, and two-pass training
was 80.3300 mJ. See `two-epoch-report.json` and `untrained-report.json`.
These runs took 462 and 336 seconds respectively on the development host.

The untrained measurement uses the same submitted circuit with `--epochs 0
--startup .02`; the optional `--freeze` generator control is a separate wiring
variant, not the circuit measured in that report. These checks demonstrate
readout learning on this short schedule; they do not earn a ranked score.
Full 20-pass reference and half-step checks are in progress. The physical feature
front end consumes substantially more energy than the small linear baseline.
