# Can an electrical circuit learn to recognize handwriting?

MNIST is a collection of handwritten digits. Usually, a computer learns to recognize
them by adjusting numbers in a neural network. **Here, we want the electrical circuit
itself to learn.** An image becomes a set of voltages, transistors turn those voltages
into currents, and capacitors store learned weights as electrical charge.

This project is a playground for learning that idea and a developing competition:
**design a circuit that learns to classify digits accurately, using little energy.**
You submit the circuit; a Python runner presents the data, simulates it, and measures
its performance. You can do everything in software—no electronics lab is needed.

Why try this? Currents naturally add at a wire junction, and a capacitor can remember
a voltage without repeatedly moving a weight between a processor and memory.
That could make some learning operations efficient. Whether a particular circuit
actually saves energy is something to measure, not assume.

## 1. Meet an analog circuit

A digital signal represents discrete values such as 0 and 1. An analog signal can
represent a value by a continuously varying voltage or current. A dark pixel might
become 0.3 V and a bright one 1.7 V.

Start with something smaller than a neural network: **a resistor charging a capacitor**.

![A voltage source drives a resistor, which charges a capacitor. The capacitor voltage is labeled memory.](docs/figures/first_circuit.svg)

The resistor limits current. The capacitor stores charge, so the voltage at `memory`
changes gradually when the input changes. Its time scale is `R × C`: here, about
1 millisecond. This is one building block of a learned weight. By itself, the circuit
stores a voltage; it is not a classifier.

### Program the connections

A SPICE **netlist** describes which component connects to which wire. These are the
essential lines of [the complete example](circuits/first_capacitor.cir):

```spice
Vin input 0 PULSE(0 1 0 1u 1u 5m 10m)
R1 input memory 10k
C1 memory 0 100n
.tran 10u 4m
```

`Vin` drives a pulse from 0 to 1 volt. `R1` connects `input` to `memory` through
10 kΩ. `C1` connects `memory` to ground (`0`) through 100 nF. Names such as `input`
and `memory` identify wires; using the same name means connecting to the same wire.
`.tran` asks the simulator to follow the circuit over time. The full example also
requests the voltages to be printed and ends the netlist.

For a neural circuit, we add transistors. A MOS transistor uses a voltage on its
**gate** to control current through another connection. A stored weight voltage can
therefore control how strongly an input contributes to a prediction.

### Draw it and watch it work

Open **[Spice Sim in your browser](https://thomasahle.com/spice-sim/)** to draw circuits
and run ngspice simulations without installing anything. Build the source–resistor–
capacitor circuit above, run a transient analysis, and look at the capacitor voltage.
Try increasing the resistance or capacitance: the voltage should respond more slowly.
The diagram and netlist describe the same connections.

For batch experiments, install [ngspice](https://ngspice.sourceforge.io/download.html)
or an [open source build of Xyce](https://xyce.sandia.gov/downloads/source-code/).
Both solve the circuit's electrical equations over time. Xyce also supports parallel
simulation; whether that is faster depends on the circuit and build. From this repository:

```bash
ngspice -b circuits/first_capacitor.cir
# Or:
Xyce circuits/first_capacitor.cir
```

The browser is useful for exploring small circuits. The Python runner below handles
the repeated data presentation and measurements needed for learning experiments.

## 2. Teach a small circuit first

Before handwriting, try two clouds of points. Each point has just two inputs and
belongs to class 0 or class 1. This deliberately simple problem makes it easier to
see whether the circuit is learning.

![Two separated clouds of points: blue class 0 and orange class 1.](docs/figures/tutorial_blobs.svg)

The [two-input learning circuit](competition/examples/blobs.cir) has a score for each
class and a capacitor for each trainable weight. Its design comes from this project's
existing multiclass learner, adapted to the submission interface.

![Inputs pass through transistor synapses to produce scores. An error circuit and update cell feed changes back into the weight capacitors.](docs/figures/learning_loop.svg)

It uses an **analog delta rule**, the single-layer relative of gradient descent:

```text
weight change ∝ input × (target − prediction)
```

The transistor update cell approximates this relationship electrically. When a score
is too low or too high, the update current changes the corresponding capacitor voltage.
That voltage changes the next prediction. This baseline has no hidden layer, so it
does not need backpropagation through several layers or an equilibrium-propagation
free/nudged phase pair. The [research archive](docs/README.md) explores those alternatives.

Here is the **actual transistor circuit for one weight cell**, not just a block
diagram. `Cw0_0` stores the weight; `Mreset0_0` initializes it; the `Mu` transistors
charge or discharge it during learning. `Mpos0_0` and `Mneg0_0` read the input as
currents that feed the class score circuit. Matching blue wire labels mean an
electrical connection, even where no wire is drawn between them.

![Transistor-level schematic of one complete weight cell, with its weight capacitor, reset switch, input-gated update circuit, and two readout transistors.](docs/figures/blobs_weight_cell.svg)

The full two-input learner repeats this cell six times: one per input plus a bias
for each of the two classes. [Open the complete, zoomable circuit schematic](docs/figures/blobs_circuit.svg)
to see all **74 transistors, 6 capacitors, and 4 resistors**, including both score
and error circuits. The runner supplies the external signals and bias voltages.
Both drawings use the device names and values from the
[raw netlist](competition/examples/blobs.cir); regenerate them with
`python competition/draw_blobs.py`.

Here is the corresponding netlist excerpt for the capacitor, reset, and readout:

```spice
* Store the weight and initialize it during reset.
Cw0_0 n_w0_0 0 1.67e-06
Mreset0_0 vweight reset n_w0_0 0 syn W=1000u L=1u
* Read the input using the learned gate voltage.
Mpos0_0 in0 n_w0_0 n_pos0 0 syn W=10u L=1u
Mneg0_0 in0 vweight n_neg0 0 syn W=10u L=1u
```

For a MOS device, the four node names are drain, gate, source, and body; `syn` selects
the supplied transistor model, and `W`/`L` set its width and length. The rest of the
file contains the score and update circuitry. During initialization, a real switch
charges the capacitor from a metered bias source; it does not begin with free stored charge.

### Run the circuit on the dataset

Install Python 3.9+ and ngspice, then set up the runner:

```bash
git clone --depth 1 https://github.com/thomasnormal/spicenn2.git
cd spicenn2
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r competition/requirements.txt

python competition/prepare_toy.py /tmp/blobs.npz
python competition/runner.py competition/examples/blobs.cir /tmp/blobs.npz \
  --epochs 10 --startup 0.02 --output /tmp/blobs-score
```

The first argument is **your circuit file**; the second is the dataset. The runner
presents 40 training examples for 10 passes, disables the learning signal, then
presents 100 test examples without labels—all in one continuous simulation. It reads
the output voltages and reports accuracy and energy. Python does not compute or apply
weight updates.

This example scored **100/100 test points** on both ngspice and Xyce, drawing about
**4.81 µJ per test example** with the settings above. Without training (`--epochs 0`),
the symmetric initial outputs tie and count as invalid predictions. These are small
demonstration results, not an estimate of handwriting accuracy.

To use Xyce, add `--simulator xyce` and choose a new output directory. Its executable
should be named `Xyce` on your PATH, or supplied through `--binary /path/to/Xyce`.
Each run saves a `report.json`, generated test harness, simulator log, and voltage/
power trace. Output directories must be new so one experiment cannot overwrite another.

## 3. From two inputs to MNIST

The same idea scales to pixels: one input voltage per pixel, a learned set of weights
for each digit, and one output score per digit. Start with **real MNIST digits 0, 1,
and 7, reduced from 28×28 to 4×4** so the simulation remains approachable.

![Original MNIST digits 0, 1, and 7 above their normalized 4 by 4 circuit inputs.](docs/figures/tutorial_mnist.svg)

Open [the MNIST learning circuit](competition/examples/mnist_017.cir). It has 16 pixel
inputs, 3 class scores, and 51 weight capacitors: 16 input weights plus a bias weight
per class. All its trainable weights learn inside SPICE. The small
[circuit generator](competition/make_baseline.py) simply writes repeated connections;
it does not read training data or learn weights in Python.

```bash
python competition/prepare_mnist.py data/mnist-idx /tmp/mnist017.npz \
  --download --labels 0,1,7 --normalize zscore
python competition/runner.py competition/examples/mnist_017.cir /tmp/mnist017.npz \
  --epochs 20 --startup 0.02 --output /tmp/mnist017-score
```

Preparation downloads and checks the original MNIST archives, keeps the official
training and test splits separate, then selects 24 training and 50 test images per
class. Each 7×7 block becomes one pixel; per-image contrast normalization then maps
the 16 values into the input-voltage range.

With the command above, the circuit scored **66% (99/150 test images)** in ngspice 46,
using about **10.72 µJ per test image**. Chance is 33.3%. This is a **three-digit, 4×4
baseline**, not the full ten-digit, 28×28 task. See [the run record](docs/TUTORIAL_RESULTS.md)
for settings, energy breakdowns, simulator versions, and limitations.

To explore all ten digits, export a circuit with
`--labels 0,1,2,3,4,5,6,7,8,9` using `make_baseline.py`, and prepare the matching
dataset. More classes are harder; simply adding outputs does not guarantee good accuracy.

## 4. Enter the competition

The goal is to learn MNIST accurately with low electrical energy. **The runner and
example submissions are available; official benchmark settings and the leaderboard
are still being finalized.** Current scores are development results.

1. Start from [an example circuit](competition/examples/) and change the connections
   or component values. The current format accepts resistors, capacitors, diodes and
   MOS transistors using the provided device models.
2. Run it locally with `competition/runner.py`. The runner supplies the image, training
   label, power/bias voltages, reset, and timing. Your circuit supplies the class scores.
3. Open a pull request adding `submissions/<name>/circuit.cir` and a short README with
   your learning algorithm, exact run command, data configuration, and local results.
   Include attribution and a license. See [submission instructions](submissions/README.md).
4. The organizer reruns the circuit using the common configuration. Comparisons must
   use the same dataset, labels, preprocessing, training budget, timing, models,
   and simulator version.

### What gets measured?

| Measurement | Meaning |
| --- | --- |
| Accuracy | Fraction of test examples correct; highest output voltage wins |
| Inference energy | Joules supplied to the circuit per test example |
| Training energy | Total joules supplied while the circuit learns |
| Startup energy | Energy used to establish voltages and initialize stored state |
| Latency | Simulated time allowed per example; default 1 ms |

Energy is the integral of supplied power, `voltage × current`, over time. We meter
**every driven pin**, including image, label, clock and bias pins: counting only the
main supply would miss circuits powered through their inputs. Returned energy is
reported separately and does not cancel energy drawn elsewhere.

Accuracy and energy are separate objectives: always guessing one digit is cheap but
not useful. We intend to compare the accuracy–energy tradeoff at fixed training and
latency budgets; an official ranking formula has not yet been frozen.
See [the runner specification](competition/README.md) for the exact interface and measurements.

These are **simulated circuit energy** figures, not the computer's electricity use.
The current transistor models are simplified, and ideal input drivers and output
readout exclude realistic DAC/ADC/controller losses. Treat the numbers as a reproducible
circuit-model benchmark, not fabricated-chip efficiency.

## Explore the research

The [research guide](docs/README.md) links earlier findings, figures, and the experiment
ledger. [pc_deep.py](pc_deep.py) generates a different family of deep predictive-coding
circuits; it is a research program, not the submission runner. Historical scripts live
in [experiments/](experiments/README.md), reference circuits in [circuits/](circuits/README.md),
and manuscript sources in [paper/](paper/). Generated output and datasets are
[kept out of Git](docs/REPOSITORY_HYGIENE.md).
