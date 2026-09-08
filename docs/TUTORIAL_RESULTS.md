# Tutorial run record

Measured on 2026-09-08 with the circuits and commands in the main README.
Rechecked after the runner reliability changes, using protocol
`mnist4-circuit-learning-v0`. The MNIST reference-image reruns and half-timestep
checks are linked from the [leaderboard](../competition/LEADERBOARD.md).
No Python optimizer trains the weights in these examples.

Small source reports are retained for the blobs tutorial on
[ngspice](results/tutorial-v0/blobs-ngspice.json),
[Xyce](results/tutorial-v0/blobs-xyce.json), and the
[untrained control](results/tutorial-v0/blobs-untrained.json). Full traces stay local.
The tuned starter also has an exploratory
[Xyce cross-check](results/tutorial-v0/mnist017-tuned-xyce.json): 134/150 correct,
zero invalid predictions, and 9.86249 µJ/image. This is not a separate leaderboard
entry; v0 ranking uses ngspice only.

| Dataset / backend | Correct / test | Accuracy | Startup | Training | Inference / image |
| --- | --- | --- | --- | --- | --- |
| Two blobs / ngspice 46 | 100 / 100 | 100% | 98.034 µJ | 4.46117 mJ | 4.81176 µJ |
| Two blobs / Xyce development 7.10 | 100 / 100 | 100% | 98.028 µJ | 4.46117 mJ | 4.81175 µJ |
| MNIST 0,1,7 / ngspice 46 | 99 / 150 | 66% | 401.435 µJ | 69.6601 mJ | 10.7243 µJ |
| MNIST 0,1,7 / ngspice 46, 16 µF variant | 134 / 150 | 89.33% | 2.76849 mJ | 67.9689 mJ | 9.86249 µJ |
| MNIST all ten / ngspice 46, 1.67 µF | 93 / 500 | 18.6% | 1.33810 mJ | 737.352 mJ | 27.0879 µJ |

All energy columns use gross energy delivered through all metered pins; returned
energy is recorded separately. The blobs run without training (`--epochs 0`) has
100 tied/invalid predictions: the circuit begins symmetrically and must learn the
class distinction. Its inference energy is 4.98745 µJ/image. This is not a claim
that a random two-class classifier has 0% expected accuracy: invalid ties score zero.

Common settings: seed 0, 1 ms/example, 20 ms startup, maximum timestep 10 µs,
3 V supply, 27 °C, output load 1 pF || 1 GΩ, NumPy 2.0.2. Solver tolerances,
models and bias values are defined in `competition/runner.py`. ngspice reference
runs use the exact image in [release.json](../competition/release.json).
The exploratory Xyce build identifies itself as
`DEVELOPMENT-202606041546-(Release-7.10.0-182-gd72b5846)-opensource`.

The blob task uses 20 training and 50 test examples per class, generated with seed
7, and 10 training passes. The circuit has 74 MOS transistors, 6 capacitors and
4 resistors, excluding harness sources and output loads.

The MNIST task uses the original training/test split, then selects 24 training and
50 test images for each of digits 0, 1 and 7. Preprocessing is nonoverlapping 7×7
block averaging, division by 255, per-image z-scoring with epsilon 1e-6, then
`clip(0.25*z + 0.5, 0, 1)`. The runner maps those values to 0.3–1.7 V. Twenty
training passes give 1,440 labeled presentations. The circuit has 489 MOS devices,
51 weight capacitors and 6 resistors. This result is a single seed and one
configuration; its score does not establish performance on all ten digits.
The separate ten-digit run has 240 training images, 500 evaluation images, and
4,800 training presentations, totaling 5.32 simulated seconds. It uses 1,630 MOS
devices, 170 weight capacitors, and 20 resistors. That untuned circuit's 18.6%
accuracy is a weak starting point, not a claim that the three-digit tuning carries
over. Its reference run took about 22 minutes on this host.

## Reproduction identifiers

| Artifact | SHA-256 |
| --- | --- |
| `blobs.cir` | `c600f94f5d7f1fe4afb322218d2b6996ab602fc4c2457d3ecae0aa08d6d08b3f` |
| Blob dataset **contents** | `55b8c2612669e6c77df3a881126a08f1f2d02a8392bcd8891ea0e7c3af8d42ed` |
| Blob ngspice harness | `793daad927da41c501c0026eec15b4eaad14a4c9ff0aa7a7ca97a463b80894da` |
| Blob Xyce harness | `f53be386d8d182291479c3e06fd1fdccdcf8e278fae17e9810a36734169b0c72` |
| `mnist_017.cir` | `5299c433178189f3afde4493fc050375d42bfb29246005e56efe9c98f6adfc21` |
| MNIST subset **contents** | `7e615774a3a79e96bd5f50bf16ae2eda0b4b7521a47cb154d759ba7ac8f04a3e` |
| MNIST ngspice harness | `0bd56fa7018276740305b1d5c93e08ef9268e7b3f3bd4ef32ecf99e2c22f843d` |
| `mnist_017_tuned.cir` (16 µF) | `d62626754f08724364da40fc028eb400ecd21641140e38f9d3c9939a89f43c0b` |
| Tuned MNIST ngspice harness | `c684ddb517f64378708e4a179e524a4fada294b0602a4cb2d7889b8933f13e17` |
| `mnist_10.cir` | `8f4f60c31ec3d0c9f87360a0bfd74b87ae9cbb4eb72451d3d87b32eab432c6cc` |
| Ten-digit dataset **contents** | `9c7fd4db02f655862f8f50a94bf331da47a77986e747a201d82f46d7dd4a07b0` |
| Ten-digit ngspice harness | `c35e9c21fd3455c3569199c0c97eeca4a4ae8498e4071c5cffde9c4945eedae8` |
| Tuned MNIST Xyce harness | `c2b93c7e9b05f74f0989b17ca14eee953028a0854713c10b9d951ed629bb92a3` |

Dataset hashes now use `spicenn2-dataset-content-v1`, defined in
[`competition/data.py`](../competition/data.py): fixed array names, shapes, and
little-endian float64/int64 contents, independent of ZIP metadata and compression.
The preparation sidecar also records source-archive hashes. Raw traces are generated
locally and deliberately not committed. Older draft-2 runs used archive-byte hashes
and different harness text; those identifiers remain in Git history.

The v0 rerun retained 99/150 for the original MNIST circuit and 100/100 for blobs
on both simulators. Reported energy agrees at the displayed precision. Both
three-digit circuits and the ten-digit circuit passed the half-timestep check:
every prediction matched, and each phase's delivered energy differed by less
than 1%. For the ten-digit circuit, the largest phase difference was below 0.001%;
the verification run finished in about 29 minutes. Runtimes depend on
CPU and competing work; three-digit runs take minutes, and ten-digit runs can
take tens of minutes. Reports record elapsed wall-clock time separately from
simulated latency and circuit energy.

## Capacitance tuning

The 16 µF variant follows the component-value suggestion in the user-provided
entrant trial feedback. It was independently generated and rerun here; it is not
Claude's unavailable Mac-local submission or a claim to have recovered that PR.
Only the 51 weight capacitances change. The larger capacitance slows the weight
updates and raises startup energy; no external optimizer supplies the weights.

The capacitance was selected using the public development score. Treat the result
as development-set tuning, not accuracy on an untouched final evaluation set.
All table rows are single-seed demonstrations, not statistical confidence
estimates or evidence of physical chip efficiency.

## Interpretation

The circuit is adapted from the DIRECT path of `experiments/gen_mc.py`. The update
cell implements an analog approximation to an error-times-input rule. Reset and
learning-enable transistors make it work in the runner's continuous train/test
schedule; the supplied bias rails are all metered. The stored capacitor voltage is
the weight parameter, but neither transistor current nor the full transfer function
is perfectly linear in that voltage. Do not read the pedagogical delta-rule formula
as an exact mathematical gradient at every operating point.

The models are simplified and the weight capacitors are large (1.67 µF each).
There is no claim that this is an area-efficient integrated circuit. No layout,
process-variation, real DAC/ADC or controller energy is included. Agreement between
the two simulators on the blob example is a useful cross-check, not proof of
fabrication accuracy. The historical MNIST results elsewhere use different circuits,
data preparation, initialization and scoring; they are not directly comparable.

Figures can be regenerated with matplotlib installed:

```bash
python3 competition/plot_tutorial.py data/mnist-idx docs/figures
```

`first_circuit.svg` and `learning_loop.svg` are hand-authored schematic/explanatory
drawings. `tutorial_blobs.svg` plots the generated test dataset; it is not a fitted
decision boundary. `tutorial_mnist.svg` shows actual official-test images and the
preprocessing applied to them.
