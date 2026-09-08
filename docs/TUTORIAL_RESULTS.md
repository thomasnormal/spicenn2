# Tutorial run record

Measured on 2026-09-08 with the circuits and commands in the main README.
These are development results for `mnist4-circuit-learning-draft2`, not official
competition scores. No Python optimizer trains the weights in these examples.

| Dataset / backend | Correct / test | Accuracy | Startup | Training | Inference / image |
| --- | --- | --- | --- | --- | --- |
| Two blobs / ngspice 46 | 100 / 100 | 100% | 98.034 µJ | 4.46117 mJ | 4.81176 µJ |
| Two blobs / Xyce development 7.10 | 100 / 100 | 100% | 98.028 µJ | 4.46117 mJ | 4.81175 µJ |
| MNIST 0,1,7 / ngspice 46 | 99 / 150 | 66% | 401.435 µJ | 69.6601 mJ | 10.7243 µJ |

All energy columns use gross energy delivered through all metered pins; returned
energy is recorded separately. The blobs run without training (`--epochs 0`) has
100 tied/invalid predictions: the circuit begins symmetrically and must learn the
class distinction. Its inference energy is 4.98745 µJ/image. This is not a claim
that a random two-class classifier has 0% expected accuracy: invalid ties score zero.

Common settings: seed 0, 1 ms/example, 20 ms startup, maximum timestep 10 µs,
3 V supply, 27 °C, output load 1 pF || 1 GΩ, NumPy 2.0.2. Solver tolerances,
models and bias values are defined in `competition/runner.py`. ngspice reports
version 46 with KLU support. The Xyce build identifies itself as
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
configuration; it does not establish performance on all ten digits.

## Reproduction identifiers

| Artifact | SHA-256 |
| --- | --- |
| `blobs.cir` | `c600f94f5d7f1fe4afb322218d2b6996ab602fc4c2457d3ecae0aa08d6d08b3f` |
| Blob dataset archive | `5d610d1ef39269288284fa2b79ced84797fc9b5e09f3d7594138a382cda52c9e` |
| Blob ngspice harness | `4139c0a75203bc4b66410f05984610c00ad3d6923b6d680bc034e5a0b50d38b3` |
| Blob Xyce harness | `57fbb16e3bc463aef742d4ed4e4ce5dc3248b9a3b8190cc32bd180110a390e9a` |
| `mnist_017.cir` | `5299c433178189f3afde4493fc050375d42bfb29246005e56efe9c98f6adfc21` |
| MNIST subset archive | `9b7b5ccf18d07c2916143cb8de40b2651ae257fe4af474f7e32aaf01e1be34e2` |
| MNIST ngspice harness | `a7f07ac4ed070bddd8b0fec6692df1f193d409a4d1ff242a2372a8e8e98bb636` |

The preparation sidecar records the source-archive hashes. NumPy archive bytes can
depend on packaging versions; compare arrays and preprocessing as well as archive
hashes when investigating a reproduction mismatch. Raw traces are generated locally
and deliberately not committed.

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
