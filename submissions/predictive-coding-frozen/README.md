# Frozen random features with a continuous-PC readout

Author: Thomas Dybdahl Ahle and contributors. MIT license; see LICENSE.

This is a fixed-feature architectural comparison for ../predictive-coding. It
retains the 16→8→3 layer sizes, local 2x2 receptive fields, deterministic seed,
transistor cell family and dense output readout. The 32 hidden weights are data-independent
resistor-divider voltages; the 24 output weights learn in capacitor charge using
the local `gprod` error-times-activity update. The hidden layer has no feedback
or update cells. It also removes the hidden error cells, merges hidden value
and prediction nodes, and replaces capacitor initialization with continuously
driven dividers. These changes affect loading and startup values; this is a
different fixed-feature architecture, not a tightly matched hidden-learning
ablation. It does not claim hidden feature learning.

The design represents the historical PC family's frozen-random-feature strategy,
whose strongest 74–75% scores were on sklearn 8x8 digits. Historical real-MNIST
frozen results were approximately 49.6% under different preprocessing, circuitry
and readout. None of those numbers is a v0 submission score.

The submitted circuit is self-contained. Regeneration imports the cell library
from the adjacent ../predictive-coding/generate.py using an explicit file path;
neither generator reads a dataset. Seed 1 chooses Gaussian per-rail offsets `v`
(standard deviation 0.25 V), clipped to ±0.45 V. Rails are `wp = 1.5 V + v` and
`wn = 1.5 V - v`, so the differential weight `wp - wn = 2v` is bounded by
±0.90 V. Fixed hidden weights use continuously metered
dividers. Output capacitors start at zero and are initialized by reset-powered
dividers through MOS switches. The `learn` pin gates output updates. All outputs
are direct circuit voltages, without Python centering or label matching.

Activate the repository's Python environment and
[download, verify and load the reference image](../../competition/container/README.md#reference-image-artifact)
using Docker. The archive is on the
[v0 release](https://github.com/thomasnormal/spicenn2/releases/download/v0/ngspice46-v0-image.tar.gz).
Run from the repository root with unused temporary output paths:

```bash
python3 competition/prepare_mnist.py data/mnist-idx /tmp/pc-frozen-data.npz \
  --download --labels 0,1,7 --normalize zscore \
  --train-per-class 24 --test-per-class 50 --seed 0
python3 submissions/predictive-coding-frozen/generate.py --output /tmp/pc-frozen-regenerated.cir
cmp submissions/predictive-coding-frozen/circuit.cir /tmp/pc-frozen-regenerated.cir
python3 competition/validate.py submissions/predictive-coding-frozen/circuit.cir
python3 competition/runner.py submissions/predictive-coding-frozen/circuit.cir \
  /tmp/pc-frozen-data.npz --task mnist017-v0 \
  --docker-image sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d \
  --output /tmp/pc-frozen-v0
python3 competition/runner.py submissions/predictive-coding-frozen/circuit.cir \
  /tmp/pc-frozen-data.npz --epochs 20 --startup 0.02 --slot 0.001 --seed 0 --step 5e-6 \
  --docker-image sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d \
  --output /tmp/pc-frozen-v0-half
```

The named task fixes the official data and preprocessing, shuffle seed 0, 20
passes, 20 ms startup, 1 ms slots, 10 µs timestep and ngspice 46. For the numerical
check, omit `--task`, explicitly set `--epochs 20 --startup 0.02 --slot 0.001
--seed 0 --step 5e-6`, and use the same image with a new output directory.
The generator requires `--output` and refuses to overwrite an existing path.
Use fresh names when repeating these commands; the shipped circuit stays intact.

The full 20-pass reference-image run scored **33.33% (50/150)**, predicting digit
7 for every example. Startup energy was 0.0980771 J, training energy 0.1067700 J,
and inference energy 34.5990 µJ/image. The report is `report.json`. This is a weak
adaptation, not a competitive accuracy result.
The 5 µs run (`report-half-step.json`) produced the identical prediction list,
and every phase's delivered energy agreed within 1%; see `numerical-check.json`.
The organizer has recorded this pair in the
[starter leaderboard](../../competition/LEADERBOARD.md), with the
[reference report](../../competition/results/mnist017-v0/predictive-coding-frozen/report.json)
and [half-step report](../../competition/results/mnist017-v0/predictive-coding-frozen/verification.json).
The read-only report verifier also passes, including dataset, image, task and
regenerated harness identities. To verify your own runs and compare with the
stored reference result without writing leaderboard files:

```bash
python3 competition/leaderboard.py verify --task mnist017-v0 \
  --circuit submissions/predictive-coding-frozen/circuit.cir \
  --report /tmp/pc-frozen-v0/report.json \
  --verification /tmp/pc-frozen-v0-half/report.json \
  --dataset /tmp/pc-frozen-data.npz \
  --against submissions/predictive-coding-frozen/report.json
```

The zero-training reference-image control also scores
33.33%, at 34.6564 µJ/inference (`untrained-report.json`). The adaptation has not
demonstrated an accuracy improvement from training. One-pass development checks used the public starter
evaluation set. At 100 µF and 10 µF, the initial port scored 33.3% and predicted
one class; a 1 µF screen also scored 33.3%. No data-dependent weights
are fitted offline. The corresponding trained-PC README explains the electrical
adaptation, historical limitations and internal-state diagnostic.
The stored `weight-diagnostic.json` shows fixed hidden rails changing by at most
0.112 µV during six training examples, while an output weight pair changes by
hundreds of µV. Thus only the output stage implements learning in this control.
