# Continuous predictive coding: 16 inputs, 8 hidden units, 3 classes

Author: Thomas Dybdahl Ahle and contributors. MIT license; see LICENSE.

This is a new `mnist017-v0` adaptation of the repository's continuous predictive
coding research, not a reproduction of a historical accuracy number. All 32
input-to-hidden and 24 hidden-to-output weights are capacitor voltages. Training
and inference run in the same transient. Outputs 0, 1 and 7 correspond to those
digit labels; the seven unused outputs are grounded through resistors.

Each hidden unit reads one 2x2 image patch; two independently initialized units
cover each of the four patches. Every output reads all eight hidden units.
The historical degenerated Gilbert `gsyn` cell computes predictions. A second
copy drives the hidden value node, which also receives output-error feedback
through the same physical output-weight capacitors. `dneuron` creates the
nonlinear activation, `esub` represents the local prediction error, and `gprod`
multiplies that error by presynaptic activity to charge the weight pair. Hidden
and output update polarities follow the historical two-sign convention.

This is a PC-inspired circuit with explicit errors and local continuous updates,
not an implementation of exact digital backpropagation or a two-phase equilibrium
propagation estimator. Nonideal transistor transfers and the approximate backward
path do not establish an exact global energy-gradient theorem.

The source is adapted from `experiments/gen_pcL.py` and `pc_deep.py`. The latter's
best completed full-network real-MNIST run used 64→36→16→10 units, dense output
connectivity, different custom models, 400 ns phases and 24 passes. Its reported
45.2% peak used external evaluation-block centering, which this submission does
not perform. The 74–75% frozen-wide historical scores used sklearn 8x8 digits,
not MNIST. This compact port retains the continuous-PC cell mechanics and dense
readout, reduces the input to the official 4x4 interface, and uses hard label
clamping. It does not carry over the historical chopper/symmetric-clock variant,
external annealing schedule, .ic initialization or offline readout processing.

Historical provenance: the [curated record](historical-provenance.json) identifies
`pd_gph_mn24d.scs` and `pd_gph_mn24d.log` by SHA-256 and preserves the completed
trajectory from the later research notes. The 115-MB generated historical deck is
identified by hash, not bundled here. Recovered settings were `TASK=mnist C=10
LAYERS=64,36,16 FANIN=4 KOUT=16 RFGRID=1 NTR=40 NTE=25 NEP=24 WSEED=1 CHL=3
HCHOP=4 SYMNUDGE=1 SOFTC=1 BKSIGN=0 ZEROSUM=1 PERAZ=1 CWW=300p RWL=5meg
SGNH=-1 SGNO=-1 TH=400 GBLH=0.6 GBLO=0.6 AFLOOR=0.4 ANNS=2400 VBBK=0.6`.
The archived deck is authoritative: that CHL/SYM combination leaves the reverse
clock off, so its name must not be read as evidence of centered EP. The compact
port instead uses the earlier, explicit-error, single-phase PC update.

## Electrical adaptation

Only the supplied `nch` and `pch` models are used. Biases are resistor dividers
from metered pins. Device W/L ratios match the historical cells with L reduced
to 1 µm; the model and 3 V operating point differ from the historical 1 V cells.
The resistances are scaled for the new voltage regime. Each learned weight rail
has a 100 µF capacitor and a 1 TΩ common-mode return. These are prototype values,
not a chip-area or fabrication claim.

Python's standard-library RNG, seed 1, selects independent Gaussian per-rail
offsets `v` (standard deviation 0.25 V), clipped to ±0.45 V. The requested rail
voltages are `wp = 1.5 V + v` and `wn = 1.5 V - v`; their differential weight
`wp - wn = 2v` is therefore bounded by ±0.90 V. Python never reads images or
labels. During startup, `reset` powers a resistor divider and an NMOS connects
each capacitor to its initialization voltage. Both start at zero, and all
initialization energy is metered. After reset, the divider supply is zero and
the switch isolates the stored charge. `learn` controls the update and backward
tail biases. Inference uses the direct positive prediction voltage at each output;
there is no evaluation-set centering or class-permutation fitting.

## Reproduce

Activate the repository's Python environment, install Docker, and
[download, verify and load the reference image](../../competition/container/README.md#reference-image-artifact).
The image archive is available from the
[v0 release](https://github.com/thomasnormal/spicenn2/releases/download/v0/ngspice46-v0-image.tar.gz).
No local ngspice installation is needed for scoring. Run from the repository root;
all temporary output paths below must be unused (choose fresh names on a rerun):

```bash
python3 competition/prepare_mnist.py data/mnist-idx /tmp/pc-data.npz \
  --download --labels 0,1,7 --normalize zscore \
  --train-per-class 24 --test-per-class 50 --seed 0
python3 submissions/predictive-coding/generate.py --output /tmp/pc-regenerated.cir
cmp submissions/predictive-coding/circuit.cir /tmp/pc-regenerated.cir
python3 competition/validate.py submissions/predictive-coding/circuit.cir
python3 competition/runner.py submissions/predictive-coding/circuit.cir \
  /tmp/pc-data.npz --task mnist017-v0 \
  --docker-image sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d \
  --output /tmp/pc-v0
python3 competition/runner.py submissions/predictive-coding/circuit.cir \
  /tmp/pc-data.npz --epochs 20 --startup 0.02 --slot 0.001 --seed 0 --step 5e-6 \
  --docker-image sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d \
  --output /tmp/pc-v0-half
```

The named task fixes the official dataset/preprocessing, shuffle seed 0, 20
training passes, 20 ms startup, 1 ms slots, 10 µs maximum timestep, and ngspice 46.
For the paired numerical check, omit `--task` and explicitly supply `--epochs 20
--startup 0.02 --slot 0.001 --seed 0 --step 5e-6` with the same reference image
and a different output directory. Named tasks lock the original timestep.
The generator requires an explicit `--output` and refuses existing files, so
regeneration does not overwrite the shipped circuit or an entrant's edits.

An optional unranked diagnostic observes internal capacitor voltages over six
training examples and three inference examples using the same model and timings:
it requires a local ngspice 46 executable, unlike the Docker scoring commands.

```bash
python3 submissions/predictive-coding/diagnose.py /tmp/pc-data.npz
```

At the initial port's 100 µF setting, sampled hidden differential weights changed
+127 µV and −118 µV during training. Individual hidden rails changed at most
0.214 µV over the following three inference slots. This shows electrically active
hidden updates and short-term retention; it does not prove useful feature learning.
The frozen control's hidden rails changed at most 0.112 µV during the same
training diagnostic, versus 230–357 µV in the trained circuit. Both designs use
the same requested initial weights; finite reset charging leaves their actual
startup values a few millivolts apart. Observations are in `weight-diagnostic.json`.
Repeating with `--permute-labels` keeps startup and inputs identical but cycles
the training labels. Hidden differential updates then differ by 10.7 and 37.1 µV
from the original diagnostic (`permuted-weight-diagnostic.json`), demonstrating
that the hidden update responds to supervision, rather than only passive drift.

## Results and tuning disclosure

The full reference-image v0 run scored **33.33% (50/150)** and predicted digit 7
for every image. Startup energy was 0.2279181 J, training energy 0.1712083 J,
and inference energy 41.9378 µJ/image; see `report.json`. Runtime was 457 s.
This compact port has not recovered useful classification or the historical
accuracy. Its hidden weights respond to labels, but that is not evidence of an
accuracy benefit. The 5 µs run (`report-half-step.json`) gives an identical
prediction list and phase energy differences below 0.000004%; its runtime was
730 s. The read-only report verifier passes, including circuit, dataset, image,
settings, harness and score consistency. The organizer has recorded this pair in
the [starter leaderboard](../../competition/LEADERBOARD.md), with the
[reference report](../../competition/results/mnist017-v0/predictive-coding/report.json)
and [half-step report](../../competition/results/mnist017-v0/predictive-coding/verification.json).
Historical research scores are not submission scores.

To verify your reproduced pair and compare it with the stored result:

```bash
python3 competition/leaderboard.py verify --task mnist017-v0 \
  --circuit submissions/predictive-coding/circuit.cir \
  --report /tmp/pc-v0/report.json --verification /tmp/pc-v0-half/report.json \
  --dataset /tmp/pc-data.npz --against submissions/predictive-coding/report.json
```

The zero-training reference-image control scores 33.33% (50/150), at
42.1500 µJ/inference; see `untrained-report.json`. It is an unranked control.
Development checks use the public starter dataset and include a one-pass sanity
run, the internal-weight diagnostic, and output capacitance screens at 100, 10
and 1 µF. No task-dependent weights are trained or loaded outside SPICE.
The companion `predictive-coding-frozen` entry is a fixed-feature architectural
comparison: it removes hidden errors and feedback, merges value/prediction nodes,
and continuously drives hidden weights with dividers. It is not a tightly matched
hidden-learning ablation.
