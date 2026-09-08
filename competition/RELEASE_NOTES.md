# Circuit-learning competition v0

Design an analog circuit that learns to recognize handwritten digits. Submit a
SPICE netlist; the organizer runs it and measures accuracy, startup/training
energy, and inference energy per image. No electronics lab is required.

Start with the [tutorial](https://github.com/thomasnormal/spicenn2/blob/v0/README.md), then read the
[rules](https://github.com/thomasnormal/spicenn2/blob/v0/competition/RULES.md) and
[submission instructions](https://github.com/thomasnormal/spicenn2/blob/v0/submissions/README.md). Submissions are rolling,
through pull requests, with no closing date. Original project code, examples,
documentation, and submission templates use MIT; third-party materials retain
their own licenses.

## Two separate tracks

Both tracks use 4×4 MNIST inputs, 24 training and 50 evaluation images per class,
20 training passes, 20 ms startup, 1 ms/image, and the pinned ngspice 46 image.
The [task manifests](https://github.com/thomasnormal/spicenn2/blob/v0/competition/tasks/README.md) fix preprocessing, seeds, array content
hashes, and all runner settings. Public evaluation data means these are
development-set results, not scores on a secret final test set.

| Track | Reference circuit | Accuracy | Inference / image |
| --- | --- | --- | --- |
| `mnist10-v0` — main | Ten-digit delta rule, 1.67 µF | 18.6% (93/500) | 27.0879 µJ |
| `mnist017-v0` — starter | Three-digit delta rule, 16 µF | 89.33% (134/150) | 9.86249 µJ |

The main baseline is deliberately identified as weak; the starter score does not
carry over to ten digits. All three recorded baseline circuits passed a half-step
check with identical predictions and energy changes below 1%. The
[leaderboard](https://github.com/thomasnormal/spicenn2/blob/v0/competition/LEADERBOARD.md) keeps accuracy and energy separate and marks the
Pareto frontier; startup and training energy remain visible.

## Runner and workflow

- Continuous on-circuit training and evaluation; no Python weight updates.
- Fixed ngspice reference backend; Xyce supported for exploration and cross-checks.
- Long-transient waveform fix verified through the complete 5.32-second main task.
- All driven pins metered; returned energy cannot cancel gross delivered energy.
- Actionable validation/errors and retained debugging artifacts on failure.
- Streaming trace scoring; successful traces saved only with `--keep-trace`.
- Reproducible array-content hashes, independent of NumPy archive packaging.
- Submission scaffolding, MIT template, PR template, and automated checks.
- Restricted simulator container and independently rerun result records.

Energy is simulated circuit-boundary energy under simplified device models, not
computer electricity use or fabricated-chip efficiency. DAC/ADC/controller
implementation losses are excluded.

## Reference simulator download

The release asset `ngspice46-v0-image.tar.gz` contains the Linux/amd64 reference
image. See [loading and scoring instructions](https://github.com/thomasnormal/spicenn2/blob/v0/competition/container/README.md).

Archive SHA-256:
`ffad711075b754155b406ef7e49a8716f290a48ce209a6e653b17ffe2a44c045`

Image ID:
`sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d`

ngspice's notices, source archive, and build instructions are included in the
image. Its licenses and those of the base image are not replaced by MIT.
