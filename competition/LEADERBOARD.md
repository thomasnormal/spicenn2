# Circuit-learning leaderboard

Separate tracks; fixed 1 ms/image latency. See [v0 rules](RULES.md).
A frontier entry is not beaten on both accuracy and inference energy by another entry.
Startup and training energy are shown separately, not hidden in an amortized score.

These are public-development-set results, not estimates on a secret held-out set.
Only organizer reruns with matching reference-image/configuration metadata and a passing
half-timestep check are recorded here. Metadata checks alone do not authenticate a run.

[Reproduce a score and verify your reports](REPRODUCE.md) without changing the leaderboard.

## mnist10-v0

| Circuit | Accuracy | Inference / image | Startup | Training | Frontier | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| [Delta rule, 1.67 µF](../competition/examples/mnist_10.cir) | 18.60% (93/500) | 27.0879 µJ | 1.3381 mJ | 737.352 mJ | Yes | [Report](results/mnist10-v0/delta-rule-original/report.json) · [Half-step](results/mnist10-v0/delta-rule-original/verification.json) · [Identity](results/mnist10-v0/delta-rule-original/entry.json) |

## mnist017-v0

| Circuit | Accuracy | Inference / image | Startup | Training | Frontier | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| [Delta rule, 16 µF](../competition/examples/mnist_017_tuned.cir) | 89.33% (134/150) | 9.86249 µJ | 2.76849 mJ | 67.9689 mJ | Yes | [Report](results/mnist017-v0/delta-rule-16uf/report.json) · [Half-step](results/mnist017-v0/delta-rule-16uf/verification.json) · [Identity](results/mnist017-v0/delta-rule-16uf/entry.json) |
| [CMSUB trained-hidden backprop](../submissions/backprop-cmsub/circuit.cir) | 81.33% (122/150) | 59.3951 µJ | 14.9601 mJ | 174.66 mJ | No | [Report](results/mnist017-v0/backprop-cmsub/report.json) · [Half-step](results/mnist017-v0/backprop-cmsub/verification.json) · [Identity](results/mnist017-v0/backprop-cmsub/entry.json) |
| [Delta rule, 1.67 µF](../competition/examples/mnist_017.cir) | 66.00% (99/150) | 10.7243 µJ | 0.401435 mJ | 69.6601 mJ | No | [Report](results/mnist017-v0/delta-rule-original/report.json) · [Half-step](results/mnist017-v0/delta-rule-original/verification.json) · [Identity](results/mnist017-v0/delta-rule-original/entry.json) |
| [Fixed-feature PC control](../submissions/predictive-coding-frozen/circuit.cir) | 33.33% (50/150) | 34.599 µJ | 98.0771 mJ | 106.77 mJ | No | [Report](results/mnist017-v0/predictive-coding-frozen/report.json) · [Half-step](results/mnist017-v0/predictive-coding-frozen/verification.json) · [Identity](results/mnist017-v0/predictive-coding-frozen/entry.json) |
| [Continuous predictive coding (trained hidden)](../submissions/predictive-coding/circuit.cir) | 33.33% (50/150) | 41.9378 µJ | 227.918 mJ | 171.208 mJ | No | [Report](results/mnist017-v0/predictive-coding/report.json) · [Half-step](results/mnist017-v0/predictive-coding/verification.json) · [Identity](results/mnist017-v0/predictive-coding/entry.json) |

Regenerate with `python competition/leaderboard.py render`. CI checks metadata and table consistency;
it does not promote self-reported submissions to verified results.
