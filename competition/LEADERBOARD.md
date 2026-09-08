# Circuit-learning leaderboard

Separate tracks; fixed 1 ms/image latency. See [v0 rules](RULES.md).
A frontier entry is not beaten on both accuracy and inference energy by another entry.
Startup and training energy are shown separately, not hidden in an amortized score.

These are public-development-set results, not estimates on a secret held-out set.
Only organizer reruns with matching reference-image/configuration metadata and a passing
half-timestep check are recorded here. Metadata checks alone do not authenticate a run.

## mnist10-v0

| Circuit | Accuracy | Inference / image | Startup | Training | Frontier |
| --- | --- | --- | --- | --- | --- |
| [Delta rule, 1.67 µF](results/mnist10-v0/delta-rule-original/entry.json) | 18.60% (93/500) | 27.0879 µJ | 1.3381 mJ | 737.352 mJ | Yes |

## mnist017-v0

| Circuit | Accuracy | Inference / image | Startup | Training | Frontier |
| --- | --- | --- | --- | --- | --- |
| [Delta rule, 16 µF](results/mnist017-v0/delta-rule-16uf/entry.json) | 89.33% (134/150) | 9.86249 µJ | 2.76849 mJ | 67.9689 mJ | Yes |
| [Delta rule, 1.67 µF](results/mnist017-v0/delta-rule-original/entry.json) | 66.00% (99/150) | 10.7243 µJ | 0.401435 mJ | 69.6601 mJ | No |

Regenerate with `python competition/leaderboard.py render`. CI checks metadata and table consistency;
it does not promote self-reported submissions to verified results.
