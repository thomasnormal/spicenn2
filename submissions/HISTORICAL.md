# Historical solutions, represented as submissions

The research archive contains several genuinely different learning circuits.
This catalog connects the strongest recorded families to flat circuit submissions
that use the public runner. **An old research accuracy is not a v0 score.**
Only fresh, matching reference-image and half-step measurements belong on the
[leaderboard](../competition/LEADERBOARD.md).

Fresh full-task measurements include **81.33%** for the CMSUB backprop port and
**33.33%** for each PC port. The lower PC scores are reported, not replaced by the
historical scores. See each entry for energy, controls and numerical verification.

| Family | Submission | What learns electrically? | Earlier evidence and caveats |
| --- | --- | --- | --- |
| DIRECT delta rule | [Tuned linear learner](example-delta-rule/) | Output weights and biases | Public starter baseline: 134/150, 89.33%; already reference-verified |
| CMSUB backprop | [Backprop through a hidden layer](backprop-cmsub/) | Hidden and output weights, using a shared-weight transpose backward path | MNIST 0/1/7: 72–84.7% across eight initializations in the older study |
| Random features + PERAZ | [Physical random-feature port](random-feature-peraz/) | Readout weights; hidden projections are fixed | Ten-digit MNIST: 74.7%, or 75.0% with more training data; historical features were computed in Python |
| Predictive coding | [Trainable-hidden PC port](predictive-coding/) | Hidden and output weights via local prediction errors | Completed older ten-digit MNIST experiment: 45.2% peak, 44.4% final |
| Fixed-feature PC control | [Frozen-hidden PC port](predictive-coding-frozen/) | Output weights only | Older ten-digit MNIST fixed-feature control reached 49.6% |

The new ports initially target the smaller `mnist017-v0` task. They preserve their
stated circuit mechanisms, **not every architecture, cell parameter or evaluation
procedure of the historical best run**. In particular, the historical deep PC
architecture was 64→36→16→10; its smaller starter port is not a ten-digit reproduction.
Each entry explains its departures, includes a data-independent generator and MIT
license, and separates fresh measurements from historical provenance.

## Why not simply submit the old netlists unchanged?

Earlier experiments were exploratory rather than standardized competitions:

- The backprop and keystone pipelines initialized capacitors with ideal `.ic`
  states, then evaluated a separate circuit with late weight samples averaged
  in Python. v0 requires metered physical initialization and evaluation in the
  same transient, with no external weight processing.
- The strongest keystone experiment's 96 nonlinear features were calculated in
  Python from 64 pixels. Those operations and their energy were outside its
  submitted analog readout. The new port constructs nonlinear features from
  real components and counts their energy.
- The PC experiments used different voltage models, a different dataset/schedule,
  and evaluation-block output centering. Some research diagnostics also used
  test-label-based output permutation; that is not allowed in a v0 classifier.
  v0 directly compares the ten labeled output voltages, without either adjustment.
- Results on scikit-learn's smaller `digits` dataset are not MNIST results. Some
  stronger PC numbers in the research ledger belong to that different dataset.
- Earlier reports of long-run PC “collapse” were subsequently corrected when
  complete simulator output showed learning followed by a plateau. The final
  completed-run evidence is used here, not the truncated-run interpretation.

The predictive-coding port is described as local predictive coding rather than
silently labeled exact equilibrium propagation. The old CHL/`SYMNUDGE` experiment
flags did not by themselves establish a functioning symmetric free/positive/negative
phase sequence; its phase priority could leave the negative clock inactive.

## Locate the original designs

These are research source links, not commands to execute for a competition entry.
Some historical scripts expect proprietary simulators, local datasets and scratch
files; use the self-contained submission instructions above for the public runner.

- [Backprop study](../docs/BACKPROP_DEPTH.md),
  [backprop recipe](../experiments/run_backprop.sh), and
  [multiclass circuit generator](../experiments/gen_mc.py).
- [Keystone/PERAZ recipe](../experiments/run_iso.sh), including the exact original
  Python random-feature construction and the older training/evaluation pipeline.
- [Deep PC generator](../pc_deep.py), [research ledger](../docs/IDEAS_LEDGER.md), and
  [curated later PC provenance](predictive-coding/historical-provenance.json).
  The last record preserves the later completed-run correction and artifact hashes
  without bundling large historical simulator files. It records the MNIST sequence
  `[0.180, 0.452, 0.444, 0.436, 0.440, 0.444]`.

Keep both trained-hidden and frozen-hidden entries: an architecture name or a
changing capacitor is not evidence that hidden-layer learning improves accuracy.
Matched controls, the public dataset, complete runs and measured electrical energy
make that question testable.
The frozen PC entry is an architectural control, not a tightly matched ablation:
it also removes hidden error/feedback circuitry and continuously drives its fixed
weights. The backprop short controls, in contrast, change only update-enable wiring.
