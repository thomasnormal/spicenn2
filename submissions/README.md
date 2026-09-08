# Submit a circuit

Submit through a pull request under the [v0 rules](../competition/RULES.md).
The ten-digit main track and the faster 0/1/7 starter track have separate
[leaderboard tables](../competition/LEADERBOARD.md). There is no closing date.
CI checks file presence, circuit syntax, runner tests, and recorded-result
consistency; it does not run your generator or award a score.

Looking for existing designs to study? The [historical solution catalog](HISTORICAL.md)
connects delta-rule, backprop, random-feature and predictive-coding research to
submission circuits, with explicit differences between old results and v0 scores.

## Create an entry

From your fork of the repository:

```bash
git switch -c submission/my-design
python competition/new_submission.py my-design --author "Your Name"
```

This creates the tuned 0/1/7 learner, an attribution-preserving MIT license, and a
README with working run commands. Edit the circuit, run those commands, and fill
in your results. The [example submission](example-delta-rule/) shows the full
directory structure. For the ten-digit main track, add `--baseline mnist_10`;
for the unranked two-input tutorial, add `--baseline blobs`.

Add a directory containing:

```text
submissions/your-design/
  circuit.cir
  README.md
  LICENSE
```

Use the [runner's circuit format](../competition/README.md). In your README, explain
the architecture and learning rule, whether anything was trained outside the circuit,
the exact runner command, dataset/preprocessing and seed, training passes, simulator
version, accuracy, training energy and inference energy per example. Include author
attribution. New submissions should use the [MIT License](../LICENSE), matching the
project: copy that license into your directory and add your own copyright notice.
If you adapt an existing example, retain its original copyright notice too. Only
submit work you have the right to license, and preserve any third-party notices.
Ask the organizer before submitting material under different terms. Add generator
source if the circuit was generated.

Do not commit downloaded datasets, full traces, simulator output directories, or
weight snapshots. You can quote a compact result summary in the README.
Small `report.json` files are welcome; generated `.npz`, `.log`, `.dat`, and full
traces are intentionally ignored. Root-level `.cir` files are also ignored because
that directory was used for generated experiments. Put your circuit under
`submissions/<name>/circuit.cir`, where Git will see it without `git add -f`.

## Open the pull request

```bash
python competition/validate.py submissions/my-design
git add submissions/my-design
git commit -m "Add my analog learning circuit"
git push -u origin submission/my-design
```

On GitHub, open a pull request **from your fork's branch to
`thomasnormal/spicenn2:master`** and complete the PR template. You do not need
write access to the main repository or to switch to the organizer's account.
Do not run someone else's submission generator on a machine with credentials.

The organizer reviews the circuit and reruns it in an isolated environment using
the common settings. Self-reported development scores do not become official scores
until reproduced. Results on different digit subsets or model/timing settings are
not directly comparable. Use the [named task](../competition/tasks/README.md)
for your track. The [reference image and resource limits](../competition/release.json)
and the [verification policy](../competition/RULES.md#verification-and-failures)
determine which reruns can enter the table.
