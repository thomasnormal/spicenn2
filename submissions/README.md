# Submit a circuit

We are collecting circuit designs while the official competition configuration is
being finalized. There is no official leaderboard or deadline yet. A pull request
is the submission mechanism; no upload service or GitHub Actions job executes it.

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
attribution and choose a license that permits the organizer to share and evaluate
your circuit. Add generator source if the circuit was generated.

Do not commit downloaded datasets, full traces, simulator output directories, or
weight snapshots. You can quote a compact result summary in the README.

The organizer reviews the circuit and reruns it in an isolated environment using
the common settings. Self-reported development scores do not become official scores
until reproduced. Results on different digit subsets or model/timing settings are
not directly comparable. The common final dataset, simulator build, resource budgets
and ranking policy will be published before official scoring.
