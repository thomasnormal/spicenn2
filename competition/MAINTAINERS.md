# Organizer scoring workflow

Use a trusted checkout of the v0 runner, not a PR's modified runner or workflow.
Review the submitted netlist and licensing before simulation. Do not run entrant
generators. Scoring should take place on a disposable machine without credentials;
the Docker backend adds restrictions but is not a replacement for that separation.

1. Validate `circuit.cir`, README, MIT license, attribution, and tuning disclosure.
2. Prepare the exact task dataset and verify its content hash.
3. Load the reference image identified in `release.json`. Do not silently substitute
   a rebuild with a different image ID. A rebuilt image can be tested locally, but
   only the frozen reference image is used for recorded v0 scores.
4. Run the named task, then rerun with half the maximum timestep.
5. Review both logs/reports and any eligibility concerns; record only successful,
   stable independent reruns. Commit the small reports and entry metadata, not traces.

Example for the starter baseline (replace the circuit and entry ID for an entrant):

```bash
python competition/prepare_mnist.py data/mnist-idx /tmp/scoring-mnist017.npz \
  --download --labels 0,1,7 --normalize zscore --train-per-class 24 --test-per-class 50 --seed 0
python competition/runner.py competition/examples/mnist_017_tuned.cir /tmp/scoring-mnist017.npz \
  --task mnist017-v0 --docker-image sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d \
  --output /tmp/scoring-reference
python competition/runner.py competition/examples/mnist_017_tuned.cir /tmp/scoring-mnist017.npz \
  --epochs 20 --startup 0.02 --step 5e-6 \
  --docker-image sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d \
  --output /tmp/scoring-half-step
python competition/leaderboard.py record --id baseline-tuned --name "Delta rule (16 uF)" \
  --task mnist017-v0 --circuit competition/examples/mnist_017_tuned.cir \
  --dataset /tmp/scoring-mnist017.npz --report /tmp/scoring-reference/report.json \
  --verification /tmp/scoring-half-step/report.json
python competition/leaderboard.py check
```

The recorder checks data, circuit, task, image, settings, counts, metered-source
energy totals, prediction/label agreement, regenerated harness hashes, and numerical
stability. It does not attest that arbitrary reports were honestly produced; the
organizer must actually perform the reruns. Do not let a PR add its own "verified"
result without that step. Review changes to rules, results, release manifests, and
the scorer with particular care.

For the main track, use all ten labels and `mnist10-v0`, with the same counts,
normalization, timing, and 20 epochs. No setting may differ between the reference
and numerical-check runs except maximum timestep. A verification failure is not
permission to change the task for that entry.

## Publication gate

Before declaring a release ready, verify the tutorials and both tracks, numerical
checks, reference-image export/import, reproducible dataset preparation, regenerated
leaderboard, submission scaffolding, license notices, and the CI jobs. Publish the
source commit, dataset specifications, and immutable image artifact together. Keep the previous
release available when changing anything that affects scoring.
