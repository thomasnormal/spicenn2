# Reproduce a published score

This is for readers and entrants, not just organizers. You will reproduce the
16 µF starter baseline, run the half-timestep check, and validate the results
**without changing the leaderboard**. Plan for two runs of a few minutes each.
The all-ten-digit task takes much longer and has a separate score.

Use a current checkout of this repository with its Python virtual environment
activated as described in the [README](../README.md). The read-only `verify`
command was added after the original v0 release. To keep the actual simulations
pinned to the original source, use a separate frozen checkout below.

## 1. Pin the runner and obtain the image

Run all commands on this page from your **current repository directory**. Do not
`cd` into the frozen checkout: the new verification tool lives in the current one.
The paths below must be unused; choose other paths consistently if they exist.

```bash
git clone --depth 1 --branch v0 https://github.com/thomasnormal/spicenn2.git /tmp/spicenn2-v0-source
git -C /tmp/spicenn2-v0-source rev-parse HEAD
```

Expected commit: `822cac19b94e064aa7be3cff4c748797c55ecfc7`.
The `v0` tag and its original runner remain frozen; newer documentation and tools
do not silently replace that source. The commands below explicitly use its runner,
task files, data preparation, and circuit.

Install Docker if needed, then [download, check, and load the reference
image](container/README.md#reference-image-artifact). Run as a non-root user with
access to a local Docker daemon. No native ngspice installation or image rebuild
is needed for this path. The image is Linux/amd64; native ARM scores are exploratory.

## 2. Prepare the data and run the circuit

```bash
python /tmp/spicenn2-v0-source/competition/prepare_mnist.py data/mnist-idx /tmp/replay-mnist017.npz \
  --download --labels 0,1,7 --normalize zscore --train-per-class 24 --test-per-class 50 --seed 0
python /tmp/spicenn2-v0-source/competition/runner.py \
  /tmp/spicenn2-v0-source/competition/examples/mnist_017_tuned.cir /tmp/replay-mnist017.npz \
  --task mnist017-v0 --docker-image sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d \
  --output /tmp/replay-reference
python /tmp/spicenn2-v0-source/competition/runner.py \
  /tmp/spicenn2-v0-source/competition/examples/mnist_017_tuned.cir /tmp/replay-mnist017.npz \
  --epochs 20 --startup 0.02 --step 5e-6 \
  --docker-image sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d \
  --output /tmp/replay-half-step
```

The half-step run intentionally omits `--task`: that named task fixes a 10 µs
step, while the numerical check needs 5 µs. All other settings stay the same.
The checker below enforces that. During either run you can watch its
`simulator.log` in another terminal. Each output directory must be new; the data
file can be reused. Expect **134/150 correct**, zero invalid predictions, and
about **9.86249 µJ/image**. Runtime depends on your computer.

## 3. Check your reports, read-only

Use the new tool in your current checkout, not the frozen v0 directory:

```bash
python competition/leaderboard.py verify --task mnist017-v0 \
  --circuit /tmp/spicenn2-v0-source/competition/examples/mnist_017_tuned.cir \
  --dataset /tmp/replay-mnist017.npz --report /tmp/replay-reference/report.json \
  --verification /tmp/replay-half-step/report.json \
  --against competition/results/mnist017-v0/delta-rule-16uf/report.json
```

Expect three `PASS` lines: comparison with the published result; circuit, data,
image, settings, harness and score consistency; and half-step stability. Both
comparisons require identical prediction lists and each phase's delivered energy
within 1%. Wall-clock runtime and archive packaging bytes need not match.

The checker regenerates harness hashes and verifies predictions against dataset
labels. It reads reports; it does not rerun SPICE, prove that reports are honest,
or award an official score. No result or leaderboard files are written. By
contrast, `leaderboard.py check` checks only already-recorded repository results,
and `record` is the organizer's write operation.

For your own design, use its circuit in both runs and in `verify`, and **omit
`--against`**: a different design is not expected to match the baseline. Follow
the [submission instructions](../submissions/README.md) when ready. Never present
self-checked reports as organizer-verified leaderboard entries.
