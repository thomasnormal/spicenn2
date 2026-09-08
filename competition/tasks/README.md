# Reproducible tasks

Use a named task to select the complete configuration. The two v0 handwriting
tracks are ranked separately under the [competition rules](../RULES.md).

| Task | Data | Training passes | Startup | Slot / maximum step | Backend |
| --- | --- | --- | --- | --- | --- |
| `blobs-dev` | 40 training / 100 test points, generator seed 7 | 10 | 20 ms | 1 ms / 10 µs | ngspice 46 |
| `mnist017-v0` — starter | Digits 0,1,7; 24 training / 50 test images per class, selection seed 0, z-score normalization | 20 | 20 ms | 1 ms / 10 µs | ngspice 46 |
| `mnist10-v0` — main | All ten digits; same counts/preprocessing as above | 20 | 20 ms | 1 ms / 10 µs | ngspice 46 |

All runner shuffle seeds are 0. Main-track duration is 5.32 simulated seconds;
the starter is 1.61 seconds. Each JSON file contains the dataset content hash, so
different examples or preprocessing cannot silently enter the same task.
`mnist017-dev` and `mnist10-dev` retain the same development settings, but use the
`-v0` identifiers for submissions. Blobs is a tutorial, not a ranked track.

Preparation downloads the original MNIST training/test archives from the public
mirror, checks their checksums, and reproduces the fixed subsets. The manifests
specify the exact dataset; downloaded archives and arrays are not committed to Git.

```bash
python competition/prepare_mnist.py data/mnist-idx /tmp/task-mnist017.npz \
  --download --labels 0,1,7 --normalize zscore --train-per-class 24 --test-per-class 50 --seed 0
python competition/runner.py competition/examples/mnist_017_tuned.cir /tmp/task-mnist017.npz \
  --task mnist017-v0 --output /tmp/task-mnist017-score
```

`--task` fills in all listed runner settings and checks both dataset contents and
simulator version before simulation. Conflicting flags are rejected. To experiment
with different timing, seeds, or Xyce, omit `--task` and supply the custom settings
explicitly. A custom run must not be compared as though it used the named task.

For the main track, change the labels to `0,1,2,3,4,5,6,7,8,9`, choose new dataset
and output paths, and use `competition/examples/mnist_10.cir` with `--task mnist10-v0`.

Reports include `task` and `task_sha256`, in addition to the actual settings,
dataset, and harness hashes. Organizer scoring additionally requires the
[reference container image](../container/README.md) and a half-step numerical check.
Local results are not automatically organizer-verified entries.

MNIST is by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges; see its
[original site](https://yann.lecun.org/exdb/mnist/index.html) and
[CVDF mirror](https://github.com/cvdfoundation/mnist). The project's MIT license
does not relicense this third-party dataset.
