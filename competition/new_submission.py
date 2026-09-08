#!/usr/bin/env python3
"""Create a runnable, MIT-licensed submission folder from a bundled example."""
import argparse
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="directory name under submissions/, e.g. my-delta-rule")
    parser.add_argument("--author", required=True, help="your name for attribution")
    parser.add_argument("--baseline", choices=("mnist_10", "mnist_017", "mnist_017_tuned", "blobs"), default="mnist_017_tuned")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.name):
        parser.error("name must contain 1..64 lowercase letters, digits or hyphens and begin with a letter/digit")
    if not args.author.strip() or any(ord(c) < 32 for c in args.author):
        parser.error("author must be a nonempty, single-line name")
    path = ROOT / "submissions" / args.name
    if path.exists():
        parser.error(f"submission already exists: {path}; choose another name")
    try:
        path.mkdir()
        shutil.copyfile(ROOT / "competition/examples" / (args.baseline + ".cir"), path / "circuit.cir")
        license_text = (ROOT / "LICENSE").read_text()
        license_text = license_text.replace("and contributors\n", f"and contributors\nCopyright (c) 2026 {args.author}\n", 1)
        (path / "LICENSE").write_text(license_text)
        toy = args.baseline == "blobs"
        full = args.baseline == "mnist_10"
        labels = "0,1,2,3,4,5,6,7,8,9" if full else "0,1,7"
        task = "blobs-dev" if toy else "mnist10-v0" if full else "mnist017-v0"
        data = "/tmp/my-blobs.npz" if toy else "/tmp/my-mnist10.npz" if full else "/tmp/my-mnist017.npz"
        prepare = ("python competition/prepare_toy.py /tmp/my-blobs.npz" if toy else
                   f"python competition/prepare_mnist.py data/mnist-idx {data} --download --labels {labels} --normalize zscore")
        command = (f"python competition/runner.py submissions/{args.name}/circuit.cir {data} "
                   f"--task {task} --output /tmp/{args.name}-score")
        (path / "README.md").write_text(f"""# {args.name}

Author: {args.author}

Starting point: `competition/examples/{args.baseline}.cir`, by Thomas Dybdahl Ahle
and contributors, under MIT. This folder is an unmodified starter until edited;
it is not an independently verified leaderboard entry.

## Design

Analog single-layer delta-rule learner. Capacitors store weights, transistor
currents update them during training, and the circuit produces the class scores.
There is no external optimizer or pretrained weight initialization.

Describe your changes here, including how you chose component values and which
datasets/scores you used for tuning. Distinguish circuit learning from offline
design or hyperparameter optimization. Disclose any pretrained information.

## Reproduce

From the repository root (use new output paths for each experiment):

```bash
{prepare}
{command}
python competition/validate.py submissions/{args.name}
```

## Results

Replace this section with measured results; do not submit invented scores.
Include dataset content hash, protocol, simulator version, exact settings,
accuracy, invalid predictions, startup energy, training energy, and inference
J/image. Add a small `report.json` if desired; do not commit full traces or datasets.

## License

MIT; see LICENSE. Original example attribution is retained.
""")
    except OSError as error:
        parser.exit(2, f"error: {error}; inspect {path} for any partially created files\n")
    print(path)
    print("Edit circuit.cir and README.md, run the benchmark, then open a pull request.")


if __name__ == "__main__":
    main()
