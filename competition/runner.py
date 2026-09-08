#!/usr/bin/env python3
"""Draft circuit-learning competition runner. See README.md before use."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

try:
    from .energy import integrate_power
except ImportError:
    from energy import integrate_power

PROTOCOL = "mnist4-circuit-learning-draft2"
INPUTS = [f"in{i}" for i in range(16)]
TARGETS = [f"target{i}" for i in range(10)]
OUTPUTS = [f"out{i}" for i in range(10)]
BIAS = {"vdd": 3.0, "vref": 0.5, "vweight": 1.8, "verror": 1.2, "voffset": 1.9, "vbias": 1.7}
SOURCES = list(BIAS) + ["reset", "learn", "clock"] + INPUTS + TARGETS
PINS = set(SOURCES + OUTPUTS + ["0"])
NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?(?:meg|[fpnumkgt])?", re.I)
SCALE = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6,
         "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}
# Explicit capacitance parameters: still a toy technology, not a foundry PDK.
MODELS = """.model nch nmos (level=1 vto=0.5 kp=120u lambda=0.02 gamma=0 phi=0.7
+ tox=20n cgso=0.3n cgdo=0.3n cj=0.2m cjsw=0.3n)
.model pch pmos (level=1 vto=-0.5 kp=40u lambda=0.02 gamma=0 phi=0.7
+ tox=20n cgso=0.3n cgdo=0.3n cj=0.2m cjsw=0.3n)
.model syn nmos (level=1 vto=0.2 kp=50u lambda=0 gamma=0 phi=0.7
+ tox=20n cgso=0.3n cgdo=0.3n)
.model diode d (is=1e-14 n=1 cj0=1p)"""


def number(token):
    if not NUMBER.fullmatch(token):
        raise ValueError(f"expected positive literal, got {token!r}")
    suffix = re.search(r"(meg|[fpnumkgt])$", token, re.I)
    value = float(token[:suffix.start()] if suffix else token)
    value *= SCALE[suffix[0].lower()] if suffix else 1
    if not math.isfinite(value) or value <= 0:
        raise ValueError("component values must be positive and finite")
    return value


def validate_submission(path):
    """Deliberately small grammar: flat, literal R/C/M/D components only.

    This is format validation, not a substitute for process isolation.
    """
    raw = path.read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError("submission exceeds 2 MB")
    lines, names, counts = [], set(), dict.fromkeys("rcmd", 0)
    for index, line in enumerate(raw.decode("ascii").splitlines(), 1):
        line = line.strip().lower()
        if not line or line.startswith("*"):
            continue
        fields = line.split()
        name = fields[0]
        kind = name[0]
        if not re.fullmatch(r"[rcmd][a-z0-9_]+", name) or name in names:
            raise ValueError(f"line {index}: invalid or duplicate component name")
        names.add(name)
        nodes = 4 if kind == "m" else 2
        expected = {"r": 4, "c": 4, "d": 4, "m": 8}[kind]
        if len(fields) != expected:
            raise ValueError(f"line {index}: wrong component syntax")
        for node in fields[1:1 + nodes]:
            if node not in PINS and not re.fullmatch(r"n_[a-z0-9_]+", node):
                raise ValueError(f"line {index}: internal nodes must start n_: {node}")
        if kind in "rc":
            value = number(fields[3])
            low, high = (1, 1e12) if kind == "r" else (1e-16, 1e-3)
            if not low <= value <= high:
                raise ValueError(f"line {index}: component outside draft bounds")
        elif kind == "d":
            if fields[3] != "diode":
                raise ValueError("only organizer diode model is allowed")
        else:
            if fields[5] not in ("nch", "pch", "syn"):
                raise ValueError("only organizer MOS models are allowed")
            if not fields[6].startswith("w=") or not fields[7].startswith("l="):
                raise ValueError("MOS syntax: Mname d g s b nch|pch|syn W=value L=value")
            for field in fields[6:]:
                if not 1e-6 <= number(field[2:]) <= 1e-2:
                    raise ValueError("MOS W/L must be between 1 um and 10 mm")
        counts[kind] += 1
        lines.append(line)
    if not lines or len(lines) > 20000:
        raise ValueError("submission must contain 1..20000 components")
    return "\n".join(lines), counts, hashlib.sha256(raw).hexdigest()


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in ("train_x", "train_y", "test_x", "test_y")}
    for split in ("train", "test"):
        x, y = arrays[f"{split}_x"], arrays[f"{split}_y"]
        if (x.ndim != 2 or x.shape[1] != 16 or len(x) == 0
                or y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer)
                or not np.isfinite(x).all() or np.any((x < 0) | (x > 1))
                or np.any((y < 0) | (y > 9))):
            raise ValueError(f"invalid {split} data; expected X Nx16 in [0,1], integer Y in 0..9")
    return arrays


def build_deck(circuit, data, epochs, seed, slot, startup, step, simulator="ngspice"):
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(data["train_y"])) for _ in range(epochs)]) if epochs else np.array([], dtype=int)
    test_order = rng.permutation(len(data["test_y"]))
    xs = np.vstack((data["train_x"][order], data["test_x"][test_order]))
    # Test labels never enter the deck or the simulator process.
    ys = np.r_[data["train_y"][order], np.full(len(test_order), -1)]
    ntrain = len(order)
    train_end = startup + ntrain * slot
    stop = startup + len(xs) * slot
    ramp = slot * 0.01
    options = ([".options reltol=1e-5 abstol=1e-12 vntol=1e-7 temp=27"] if simulator == "ngspice"
               else [f".options timeint reltol=1e-5 abstol=1e-12 delmax={step:.16g}", ".temp 27"])
    deck = ["Competition harness " + PROTOCOL, MODELS] + options
    values = {name: [] for name in SOURCES}
    for k, (x, y) in enumerate(zip(xs, ys)):
        for name, voltage in BIAS.items():
            values[name].append(voltage)
        values["reset"].append(0.0)
        values["learn"].append(3.0 if k < ntrain else 0.0)
        values["clock"].append(0.0)
        for name, value in zip(INPUTS, x):
            values[name].append(0.3 + 1.4 * float(value))
        for c, name in enumerate(TARGETS):
            values[name].append(1.0 if y == c else 0.0)
    for name in SOURCES:
        initial = 3.0 if name == "reset" else BIAS.get(name, 0.0)
        points = [(0.0, 0.0), (ramp, initial)]
        if name == "reset":
            points.extend([(startup - ramp, 3.0), (startup, 0.0)])
        else:
            points.append((startup, initial))
        for k, val in enumerate(values[name]):
            begin = startup + k * slot
            if name == "clock":
                points.extend([(begin + ramp, 3.0), (begin + slot / 2, 3.0),
                               (begin + slot / 2 + ramp, 0.0), (begin + slot, 0.0)])
            else:
                points.extend([(begin + ramp, val), (begin + slot, val)])
        # Continuations avoid enormous individual SPICE lines.
        deck.append(f"V_{name} {name} 0 PWL(")
        deck.extend(f"+ {t:.16g} {v:.16g}" for t, v in points)
        deck.append("+ )")
    deck.extend([".subckt submission " + " ".join(SOURCES + OUTPUTS), circuit, ".ends submission",
                 "Xentry " + " ".join(SOURCES + OUTPUTS) + " submission"])
    for i, name in enumerate(OUTPUTS):
        deck.extend([f"R_load{i} {name} 0 1g", f"C_load{i} {name} 0 1p"])
    if simulator == "xyce":
        deck.extend([f".tran {step:.16g} {stop:.16g} 0 {step:.16g} uic",
                     ".print tran format=csv file=trace.txt precision=16 "
                     + " ".join([f"V({n})" for n in OUTPUTS]
                                + [f"{{-V({n})*I(V_{n})}}" for n in SOURCES]), ".end"])
        return "\n".join(deck) + "\n", test_order, train_end, stop
    deck.extend([".control", "set noaskquit", "set wr_singlescale", "set wr_vecnames",
                 "set numdgt=15", f"tran {step:.16g} {stop:.16g} 0 {step:.16g} uic"])
    # SPICE current points into the positive terminal: negate for delivered power.
    for name in SOURCES:
        deck.append(f"let p_{name} = -v({name}) * i(V_{name})")
    deck.extend(["wrdata trace.txt " + " ".join([f"v({n})" for n in OUTPUTS] + [f"p_{n}" for n in SOURCES]),
                 "quit", ".endc", ".end"])
    return "\n".join(deck) + "\n", test_order, train_end, stop


def score_trace(trace, labels, train_end, stop, slot, startup):
    if trace.ndim != 2 or trace.shape[1] != 1 + len(OUTPUTS) + len(SOURCES):
        raise ValueError("unexpected simulator trace columns")
    t, scores, power = trace[:, 0], trace[:, 1:11], trace[:, 11:]
    if not np.isfinite(trace).all() or np.any(np.diff(t) <= 0):
        raise ValueError("non-finite or non-monotonic simulator trace")
    # uic starts at a small positive time. All source voltages are zero at t=0.
    if t[0] > 0:
        t = np.r_[0, t]
        scores = np.vstack((np.zeros(10), scores))
        power = np.vstack((np.zeros(len(SOURCES)), power))
    # Decimal trace output can round the requested end time by a few ulps.
    if abs(t[-1] - stop) <= 1e-12 * max(abs(stop), 1e-12):
        t[-1] = stop
    if t[-1] < stop:
        raise ValueError("simulation did not reach the scoring deadline")
    predictions, correct, invalid = [], 0, 0
    for i, label in enumerate(labels):
        sample = train_end + (i + 0.9) * slot
        voltages = np.array([np.interp(sample, t, col) for col in scores.T])
        winners = np.flatnonzero(voltages.max() - voltages <= 1e-6)
        prediction = int(winners[0]) if len(winners) == 1 else -1
        invalid += prediction == -1
        correct += prediction == label
        predictions.append(prediction)
    windows = {"startup": (0, startup), "inference": (train_end, stop), "total": (0, stop)}
    if train_end > startup:
        windows["training"] = (startup, train_end)
    energy = {}
    for phase, (begin, end) in windows.items():
        measured = integrate_power(t, power, begin, end)
        energy[phase] = {key: float(value.sum()) for key, value in measured.items()}
        energy[phase]["by_source"] = {name: {key: float(value[i]) for key, value in measured.items()}
                                      for i, name in enumerate(SOURCES)}
    energy["inference"]["delivered_j_per_image"] = energy["inference"]["delivered_j"] / len(labels)
    confusion = np.zeros((10, 11), dtype=int)
    for label, prediction in zip(labels, predictions):
        confusion[label, prediction if prediction >= 0 else 10] += 1
    return {"accuracy": float(correct / len(labels)), "correct": int(correct),
            "test_images": len(labels), "invalid_predictions": int(invalid),
            "confusion_matrix": confusion.tolist(), "energy": energy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new directory for report and trace")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slot", type=float, default=1e-3)
    parser.add_argument("--startup", type=float, default=1e-3)
    parser.add_argument("--step", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--simulator", choices=("ngspice", "xyce"), default="ngspice")
    parser.add_argument("--binary", help="path to simulator; defaults to ngspice or Xyce on PATH")
    args = parser.parse_args()
    if args.epochs < 0 or args.seed < 0 or not all(math.isfinite(v) and v > 0 for v in (args.slot, args.startup, args.step, args.timeout)):
        parser.error("epochs/seed must be nonnegative and timing values positive and finite")
    if args.step > args.slot / 100 or args.startup <= args.slot / 50:
        parser.error("step must be <= slot/100 and startup > slot/50")
    circuit, counts, circuit_hash = validate_submission(args.submission)
    data = load_data(args.dataset)
    deck, order, train_end, stop = build_deck(circuit, data, args.epochs, args.seed, args.slot, args.startup, args.step, args.simulator)
    binary = shutil.which(args.binary or ("ngspice" if args.simulator == "ngspice" else "Xyce"))
    if not binary:
        parser.error("simulator not found")
    version = subprocess.run([binary, "--version" if args.simulator == "ngspice" else "-v"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    args.output.mkdir(parents=True, exist_ok=False)
    # No dataset files in the simulation working directory.
    with tempfile.TemporaryDirectory(prefix="circuit-bench-") as temporary:
        work = Path(temporary)
        (work / "harness.cir").write_text(deck)
        with (args.output / "simulator.log").open("w") as log:
            command = [binary, "-n", "-b", "harness.cir"] if args.simulator == "ngspice" else [binary, "harness.cir"]
            result = subprocess.run(command, cwd=work,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        if result.returncode or not (work / "trace.txt").exists():
            raise RuntimeError("simulator failed; see simulator.log")
        trace = np.loadtxt(work / "trace.txt", skiprows=1, ndmin=2,
                           delimiter="," if args.simulator == "xyce" else None,
                           comments="End of")
        report = score_trace(trace, data["test_y"][order], train_end, stop, args.slot, args.startup)
        shutil.copyfile(work / "trace.txt", args.output / "trace.txt")
    report.update({"protocol": PROTOCOL, "simulator": version, "components": counts,
                   "submission_sha256": circuit_hash,
                   "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                   "harness_sha256": hashlib.sha256(deck.encode()).hexdigest(),
                   "settings": {key: getattr(args, key) for key in ("simulator", "epochs", "seed", "slot", "startup", "step")},
                   "training_presentations": args.epochs * len(data["train_y"]),
                   "inference_latency_s": args.slot, "status": "draft_local_benchmark"})
    (args.output / "harness.cir").write_text(deck)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Accuracy: {report['accuracy']:.2%}; inference: {report['energy']['inference']['delivered_j_per_image']:.6g} J/image")
    print(args.output / "report.json")


if __name__ == "__main__":
    main()
