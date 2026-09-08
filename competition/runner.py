#!/usr/bin/env python3
"""Circuit-learning competition runner. See README.md for the v0 protocol."""
import argparse
import gzip
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

import numpy as np

try:
    from .container_backend import ContainerBackend
    from .data import content_hash, load_data
    from .energy import integrate_power
except ImportError:
    from container_backend import ContainerBackend
    from data import content_hash, load_data
    from energy import integrate_power

PROTOCOL = "mnist4-circuit-learning-v0"
TASK_DIRECTORY = Path(__file__).resolve().parent / "tasks"
DEFAULT_SETTINGS = {"epochs": 1, "seed": 0, "slot": 1e-3, "startup": 1e-3,
                    "step": 1e-5, "simulator": "ngspice"}
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
    with path.open("rb") as source:
        raw = source.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("submission exceeds 2 MB")
    lines, names, counts = [], set(), dict.fromkeys("rcmd", 0)
    try:
        source = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("circuit must be UTF-8 text; non-ASCII text is allowed only in comments") from error
    for index, line in enumerate(source.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        line = line.split(";", 1)[0].strip().lower()
        if not line:
            continue
        if not line.isascii():
            raise ValueError(f"line {index}: device syntax must be ASCII; use 'u' for micro and omit unit symbols")
        if line.startswith("."):
            raise ValueError(f"line {index}: {line.split()[0]} is not allowed; submit only devices, without a title or .end")
        fields = line.split()
        name = fields[0]
        kind = name[0]
        if not re.fullmatch(r"[rcmd][a-z0-9_]+", name) or name in names:
            raise ValueError(f"line {index}: invalid or duplicate component name; use R/C/M/D devices and prefix title comments with '*'")
        names.add(name)
        nodes = 4 if kind == "m" else 2
        expected = {"r": 4, "c": 4, "d": 4, "m": 8}[kind]
        if len(fields) != expected:
            syntax = {"r": "Rname node1 node2 value", "c": "Cname node1 node2 value",
                      "d": "Dname node1 node2 diode", "m": "Mname drain gate source body nch|pch|syn W=value L=value"}
            raise ValueError(f"line {index}: expected {syntax[kind]}; extra parameters (including M= or IC=) are not allowed")
        fields[1:1 + nodes] = ["0" if node == "gnd" else node for node in fields[1:1 + nodes]]
        for node in fields[1:1 + nodes]:
            if node not in PINS and not re.fullmatch(r"n_[a-z0-9_]+", node):
                raise ValueError(f"line {index}: internal nodes must start n_: {node}")
        if kind in "rc":
            value = number(fields[3])
            low, high = (1, 1e12) if kind == "r" else (1e-16, 1e-3)
            if not low <= value <= high:
                raise ValueError(f"line {index}: component outside allowed bounds")
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
        lines.append(" ".join(fields))
    if not lines or len(lines) > 20000:
        raise ValueError("submission must contain 1..20000 components")
    return "\n".join(lines), counts, hashlib.sha256(raw).hexdigest()


def compact_pwl(points):
    """Remove redundant interior points in constant-voltage runs.

    This preserves the piecewise-linear waveform exactly, including every ramp.
    This reduces deck size without changing any experiment timing. Do not merge
    close *distinct* times or values: that would change the experiment.
    """
    return [point for i, point in enumerate(points)
            if i == 0 or i == len(points) - 1
            or not points[i-1][1] == point[1] == points[i+1][1]]


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
        if simulator == "ngspice" and name == "clock":
            deck.append(f"V_clock clock 0 PULSE(0 3 {startup:.16g} {ramp:.16g} {ramp:.16g} {slot/2-ramp:.16g} {slot:.16g})")
        elif simulator == "ngspice":
            # Native voltage-source PWL breakpoints in ngspice 46 can force
            # zero-progress timesteps near t=4 s (TMAX=10 us). Its behavioral
            # pwl(time,...) has the SAME voltage waveform but a different
            # evaluation path. A series 0 V source meters current at each pin.
            # The real clock PULSE supplies every slot/ramp breakpoint; the
            # disconnected startup guard below supplies startup ramp corners.
            # B sources belong only to the harness, never to submitted circuits.
            deck.append(f"V_{name} {name} h_drive_{name} 0")
            deck.append(f"B_drive_{name} h_drive_{name} 0 V=pwl(time,")
            samples = compact_pwl(points)
            deck.extend(f"+ {t:.16g}, {v:.16g}" + ("," if i < len(samples)-1 else ")")
                        for i, (t, v) in enumerate(samples))
        else:
            # Continuations avoid enormous individual SPICE lines.
            deck.append(f"V_{name} {name} 0 PWL(")
            deck.extend(f"+ {t:.16g} {v:.16g}" for t, v in compact_pwl(points))
            deck.append("+ )")
    if simulator == "ngspice":
        # Unconnected to the submission: this timing-only source delivers zero
        # current/energy. It forces t=ramp, startup-ramp, startup breakpoints.
        deck.append(f"V_startup_guard h_startup_guard 0 PULSE(0 1 0 {ramp:.16g} {ramp:.16g} {startup-2*ramp:.16g} {2*stop:.16g})")
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
    # Store only the vectors needed for scoring, not every internal weight/node.
    saved = [f"v({name})" for name in OUTPUTS + SOURCES] + [f"i(V_{name})" for name in SOURCES]
    deck.extend([".control", "set noaskquit", "set wr_singlescale", "set wr_vecnames",
                 "set numdgt=17", "save " + " ".join(saved),
                 f"tran {step:.16g} {stop:.16g} 0 {step:.16g} uic"])
    # SPICE current points into the positive terminal: negate for delivered power.
    for name in SOURCES:
        deck.append(f"let p_{name} = -v({name}) * i(V_{name})")
    deck.extend(["wrdata trace.txt " + " ".join([f"v({n})" for n in OUTPUTS] + [f"p_{n}" for n in SOURCES]),
                 "quit", ".endc", ".end"])
    return "\n".join(deck) + "\n", test_order, train_end, stop


def trace_chunks(path, simulator, rows=8192):
    """Read ASCII simulator output in bounded batches, without loading it all."""
    if rows < 1:
        raise ValueError("trace chunk size must be positive")
    with path.open() as source:
        next(source, None)  # Column header.
        lines = (line for line in source if line.strip() and not line.startswith("End of"))
        while True:
            batch = list(islice(lines, rows))
            if not batch:
                return
            yield np.loadtxt(batch, ndmin=2, delimiter="," if simulator == "xyce" else None)


def score_trace(trace, labels, train_end, stop, slot, startup):
    """In-memory entry point, using the same scorer as large streamed runs."""
    return score_chunks([trace], labels, train_end, stop, slot, startup)


def score_chunks(chunks, labels, train_end, stop, slot, startup):
    if not len(labels) or not 0 < startup <= train_end < stop or slot <= 0:
        raise ValueError("invalid scoring schedule")
    windows = {"startup": (0, startup), "inference": (train_end, stop), "total": (0, stop)}
    if train_end > startup:
        windows["training"] = (startup, train_end)
    totals = {phase: {key: np.zeros(len(SOURCES)) for key in ("delivered_j", "returned_j", "net_j")}
              for phase in windows}
    samples = train_end + (np.arange(len(labels)) + 0.9) * slot
    predictions = []
    previous = None
    for chunk in chunks:
        if chunk.ndim != 2 or chunk.shape[1] != 1 + len(OUTPUTS) + len(SOURCES) or not len(chunk):
            raise ValueError("unexpected simulator trace columns or empty trace")
        # Carry one row so interpolation and energy across chunk boundaries are
        # identical to one full trace. Never silently discard duplicate times.
        trace = np.vstack((previous, chunk)) if previous is not None else chunk.copy()
        if not np.isfinite(trace).all() or np.any(np.diff(trace[:, 0]) <= 0):
            raise ValueError("non-finite or non-monotonic simulator trace")
        if previous is None:
            if not 0 <= trace[0, 0] <= min(startup, slot) / 100:
                raise ValueError("simulator trace is missing startup samples")
            # uic begins at a small positive time; all sources are zero at t=0.
            if trace[0, 0] > 0:
                trace = np.vstack((np.zeros(trace.shape[1]), trace))
        # Decimal trace output can round the requested end time by a few ulps.
        if abs(trace[-1, 0] - stop) <= 1e-12 * max(abs(stop), 1e-12):
            trace[-1, 0] = stop
        t, scores, power = trace[:, 0], trace[:, 1:11], trace[:, 11:]
        previous = trace[-1:].copy()
        if len(trace) < 2:
            continue
        end_index = int(np.searchsorted(samples, t[-1], side="right"))
        selected = samples[len(predictions):end_index]
        if len(selected):
            voltages = np.array([np.interp(selected, t, col) for col in scores.T]).T
            winners = voltages.max(axis=1, keepdims=True) - voltages <= 1e-6
            predictions.extend(np.where(winners.sum(axis=1) == 1, voltages.argmax(axis=1), -1).tolist())
        for phase, (begin, end) in windows.items():
            begin, end = max(begin, t[0]), min(end, t[-1])
            if begin < end:
                measured = integrate_power(t, power, begin, end)
                for key, value in measured.items():
                    totals[phase][key] += value
    if previous is None or previous[0, 0] < stop or len(predictions) != len(labels):
        raise ValueError("simulation did not reach the scoring deadline")
    correct = int(np.count_nonzero(np.array(predictions) == labels))
    invalid = predictions.count(-1)
    energy = {}
    for phase, measured in totals.items():
        energy[phase] = {key: float(value.sum()) for key, value in measured.items()}
        energy[phase]["by_source"] = {name: {key: float(value[i]) for key, value in measured.items()}
                                      for i, name in enumerate(SOURCES)}
    energy["inference"]["delivered_j_per_image"] = energy["inference"]["delivered_j"] / len(labels)
    confusion = np.zeros((10, 11), dtype=int)
    for label, prediction in zip(labels, predictions):
        confusion[label, prediction if prediction >= 0 else 10] += 1
    return {"accuracy": float(correct / len(labels)), "correct": int(correct),
            "test_images": len(labels), "invalid_predictions": int(invalid),
            "predictions": predictions,
            "confusion_matrix": confusion.tolist(), "energy": energy}


def simulator_identity(simulator, version_output):
    if simulator == "ngspice":
        match = re.search(r"\bngspice-([^\s:]+)", version_output, re.I)
        if match:
            return "ngspice-" + match[1]
    lines = [line.strip() for line in version_output.splitlines() if line.strip()]
    return next((line for line in lines if "xyce" in line.lower()), lines[0] if lines else simulator)


def safe_trace(path):
    """Simulator artifacts are untrusted, including filesystem links."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("simulator trace must be a regular file, not a link or special file")
    if info.st_size > 2_147_483_648:
        raise ValueError("simulator trace exceeds the 2 GiB file limit")
    return path


def run(args):
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}; choose a new --output directory")
    circuit, counts, circuit_hash = validate_submission(args.submission)
    data = load_data(args.dataset)
    if args.task_config and content_hash(data) != args.task_config["dataset_sha256"]:
        raise ValueError(f"dataset contents do not match task {args.task}; use its documented preparation settings")
    attached = set(circuit.split()) & set(OUTPUTS)
    warnings = []
    if not attached:
        warnings.append("Circuit connects to no output pins; tied outputs will count as incorrect.")
    missing = sorted(set(f"out{label}" for label in np.unique(data["test_y"])) - attached)
    if missing:
        warnings.append("No submitted devices connect to these dataset class outputs: " + ", ".join(missing))
    for warning in warnings:
        print("Warning: " + warning, file=sys.stderr)
    deck, order, train_end, stop = build_deck(circuit, data, args.epochs, args.seed, args.slot, args.startup, args.step, args.simulator)
    backend = ContainerBackend(args.docker_image) if args.docker_image else None
    if backend:
        version = backend.version()
    else:
        binary = shutil.which(args.binary or ("ngspice" if args.simulator == "ngspice" else "Xyce"))
        if not binary:
            raise ValueError("simulator not found; install ngspice/Xyce or supply --binary /path/to/executable")
        version = subprocess.run([binary, "--version" if args.simulator == "ngspice" else "-v"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if args.task_config and simulator_identity(args.simulator, version) != args.task_config["simulator_version"]:
        raise ValueError(f"task {args.task} requires {args.task_config['simulator_version']}; use that version or omit --task for a custom experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    # Keep the deck *before* starting SPICE, including on timeout or parse failure.
    (args.output / "harness.cir").write_text(deck)
    started = time.monotonic()
    stage = "simulation"
    print(f"Running {simulator_identity(args.simulator, version)}: {args.epochs * len(data['train_y'])} training + "
          f"{len(order)} test presentations, {stop:g} simulated seconds (timeout {args.timeout:g}s).", flush=True)
    # No dataset files in the simulation working directory. This is NOT a sandbox.
    with tempfile.TemporaryDirectory(prefix="circuit-bench-") as temporary:
        work = Path(temporary)
        (work / "harness.cir").write_text(deck)
        try:
            with (args.output / "simulator.log").open("w") as log:
                command = (backend.command(work) if backend else
                           [binary, "-n", "-b", "harness.cir"] if args.simulator == "ngspice" else [binary, "harness.cir"])
                result = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            log_text = (args.output / "simulator.log").read_text(errors="replace")
            if result.returncode:
                raise RuntimeError(f"simulator exited with status {result.returncode}")
            # ngspice's control-language `quit` can return 0 after tran aborts.
            if re.search(r"\baborted\b|\bfatal\b|^\s*error(?:\s*:| on line)", log_text, re.I | re.M):
                raise RuntimeError("simulator reported an error or aborted analysis despite a zero exit status")
            if not (work / "trace.txt").exists():
                raise RuntimeError("simulator produced no trace")
            stage = "scoring"
            report = score_chunks(trace_chunks(safe_trace(work / "trace.txt"), args.simulator),
                                  data["test_y"][order], train_end, stop, args.slot, args.startup)
            if args.keep_trace:
                stage = "saving trace"
                with (work / "trace.txt").open("rb") as source, gzip.open(args.output / "trace.txt.gz", "wb") as target:
                    shutil.copyfileobj(source, target)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as error:
            if backend:
                try:
                    backend.cancel()
                except (RuntimeError, OSError, subprocess.SubprocessError) as cleanup_error:
                    print(f"Warning: {cleanup_error}", file=sys.stderr)
            if (work / "trace.txt").exists():
                try:
                    shutil.copyfile(safe_trace(work / "trace.txt"), args.output / "trace.partial.txt")
                except (ValueError, OSError) as artifact_error:
                    print(f"Warning: partial trace not copied: {artifact_error}", file=sys.stderr)
            message = (f"simulator timed out after {args.timeout:g} seconds; use --timeout to increase the local limit"
                       if isinstance(error, subprocess.TimeoutExpired) else str(error) or "interrupted")
            failure = {"status": "failed", "stage": stage, "error": message, "protocol": PROTOCOL,
                       "container_name": backend.name if backend else None,
                       "docker_image_id": backend.image_id if backend else None,
                       "elapsed_s": time.monotonic() - started, "submission_sha256": circuit_hash,
                       "dataset_sha256": content_hash(data), "harness_sha256": hashlib.sha256(deck.encode()).hexdigest()}
            (args.output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise RuntimeError(f"{message}; debugging files retained in {args.output} (see simulator.log)") from error
    if report["invalid_predictions"]:
        warnings.append(f"{report['invalid_predictions']} test predictions were tied/invalid and counted as incorrect.")
        print("Warning: " + warnings[-1], file=sys.stderr)
    report.update({"protocol": PROTOCOL, "simulator": simulator_identity(args.simulator, version),
                   "task": args.task, "task_sha256": args.task_sha256,
                   "docker_image_id": backend.image_id if backend else None,
                   "simulator_version_output": version, "components": counts, "warnings": warnings,
                   "submission_sha256": circuit_hash, "dataset_sha256": content_hash(data),
                   "dataset_hash_format": "spicenn2-dataset-content-v1",
                   "dataset_archive_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                   "harness_sha256": hashlib.sha256(deck.encode()).hexdigest(),
                   "settings": {key: getattr(args, key) for key in ("simulator", "epochs", "seed", "slot", "startup", "step")},
                   "training_presentations": args.epochs * len(data["train_y"]), "elapsed_s": time.monotonic() - started,
                   "inference_latency_s": args.slot, "status": "local_result"})
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Accuracy: {report['accuracy']:.2%}; inference: {report['energy']['inference']['delivered_j_per_image']:.6g} J/image")
    print(args.output / "report.json")
    return report


def apply_task(args):
    args.task_config, args.task_sha256 = None, None
    expected = {}
    if args.task:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.task):
            raise ValueError("invalid task name")
        path = TASK_DIRECTORY / (args.task + ".json")
        if not path.is_file():
            raise ValueError("unknown task; available: " + ", ".join(sorted(p.stem for p in TASK_DIRECTORY.glob("*.json"))))
        raw = path.read_bytes()
        task = json.loads(raw)
        if (not isinstance(task, dict) or not {"id", "protocol", "dataset_sha256", "simulator_version", "settings"} <= task.keys()
                or not isinstance(task["settings"], dict) or set(task["settings"]) != set(DEFAULT_SETTINGS)
                or not isinstance(task["dataset_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", task["dataset_sha256"])):
            raise ValueError("invalid task manifest")
        if task["id"] != args.task or task["protocol"] != PROTOCOL:
            raise ValueError("task configuration does not match this runner protocol")
        args.task_config = task
        args.task_sha256 = hashlib.sha256(raw).hexdigest()
        expected = task["settings"]
    for key, default in DEFAULT_SETTINGS.items():
        value = getattr(args, key)
        if key in expected and value is not None and value != expected[key]:
            raise ValueError(f"task {args.task} fixes --{key}={expected[key]}; omit --task for a custom experiment")
        setattr(args, key, expected.get(key, default) if value is None else value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new directory for report, harness and log")
    parser.add_argument("--keep-trace", action="store_true", help="also retain the successful trace as trace.txt.gz")
    parser.add_argument("--task", help="named configuration, e.g. mnist017-v0; fixes settings and checks data/version")
    parser.add_argument("--epochs", type=int, help="training passes (custom-run default: 1)")
    parser.add_argument("--seed", type=int, help="shuffle seed (custom-run default: 0)")
    parser.add_argument("--slot", type=float, help="seconds per example (custom-run default: 0.001)")
    parser.add_argument("--startup", type=float, help="startup seconds (custom-run default: 0.001)")
    parser.add_argument("--step", type=float, help="maximum timestep (custom-run default: 1e-5)")
    parser.add_argument("--timeout", type=float, default=1800, help="simulator wall-time limit in seconds (default: 1800)")
    parser.add_argument("--simulator", choices=("ngspice", "xyce"), help="custom-run default: ngspice")
    parser.add_argument("--binary", help="path to simulator; defaults to ngspice or Xyce on PATH")
    parser.add_argument("--docker-image", help="run ngspice in this prebuilt local image with network/filesystem/resource restrictions")
    args = parser.parse_args(argv)
    try:
        apply_task(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.epochs < 0 or args.seed < 0 or not all(math.isfinite(v) and v > 0 for v in (args.slot, args.startup, args.step, args.timeout)):
        parser.error("epochs/seed must be nonnegative and timing values positive and finite")
    if args.step > args.slot / 100 or args.startup <= args.slot / 50:
        parser.error("step must be <= slot/100 and startup > slot/50")
    if args.docker_image and (args.binary or args.simulator != "ngspice"):
        parser.error("--docker-image uses the image's ngspice; do not combine it with --binary or Xyce")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
