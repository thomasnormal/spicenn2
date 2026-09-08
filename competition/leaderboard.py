#!/usr/bin/env python3
"""Record organizer reruns and render separate accuracy/energy Pareto tables.

Metadata validation is not proof that a simulation was run. Only the organizer
should record results after running the trusted runner on the submitted netlist.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import stat

import numpy as np

try:
    from .data import content_hash, load_data
    from .runner import PROTOCOL, SOURCES, build_deck, validate_submission
except ImportError:
    from data import content_hash, load_data
    from runner import PROTOCOL, SOURCES, build_deck, validate_submission


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "competition"


def read_json(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "JSON artifact must be a regular file, not a link")
    require(info.st_size <= 1024 * 1024, "JSON artifact exceeds 1 MiB")
    with path.open("rb") as source:
        value = source.read(1024 * 1024 + 1)
    require(len(value) <= 1024 * 1024, "JSON artifact exceeds 1 MiB")
    return json.loads(value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_report(report, circuit_path, task_path, release, half_step=False, data=None):
    task = read_json(task_path)
    circuit, counts, digest = validate_submission(circuit_path)
    expected_settings = dict(task["settings"])
    if half_step:
        expected_settings["step"] = release["verification"]["half_step"]
    require(report["status"] == "local_result", "report is not a successful local result")
    require(report["protocol"] == task["protocol"] == release["protocol"] == PROTOCOL, "protocol mismatch")
    require(report["settings"] == expected_settings, "runner settings do not match the task")
    require(report["dataset_sha256"] == task["dataset_sha256"], "dataset mismatch")
    require(report["dataset_hash_format"] == "spicenn2-dataset-content-v1", "unknown dataset hash format")
    require(report["simulator"] == task["simulator_version"], "simulator version mismatch")
    require(report["docker_image_id"] == release["reference_image"]["id"], "result did not use the reference image")
    require(report["submission_sha256"] == digest and report["components"] == counts, "circuit identity/count mismatch")
    if not half_step:
        require(report["task"] == task["id"], "task identifier mismatch")
        require(report["task_sha256"] == hashlib.sha256(task_path.read_bytes()).hexdigest(), "task manifest hash mismatch")
    prep = task["preparation"]
    ntest = len(prep["labels"]) * prep["test_per_class"]
    ntrain = len(prep["labels"]) * prep["train_per_class"] * task["settings"]["epochs"]
    require(report["test_images"] == ntest and report["training_presentations"] == ntrain, "dataset/presentation count mismatch")
    require(report["inference_latency_s"] == task["settings"]["slot"], "latency mismatch")
    predictions = np.asarray(report["predictions"])
    confusion = np.asarray(report["confusion_matrix"])
    require(predictions.shape == (ntest,) and np.issubdtype(predictions.dtype, np.integer)
            and np.all((predictions >= -1) & (predictions < 10)), "invalid prediction list")
    require(confusion.shape == (10, 11) and np.issubdtype(confusion.dtype, np.integer)
            and np.all(confusion >= 0), "invalid confusion matrix")
    rows = np.zeros(10, dtype=int)
    rows[prep["labels"]] = prep["test_per_class"]
    require(np.array_equal(confusion.sum(axis=1), rows), "confusion rows do not match the task classes")
    require(np.array_equal(confusion.sum(axis=0), np.bincount(np.where(predictions < 0, 10, predictions), minlength=11)),
            "predictions and confusion columns disagree")
    correct = int(np.trace(confusion[:, :10]))
    require(report["correct"] == correct and report["accuracy"] == correct / ntest, "accuracy/count mismatch")
    require(report["invalid_predictions"] == int(np.count_nonzero(predictions < 0)), "invalid-prediction count mismatch")
    for phase in ("startup", "training", "inference", "total"):
        energy = report["energy"][phase]
        require(set(energy["by_source"]) == set(SOURCES), "all driven pins must be metered")
        for key in ("delivered_j", "returned_j", "net_j"):
            value = energy[key]
            values = [energy["by_source"][name][key] for name in SOURCES]
            require(math.isfinite(value) and all(math.isfinite(v) for v in values), "non-finite energy")
            if key != "net_j":
                require(value >= -1e-15 and min(values) >= -1e-15, "negative gross/returned energy")
            require(math.isclose(value, sum(values), rel_tol=1e-9, abs_tol=1e-15), "source energy sum mismatch")
        require(math.isclose(energy["delivered_j"] - energy["returned_j"], energy["net_j"], rel_tol=1e-9, abs_tol=1e-15),
                "net energy identity mismatch")
    energy = report["energy"]
    for key in ("delivered_j", "returned_j", "net_j"):
        require(math.isclose(energy["total"][key], sum(energy[p][key] for p in ("startup", "training", "inference")),
                             rel_tol=1e-9, abs_tol=1e-15), "phase energy sum mismatch")
    require(math.isclose(energy["inference"]["delivered_j_per_image"], energy["inference"]["delivered_j"] / ntest,
                         rel_tol=1e-12, abs_tol=1e-18), "inference energy/image mismatch")
    if data is not None:
        require(content_hash(data) == task["dataset_sha256"], "provided dataset does not match the task")
        settings = dict(expected_settings)
        deck, order, _, _ = build_deck(circuit, data, **settings)
        require(hashlib.sha256(deck.encode()).hexdigest() == report["harness_sha256"], "harness cannot be reproduced")
        expected_confusion = np.zeros((10, 11), dtype=int)
        for label, prediction in zip(data["test_y"][order], predictions):
            expected_confusion[label, prediction if prediction >= 0 else 10] += 1
        require(np.array_equal(confusion, expected_confusion), "predictions disagree with the actual evaluation labels")


def check_stability(report, verification, release):
    require(report["predictions"] == verification["predictions"], "predictions changed at half timestep")
    tolerance = release["verification"]["maximum_energy_relative_difference"]
    for phase in ("startup", "training", "inference", "total"):
        a, b = (r["energy"][phase]["delivered_j"] for r in (report, verification))
        require(abs(a-b) <= tolerance * max(abs(a), abs(b), 1e-30), f"{phase} energy changed by more than {tolerance:.0%}")


def frontier(reports):
    """Exact reported values determine dominance; no mixed scalar score."""
    points = [(r["accuracy"], r["energy"]["inference"]["delivered_j_per_image"]) for r in reports]
    return [not any(a >= accuracy and e <= energy and (a > accuracy or e < energy)
                    for a, e in points) for accuracy, energy in points]


def entries():
    return sorted((HERE / "results").glob("*/*/entry.json"))


def validate_entry(path, release):
    entry = read_json(path)
    require(entry["task"] in release["tasks"], "entry uses an unreleased task")
    require(path.parent.parent.name == entry["task"] and path.parent.name == entry["id"], "entry path/id mismatch")
    circuit = (ROOT / entry["circuit"]).resolve()
    require(ROOT in circuit.parents, "circuit must be in the repository")
    task_path = HERE / "tasks" / (entry["task"] + ".json")
    report, verification = read_json(path.parent / "report.json"), read_json(path.parent / "verification.json")
    validate_report(report, circuit, task_path, release)
    validate_report(verification, circuit, task_path, release, half_step=True)
    check_stability(report, verification, release)
    return entry, report


def render(release):
    records = [validate_entry(path, release) for path in entries()]
    lines = ["# Circuit-learning leaderboard", "", "Separate tracks; fixed 1 ms/image latency. See [v0 rules](RULES.md).",
             "A frontier entry is not beaten on both accuracy and inference energy by another entry.",
             "Startup and training energy are shown separately, not hidden in an amortized score.", "",
             "These are public-development-set results, not estimates on a secret held-out set.",
             "Only organizer reruns with matching reference-image/configuration metadata and a passing",
             "half-timestep check are recorded here. Metadata checks alone do not authenticate a run.", ""]
    for task in release["tasks"]:
        rows = [(entry, report) for entry, report in records if entry["task"] == task]
        rows.sort(key=lambda pair: (-pair[1]["accuracy"], pair[1]["energy"]["inference"]["delivered_j_per_image"], pair[0]["id"]))
        lines += [f"## {task}", ""]
        if not rows:
            lines += ["No verified results recorded yet.", ""]
            continue
        lines += ["| Circuit | Accuracy | Inference / image | Startup | Training | Frontier |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for (entry, report), on_frontier in zip(rows, frontier([r for _, r in rows])):
            energy = report["energy"]
            name = entry["name"].replace("|", "\\|").replace("[", "\\[").replace("]", "\\]")
            link = f"results/{task}/{entry['id']}/entry.json"
            lines.append(f"| [{name}]({link}) | {report['accuracy']:.2%} ({report['correct']}/{report['test_images']}) | "
                         f"{energy['inference']['delivered_j_per_image']*1e6:.6g} µJ | {energy['startup']['delivered_j']*1e3:.6g} mJ | "
                         f"{energy['training']['delivered_j']*1e3:.6g} mJ | {'Yes' if on_frontier else 'No'} |")
        lines.append("")
    lines += ["Regenerate with `python competition/leaderboard.py render`. CI checks metadata and table consistency;",
              "it does not promote self-reported submissions to verified results.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("render")
    sub.add_parser("check")
    record = sub.add_parser("record", help="organizer-only: record completed reference and half-step reruns")
    for name in ("id", "name", "task", "circuit", "report", "verification", "dataset"):
        record.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        release = read_json(HERE / "release.json")
        if args.command == "record":
            require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.id), "invalid entry ID")
            require(args.task in release["tasks"], "unknown released task")
            require(args.name.strip() and not any(ord(c) < 32 for c in args.name), "entry name must be nonempty and single-line")
            circuit = Path(args.circuit).resolve()
            relative = circuit.relative_to(ROOT)
            task = HERE / "tasks" / (args.task + ".json")
            report, verification = read_json(Path(args.report)), read_json(Path(args.verification))
            data = load_data(Path(args.dataset))
            validate_report(report, circuit, task, release, data=data)
            validate_report(verification, circuit, task, release, half_step=True, data=data)
            check_stability(report, verification, release)
            path = HERE / "results" / args.task / args.id
            path.mkdir(parents=True, exist_ok=False)
            entry = {"id": args.id, "name": args.name, "task": args.task, "circuit": relative.as_posix(),
                     "verification": "organizer reference-image rerun and half-step stability check"}
            for name, value in (("entry.json", entry), ("report.json", report), ("verification.json", verification)):
                (path / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        rendered = render(release)
        target = HERE / "LEADERBOARD.md"
        if args.command == "check":
            require(target.read_text() == rendered, "leaderboard is stale; run leaderboard.py render")
            print(f"Validated {len(entries())} recorded results and leaderboard formatting")
        else:
            target.write_text(rendered)
            print(target)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
