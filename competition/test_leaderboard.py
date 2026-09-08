"""Metadata and numerical verification cannot be bypassed by favorable scores."""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from competition import leaderboard as lb
from competition.runner import score_trace


class LeaderboardTests(unittest.TestCase):
    def setUp(self):
        self.release = lb.read_json(lb.HERE / "release.json")
        folder = lb.HERE / "results/mnist017-v0/delta-rule-16uf"
        self.report = lb.read_json(folder / "report.json")
        self.verification = lb.read_json(folder / "verification.json")
        self.circuit = lb.HERE / "examples/mnist_017_tuned.cir"
        self.task = lb.HERE / "tasks/mnist017-v0.json"

    def validate(self, report):
        lb.validate_report(report, self.circuit, self.task, self.release)

    def test_recorded_baseline_and_generated_table(self):
        self.validate(self.report)
        lb.validate_report(self.verification, self.circuit, self.task, self.release, half_step=True)
        lb.check_stability(self.report, self.verification, self.release)
        self.assertEqual((lb.HERE / "LEADERBOARD.md").read_text(), lb.render(self.release))
        text = lb.render(self.release)
        self.assertIn("delta-rule-16uf/report.json)", text)
        self.assertIn("delta-rule-16uf/verification.json)", text)
        self.assertIn("../competition/examples/mnist_017_tuned.cir)", text)

    def test_reject_wrong_identity_settings_or_score(self):
        changes = {
            "protocol": "different", "dataset_sha256": "0" * 64,
            "submission_sha256": "0" * 64, "docker_image_id": "sha256:" + "0" * 64,
            "task_sha256": "0" * 64, "simulator": "ngspice-45",
            "correct": 150, "accuracy": 1.0, "test_images": 149,
            "invalid_predictions": 1, "status": "failed",
            "settings": dict(self.report["settings"], epochs=1),
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                report = copy.deepcopy(self.report)
                report[key] = value
                with self.assertRaises(ValueError):
                    self.validate(report)

    def test_reject_bad_predictions_and_energy(self):
        report = copy.deepcopy(self.report)
        report["predictions"][0] = 10
        with self.assertRaisesRegex(ValueError, "prediction"):
            self.validate(report)
        for value in (float("nan"), float("inf"), -1, 999):
            report = copy.deepcopy(self.report)
            report["energy"]["inference"]["delivered_j"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(report)

    def test_half_step_checks_every_prediction_and_phase(self):
        report = copy.deepcopy(self.verification)
        # Even equal accuracy is insufficient if individual predictions differ.
        report["predictions"][0] = (report["predictions"][0] + 1) % 10
        with self.assertRaisesRegex(ValueError, "predictions changed"):
            lb.check_stability(self.report, report, self.release)
        for phase in ("startup", "training", "inference", "total"):
            report = copy.deepcopy(self.verification)
            report["energy"][phase]["delivered_j"] *= 1.02
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, phase):
                lb.check_stability(self.report, report, self.release)

    def test_frontier_is_not_just_accuracy_sort(self):
        def point(accuracy, energy):
            return {"accuracy": accuracy, "energy": {"inference": {"delivered_j_per_image": energy}}}
        reports = [point(.9, 10), point(.8, 5), point(.8, 10), point(.9, 10), point(.7, 12)]
        self.assertEqual(lb.frontier(reports), [True, True, False, True, False])

    def test_json_links_and_oversize_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text("{}")
            link = root / "link.json"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "not a link"):
                lb.read_json(link)
            source.write_bytes(b" " * (1024 * 1024 + 1))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                lb.read_json(source)


class VerifyCliTests(unittest.TestCase):
    """Synthetic reports test consistency checks, not proof of physical simulation."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.here = self.root / "competition"
        (self.here / "tasks").mkdir(parents=True)
        self.circuit = self.root / "circuit.cir"
        self.circuit.write_text("Rtest vdd out0 10k\n")
        circuit, counts, digest = lb.validate_submission(self.circuit)
        arrays = dict(train_x=np.zeros((2, 16)), train_y=np.array([0, 1]),
                      test_x=np.zeros((2, 16)), test_y=np.array([0, 1]))
        self.dataset = self.root / "data.npz"
        np.savez(self.dataset, **arrays)
        release = lb.read_json(lb.HERE / "release.json")
        release["tasks"] = ["mini-v0"]
        (self.here / "release.json").write_text(json.dumps(release))
        settings = dict(simulator="ngspice", epochs=1, seed=0, slot=.001, startup=.02, step=1e-5)
        task = dict(id="mini-v0", protocol=lb.PROTOCOL, simulator_version="ngspice-46",
                    dataset_sha256=lb.content_hash(arrays), settings=settings,
                    preparation=dict(labels=[0, 1], train_per_class=1, test_per_class=1))
        task_path = self.here / "tasks/mini-v0.json"
        task_path.write_text(json.dumps(task))
        for name, step in (("report", 1e-5), ("verification", 5e-6)):
            actual = dict(settings, step=step)
            deck, order, train_end, stop = lb.build_deck(circuit, arrays, **actual)
            trace = np.zeros((3, 11 + len(lb.SOURCES)))
            trace[:, 0] = [0, .02, stop]
            trace[:, 1] = 1
            report = score_trace(trace, arrays["test_y"][order], train_end, stop, .001, .02)
            report.update(status="local_result", protocol=lb.PROTOCOL, settings=actual,
                          dataset_sha256=lb.content_hash(arrays), dataset_hash_format="spicenn2-dataset-content-v1",
                          simulator="ngspice-46", docker_image_id=release["reference_image"]["id"],
                          submission_sha256=digest, components=counts, task="mini-v0",
                          task_sha256=hashlib.sha256(task_path.read_bytes()).hexdigest(),
                          harness_sha256=hashlib.sha256(deck.encode()).hexdigest(),
                          training_presentations=2, inference_latency_s=.001)
            (self.root / (name + ".json")).write_text(json.dumps(report))
        self.args = ["verify", "--task", "mini-v0", "--circuit", str(self.circuit),
                     "--dataset", str(self.dataset), "--report", str(self.root / "report.json"),
                     "--verification", str(self.root / "verification.json")]

    def invoke(self, args):
        output = io.StringIO()
        with patch.object(lb, "HERE", self.here), patch.object(lb, "ROOT", self.root), \
                contextlib.redirect_stdout(output):
            lb.main(args)
        return output.getvalue()

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def test_verify_and_comparison_are_read_only(self):
        before = self.snapshot()
        output = self.invoke(self.args + ["--against", str(self.root / "report.json")])
        self.assertIn("PASS: published result comparison", output)
        self.assertIn("half-step predictions match", output)
        self.assertIn("does not authenticate", output)
        self.assertEqual(before, self.snapshot())

    def test_invalid_reports_fail_without_writing(self):
        for field, value in (("harness_sha256", "0" * 64), ("docker_image_id", None),
                             ("predictions", [1, 1])):
            path = self.root / "report.json"
            original = path.read_text()
            report = json.loads(original)
            report[field] = value
            path.write_text(json.dumps(report))
            before = self.snapshot()
            with self.subTest(field=field), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.invoke(self.args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(before, self.snapshot())
            path.write_text(original)

    def test_wrong_dataset_fails_without_writing(self):
        arrays = lb.load_data(self.dataset)
        arrays["test_x"][0, 0] = .5
        np.savez(self.dataset, **arrays)
        before = self.snapshot()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.invoke(self.args)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
