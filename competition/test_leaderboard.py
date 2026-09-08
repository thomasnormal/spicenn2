"""Metadata and numerical verification cannot be bypassed by favorable scores."""
import copy
from pathlib import Path
import tempfile
import unittest

from competition import leaderboard as lb


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


if __name__ == "__main__":
    unittest.main()
