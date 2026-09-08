"""Portable PC-submission regressions; no simulator or downloaded data required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from competition.runner import validate_submission


ROOT = Path(__file__).resolve().parents[1]
TRAINED = ROOT / "submissions/predictive-coding"
FROZEN = ROOT / "submissions/predictive-coding-frozen"
SPEC = importlib.util.spec_from_file_location("pc_submission_generator", TRAINED / "generate.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def read_json(folder, filename):
    return json.loads((folder / filename).read_text())


def rows(text):
    return [line.split() for line in text.splitlines()
            if line.strip() and not line.startswith("*")]


class PredictiveCodingSubmissionTests(unittest.TestCase):
    def test_cli_writes_new_files_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            for folder in (TRAINED, FROZEN):
                with self.subTest(folder=folder.name):
                    output = Path(temporary) / (folder.name+".cir")
                    command = [sys.executable, str(folder / "generate.py"), "--output", str(output)]
                    first = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(first.returncode, 0, first.stderr)
                    self.assertEqual(output.read_text(), (folder / "circuit.cir").read_text())
                    output.write_text("entrant's edited circuit\n")
                    second = subprocess.run(command, capture_output=True, text=True)
                    self.assertNotEqual(second.returncode, 0)
                    self.assertIn("already exists", second.stderr)
                    self.assertEqual(output.read_text(), "entrant's edited circuit\n")

    def test_defaults_regenerate_without_reading_data_or_weights(self):
        for folder, frozen in ((TRAINED, False), (FROZEN, True)):
            with self.subTest(folder=folder.name):
                expected = (folder / "circuit.cir").read_text()
                with patch("builtins.open", side_effect=AssertionError("unexpected file read")), \
                        patch.object(Path, "open", side_effect=AssertionError("unexpected path read")):
                    generated = GENERATOR.generate(frozen=frozen)
                    repeated = GENERATOR.generate(frozen=frozen)
                self.assertEqual(generated, expected)
                self.assertEqual(repeated, expected)

    def test_validated_counts_and_report_hashes(self):
        expected = {TRAINED: {"r": 1164, "c": 188, "m": 1981, "d": 0},
                    FROZEN: {"r": 788, "c": 92, "m": 925, "d": 0}}
        for folder, counts in expected.items():
            with self.subTest(folder=folder.name):
                _, actual, digest = validate_submission(folder / "circuit.cir")
                self.assertEqual(actual, counts)
                regenerated = GENERATOR.generate(frozen=folder == FROZEN).encode()
                self.assertEqual(hashlib.sha256(regenerated).hexdigest(), digest)
                for filename in ("report.json", "report-half-step.json", "untrained-report.json"):
                    report = read_json(folder, filename)
                    self.assertEqual(report["submission_sha256"], digest)
                    self.assertEqual(report["components"], counts)
                    self.assertEqual(len(report["predictions"]), report["test_images"])
                    self.assertEqual(report["accuracy"], report["correct"] / report["test_images"])

    def test_hidden_and_output_weights_use_capacitors_as_declared(self):
        active = rows(GENERATOR.generate())
        frozen = rows(GENERATOR.generate(frozen=True))
        for net, hidden_count in ((active, 64), (frozen, 0)):
            hidden_caps = [r for r in net if r[0].startswith("C") and r[1].startswith(("n_wp_h", "n_wn_h"))]
            output_caps = [r for r in net if r[0].startswith("C") and r[1].startswith(("n_wp_o", "n_wn_o"))]
            self.assertEqual(len(hidden_caps), hidden_count)
            self.assertEqual(len(output_caps), 48)
            for cap in hidden_caps + output_caps:
                # Every learned rail has an electrical reset path, a forward gate
                # connection, and a distinct physical update current connection.
                node = cap[1]
                self.assertTrue(any(r[0].startswith("M") and r[1] == node and r[2] == "reset" for r in net))
                self.assertTrue(any(r[0].startswith("M") and r[2] == node for r in net))
                self.assertTrue(any(r[0].startswith("M") and r[1] == node and r[2] != "reset" for r in net))

    def test_starter_pins_use_digit_seven_not_compact_class_two(self):
        for frozen in (False, True):
            net = rows(GENERATOR.generate(frozen=frozen))
            targets = {node for r in net for node in r[1:3] if node.startswith("target")}
            self.assertEqual(targets, {"target0", "target1", "target7"})
            grounded = {r[1] for r in net if r[0].startswith("R") and r[1].startswith("out") and r[2] == "0"}
            self.assertEqual(grounded, {"out2", "out3", "out4", "out5", "out6", "out8", "out9"})

    def test_measured_hidden_updates_are_label_sensitive_and_retained(self):
        trained = read_json(TRAINED, "weight-diagnostic.json")
        frozen = read_json(FROZEN, "weight-diagnostic.json")
        permuted = read_json(TRAINED, "permuted-weight-diagnostic.json")
        self.assertEqual(trained["nodes"], frozen["nodes"])
        self.assertEqual(trained["nodes"], permuted["nodes"])
        self.assertEqual(trained["startup_volts"], permuted["startup_volts"])
        self.assertTrue(permuted["permuted_labels"])
        self.assertLess(max(map(abs, frozen["training_change_volts"][:4])), 5e-7)
        self.assertLess(max(map(abs, trained["inference_change_volts"][:4])), 1e-6)
        for index in (0, 2):
            movement = trained["training_change_volts"][index] - trained["training_change_volts"][index+1]
            changed_labels = permuted["training_change_volts"][index] - permuted["training_change_volts"][index+1]
            self.assertGreater(abs(movement), 1e-5)
            self.assertGreater(abs(changed_labels-movement), 5e-6)

    def test_frozen_paired_measurements_agree(self):
        self.check_paired_measurements(FROZEN)

    def test_trained_paired_measurements_agree(self):
        self.check_paired_measurements(TRAINED)

    def check_paired_measurements(self, folder):
        coarse = read_json(folder, "report.json")
        fine = read_json(folder, "report-half-step.json")
        summary = read_json(folder, "numerical-check.json")
        self.assertEqual(coarse["predictions"], fine["predictions"])
        self.assertEqual(coarse["submission_sha256"], fine["submission_sha256"])
        self.assertEqual(summary["coarse_submission_sha256"], coarse["submission_sha256"])
        self.assertEqual(summary["fine_submission_sha256"], fine["submission_sha256"])
        self.assertEqual(coarse["dataset_sha256"], fine["dataset_sha256"])
        self.assertEqual(fine["settings"]["step"], coarse["settings"]["step"] / 2)
        self.assertTrue(summary["prediction_lists_identical"])
        for phase in ("startup", "training", "inference", "total"):
            energy = coarse["energy"][phase]["delivered_j"]
            relative = abs(fine["energy"][phase]["delivered_j"]-energy) / energy
            self.assertLessEqual(relative, 0.01)
            self.assertAlmostEqual(relative, summary["phase_relative_energy_difference"][phase], places=14)


if __name__ == "__main__":
    unittest.main()
