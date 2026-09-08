"""The historical-family port is data-independent and remains a flat circuit."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from competition.runner import validate_submission
from competition import leaderboard


DIRECTORY = Path(__file__).resolve().parents[1] / "submissions/random-feature-peraz"
spec = importlib.util.spec_from_file_location("peraz_generator", DIRECTORY / "generate.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class PerazSubmissionTests(unittest.TestCase):
    def test_bundled_circuit_reproduces(self):
        self.assertEqual(generator.circuit(), (DIRECTORY / "circuit.cir").read_text())
        _, counts, _ = validate_submission(DIRECTORY / "circuit.cir")
        self.assertEqual(sum(counts.values()), 6915)

    def test_ten_digit_variant_is_in_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "circuit.cir"
            path.write_text(generator.circuit(labels=tuple(range(10))))
            _, counts, _ = validate_submission(path)
            self.assertLess(sum(counts.values()), 20000)

    def test_control_changes_only_learning_enable_wiring(self):
        trained = generator.circuit().splitlines()
        frozen = generator.circuit(freeze=True).splitlines()
        changes = [(a, b) for a, b in zip(trained, frozen) if a != b]
        self.assertEqual(len(changes), 3*(97+1))
        for a, b in changes:
            self.assertTrue(a.startswith(("Menable", "Maztrack")))
            self.assertEqual(a.replace(" learn ", " 0 "), b)

    def test_short_measurements_match_submitted_circuit(self):
        digest = hashlib.sha256(generator.circuit().encode()).hexdigest()
        for filename, epochs, correct in (("two-epoch-report.json", 2, 124),
                                         ("untrained-report.json", 0, 43)):
            report = json.loads((DIRECTORY / filename).read_text())
            self.assertEqual(report["submission_sha256"], digest)
            self.assertEqual(report["settings"]["epochs"], epochs)
            self.assertEqual(report["correct"], correct)
            self.assertEqual(report["accuracy"], correct/150)

    def test_reference_identity_and_finer_step_failure_cannot_rank(self):
        report = json.loads((DIRECTORY / "report.json").read_text())
        failure = json.loads((DIRECTORY / "half-step-failure.json").read_text())
        task = leaderboard.HERE / "tasks/mnist017-v0.json"
        release = leaderboard.read_json(leaderboard.HERE / "release.json")
        leaderboard.validate_report(report, DIRECTORY / "circuit.cir", task, release)
        self.assertEqual(failure["submission_sha256"], report["submission_sha256"])
        self.assertIn("timed out after 1800 seconds", failure["error"])
        with self.assertRaisesRegex(ValueError, "not a successful local result"):
            leaderboard.validate_report(failure, DIRECTORY / "circuit.cir", task, release, half_step=True)


if __name__ == "__main__":
    unittest.main()
