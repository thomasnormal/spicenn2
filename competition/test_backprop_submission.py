"""Architectural regressions for the physical CMSUB submission (no simulator)."""
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
FOLDER = ROOT / "submissions/backprop-cmsub"
SPEC = importlib.util.spec_from_file_location("backprop_submission", FOLDER / "generate.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def devices(text):
    return {row.split()[0]: row.split() for row in text.splitlines()
            if row.strip() and not row.startswith("*")}


class BackpropSubmissionTests(unittest.TestCase):
    def test_generator_refuses_overwrite_without_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "edited.cir"
            output.write_text("* entrant edits\n")
            result = subprocess.run([sys.executable, str(FOLDER / "generate.py"), str(output)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(output.read_text(), "* entrant edits\n")

    def test_reproducible_without_reading_data(self):
        expected = (FOLDER / "circuit.cir").read_text()
        # Once imported, generation must not read training samples or weight files.
        with patch("builtins.open", side_effect=AssertionError("unexpected file read")), \
                patch.object(Path, "open", side_effect=AssertionError("unexpected path read")):
            actual = GENERATOR.circuit()
        self.assertEqual(actual, expected)
        validate_submission(FOLDER / "circuit.cir")

    def test_backward_transport_shares_learned_forward_weights(self):
        net = devices(GENERATOR.circuit())
        for label in (0, 1, 7):
            for hidden in range(9):
                weight = net[f"Mposo{label}_{hidden}"][2]
                self.assertEqual(net[f"Mta{hidden}_{label}"][2], weight)
                self.assertEqual(net[f"Mtd{hidden}_{label}"][2], weight)
                self.assertEqual(net[f"Cwo{label}_{hidden}"][1], weight)
                self.assertEqual(net[f"Mu2o{label}_{hidden}"][1], weight)
                self.assertEqual(net[f"Mreseto{label}_{hidden}"][3], weight)
        weight_caps = [row for name, row in net.items() if name.startswith("Cw")]
        self.assertEqual(len(weight_caps), 75)
        # All 45 hidden weights have physical update current and reset paths.
        for name, row in net.items():
            if name.startswith("Cwh"):
                tag = name[2:]
                self.assertEqual(net["Mu2"+tag][1], row[1])
                self.assertEqual(net["Mreset"+tag][3], row[1])
                self.assertEqual(net["Menable"+tag][2], "learn")

    def test_controls_only_disable_requested_updates(self):
        active = devices(GENERATOR.circuit())
        for option, count in (("freeze_hidden", 45), ("freeze_all", 75)):
            with self.subTest(control=option):
                frozen = devices(GENERATOR.circuit(**{option: True}))
                self.assertEqual(active.keys(), frozen.keys())
                changed = [name for name in active if active[name] != frozen[name]]
                self.assertEqual(len(changed), count)
                for name in changed:
                    self.assertTrue(name.startswith("Menableh" if option == "freeze_hidden" else "Menable"))
                    expected = active[name].copy()
                    self.assertEqual(expected[2], "learn")
                    expected[2] = "0"
                    self.assertEqual(frozen[name], expected)

    def test_short_measurements_identify_reproducible_circuits(self):
        configurations = {"two-epochs": {}, "untrained": {"freeze_all": True},
                          "frozen-hidden-two-epochs": {"freeze_hidden": True}}
        for filename, options in configurations.items():
            with self.subTest(report=filename):
                report = json.loads((FOLDER / "results" / (filename+".json")).read_text())
                digest = hashlib.sha256(GENERATOR.circuit(**options).encode()).hexdigest()
                self.assertEqual(report["submission_sha256"], digest)
                self.assertEqual(len(report["predictions"]), report["test_images"])
                self.assertEqual(report["accuracy"], report["correct"] / report["test_images"])


if __name__ == "__main__":
    unittest.main()
