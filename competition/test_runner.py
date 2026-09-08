import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from competition.energy import integrate_power
from competition.runner import build_deck, validate_submission, score_trace, SOURCES
from competition.prepare_mnist import preprocess, balanced
from competition.make_baseline import circuit
from competition.prepare_toy import dataset

ROOT = Path(__file__).resolve().parent


class EnergyTests(unittest.TestCase):
    def test_zero_crossing_and_nonuniform_window(self):
        energy = integrate_power([0, 2], [[-2], [2]], 0.5, 1.5)
        self.assertAlmostEqual(energy["delivered_j"][0], 0.25)
        self.assertAlmostEqual(energy["returned_j"][0], 0.25)
        self.assertAlmostEqual(energy["net_j"][0], 0)

    def test_sources_cannot_cancel(self):
        energy = integrate_power([0, 1], [[2, -2], [2, -2]], 0, 1)
        self.assertAlmostEqual(energy["delivered_j"].sum(), 2)
        self.assertAlmostEqual(energy["net_j"].sum(), 0)

    def test_reject_truncated_and_nonfinite(self):
        for time, power, stop in [([0, 1], [[1], [1]], 2), ([0, 0], [[1], [1]], 1),
                                  ([0, 1], [[1], [np.nan]], 1)]:
            with self.assertRaises(ValueError):
                integrate_power(time, power, 0, stop)


class ProtocolTests(unittest.TestCase):
    def test_trace_endpoint_rounding_is_not_truncation(self):
        trace = np.zeros((3, 11 + len(SOURCES)))
        trace[:, 0] = [0, 0.02, 0.12]
        trace[:, 1] = 1
        result = score_trace(trace, np.array([0]), 0.02, 0.12000000000000001, 0.1, 0.02)
        self.assertEqual(result["accuracy"], 1)
        with self.assertRaises(ValueError):
            score_trace(trace, np.array([0]), 0.02, 0.13, 0.1, 0.02)

    def test_checked_in_baselines_match_generator(self):
        self.assertEqual((ROOT / "examples/blobs.cir").read_text(), circuit(2, [0, 1]))
        self.assertEqual((ROOT / "examples/mnist_017.cir").read_text(), circuit(16, [0, 1, 7]))
        for name in ("blobs.cir", "mnist_017.cir"):
            validate_submission(ROOT / "examples" / name)

    def test_preprocessing_and_balanced_selection(self):
        images = np.zeros((20, 28, 28), dtype=np.uint8)
        images[:, :7, :7] = 255
        x = preprocess(images)
        np.testing.assert_array_equal(x[:, 0], np.ones(20))
        np.testing.assert_array_equal(x[:, 1:], np.zeros((20, 15)))
        selected, labels = balanced(images, np.repeat(np.arange(10), 2), 1, np.random.default_rng(0))
        self.assertEqual(selected.shape, (10, 16))
        np.testing.assert_array_equal(np.sort(labels), np.arange(10))

    def test_reject_active_sources_and_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.cir"
            for line in [".control", ".include secret", "Vfree out0 0 3", "Bfree out0 0 V=1",
                         "Rbad out0 0 -1", "Cbad out0 0 1p ic=3", "Rbad out0 0 {x}",
                         "Rbad vdd global_node 1k", "Mbad out0 in0 0 0 custom w=1u l=1u"]:
                path.write_text(line)
                with self.assertRaises(ValueError, msg=line):
                    validate_submission(path)

    def test_test_labels_do_not_affect_deck(self):
        data = dict(train_x=np.zeros((1, 16)), train_y=np.array([0]),
                    test_x=np.ones((2, 16)), test_y=np.array([0, 1]))
        first = build_deck("Rtest out0 0 10k", data, 1, 0, 1e-3, 1e-3, 1e-5)[0]
        data["test_y"][:] = 9
        second = build_deck("Rtest out0 0 10k", data, 1, 0, 1e-3, 1e-3, 1e-5)[0]
        self.assertEqual(first, second)


class SimulatorTests(unittest.TestCase):
    def test_baseline_learns_from_training_labels(self):
        if not shutil.which("ngspice"):
            self.skipTest("ngspice unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            data = dataset()
            # Reverse both label sets: verifies that the circuit learns the labels
            # presented by the harness rather than hard-coding a class mapping.
            data["train_y"] = 1 - data["train_y"]
            data["test_y"] = 1 - data["test_y"]
            np.savez(work / "data.npz", **data)
            for epochs, output in ((0, "before"), (10, "after")):
                subprocess.run([sys.executable, str(ROOT / "runner.py"), str(ROOT / "examples/blobs.cir"),
                                str(work / "data.npz"), "--epochs", str(epochs), "--startup", "0.02",
                                "--output", str(work / output)], check=True, capture_output=True, timeout=90)
            before = json.loads((work / "before/report.json").read_text())
            after = json.loads((work / "after/report.json").read_text())
            self.assertEqual(before["invalid_predictions"], 100)
            self.assertGreaterEqual(after["accuracy"], 0.95)
            self.assertGreater(after["energy"]["training"]["delivered_j"], 0)

    def run_backend(self, simulator, binary, learning=False):
        if not binary or not shutil.which(binary):
            self.skipTest(f"{simulator} unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            np.savez(work / "data.npz", train_x=np.zeros((1, 16)), train_y=np.array([0 if learning else 7]),
                     test_x=np.zeros((2, 16)), test_y=np.array([0, 1]))
            circuit = ROOT / "examples/constant_zero.cir"
            if learning:
                circuit = work / "storage.cir"
                circuit.write_text("Mstore target0 learn out0 0 nch W=100u L=1u\nCstore out0 0 1u\n"
                                   + "\n".join(f"R{i} out{i} 0 10k" for i in range(1, 10)))
            subprocess.run([sys.executable, str(ROOT / "runner.py"),
                            str(circuit), str(work / "data.npz"),
                            "--output", str(work / "result"), "--simulator", simulator, "--binary", binary],
                           check=True, capture_output=True, text=True, timeout=30)
            result = json.loads((work / "result/report.json").read_text())
            self.assertEqual(result["accuracy"], 0.5)
            self.assertEqual(result["invalid_predictions"], 0)
            if learning:
                # A real MOS charges 1 uF from the target pin during training.
                # The capacitor must retain its score after targets go to zero.
                delivered = result["energy"]["training"]["by_source"]["target0"]["delivered_j"]
                self.assertAlmostEqual(delivered, 1e-6, delta=1e-7)
                return result
            # 3 V through 10k + (10k || 1G): ~0.45 mW, 0.45 uJ per 1 ms image.
            self.assertAlmostEqual(result["energy"]["inference"]["delivered_j_per_image"], 4.5e-7, delta=1e-10)
            self.assertGreater(result["energy"]["startup"]["delivered_j"], 0)
            return result

    def test_ngspice(self):
        self.run_backend("ngspice", shutil.which("ngspice"))

    def test_xyce(self):
        self.run_backend("xyce", os.environ.get("XYCE_BINARY", shutil.which("Xyce")))

    def test_ngspice_training_retention_and_target_energy(self):
        self.run_backend("ngspice", shutil.which("ngspice"), learning=True)

    def test_xyce_training_retention_and_target_energy(self):
        self.run_backend("xyce", os.environ.get("XYCE_BINARY", shutil.which("Xyce")), learning=True)


if __name__ == "__main__":
    unittest.main()
