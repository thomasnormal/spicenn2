"""Runner regressions: failures, large traces, data identity and long transients."""
import gzip
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from competition.data import content_hash, load_data
from competition.energy import integrate_power
from competition.make_baseline import circuit
from competition.prepare_toy import dataset
from competition.runner import (DEFAULT_SETTINGS, SOURCES, apply_task, build_deck, compact_pwl, score_chunks,
                                score_trace, trace_chunks, validate_submission)

ROOT = Path(__file__).resolve().parent


class ReliabilityTests(unittest.TestCase):
    def test_named_tasks_fix_settings_without_silent_override(self):
        for task, epochs in (("blobs-dev", 10), ("mnist017-dev", 20), ("mnist10-dev", 20),
                             ("mnist017-v0", 20), ("mnist10-v0", 20)):
            args = argparse.Namespace(task=task, **dict.fromkeys(DEFAULT_SETTINGS))
            apply_task(args)
            self.assertEqual(args.epochs, epochs)
            self.assertEqual(args.startup, .02)
            self.assertEqual(args.step, 1e-5)
            self.assertEqual(len(args.task_sha256), 64)
        args = argparse.Namespace(task="mnist017-dev", **dict.fromkeys(DEFAULT_SETTINGS))
        args.epochs = 1
        with self.assertRaisesRegex(ValueError, "fixes --epochs=20"):
            apply_task(args)
        args = argparse.Namespace(task=None, **dict.fromkeys(DEFAULT_SETTINGS))
        apply_task(args)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.startup, .001)

    def test_tutorial_data_and_harness_identifiers_reproduce(self):
        arrays = dataset()
        task = json.loads((ROOT / "tasks/blobs-dev.json").read_text())
        self.assertEqual(content_hash(arrays), task["dataset_sha256"])
        circuit, counts, digest = validate_submission(ROOT / "examples/blobs.cir")
        for name in ("blobs-ngspice", "blobs-xyce", "blobs-untrained"):
            report = json.loads((ROOT.parent / "docs/results/tutorial-v0" / (name + ".json")).read_text())
            deck, _, _, _ = build_deck(circuit, arrays, **report["settings"])
            self.assertEqual(report["submission_sha256"], digest)
            self.assertEqual(report["components"], counts)
            self.assertEqual(report["dataset_sha256"], content_hash(arrays))
            self.assertEqual(report["harness_sha256"], hashlib.sha256(deck.encode()).hexdigest())

    def test_pwl_compaction_preserves_waveform(self):
        points = list(zip(np.arange(20, dtype=float), [0, 0, 0, 3, 3, 3, 3, 1, 1, 0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1]))
        small = compact_pwl(points)
        self.assertLess(len(small), len(points))
        times = np.linspace(0, 19, 1001)
        np.testing.assert_allclose(np.interp(times, *zip(*points)), np.interp(times, *zip(*small)), rtol=0, atol=1e-15)
        self.assertEqual(small[0], points[0])
        self.assertEqual(small[-1], points[-1])

    def test_data_identity_is_not_archive_identity(self):
        arrays = dataset()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            np.savez(path / "plain.npz", **arrays)
            np.savez_compressed(path / "compressed.npz", **dict(reversed(list(arrays.items()))))
            self.assertNotEqual((path / "plain.npz").read_bytes(), (path / "compressed.npz").read_bytes())
            self.assertEqual(content_hash(load_data(path / "plain.npz")), content_hash(load_data(path / "compressed.npz")))
        big_endian = {key: np.asfortranarray(value.astype(">f8" if key.endswith("_x") else ">i4")) for key, value in arrays.items()}
        self.assertEqual(content_hash(arrays), content_hash(big_endian))
        big_endian["test_y"][0] = 9
        self.assertNotEqual(content_hash(arrays), content_hash(big_endian))

    def test_comments_and_actionable_grammar_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.cir"
            path.write_text("* A 10 kΩ resistor, not a µF capacitor\nRtest out0 gnd 10K ; Unicode Ω comment\n")
            self.assertEqual(validate_submission(path)[0], "rtest out0 0 10k")
            for line, message in [(".end", "without a title or .end"), ("Example title", "prefix title comments"),
                                  ("M1 out0 in0 0 0 nch W=10u L=1u M=2", "extra parameters"),
                                  ("C1 out0 0 1µF", "use 'u'")]:
                path.write_text(line)
                with self.assertRaisesRegex(ValueError, message):
                    validate_submission(path)

    def test_streaming_matches_independent_energy_integral(self):
        rng = np.random.default_rng(3)
        times = np.r_[0, np.sort(rng.uniform(0, .12, 201)), .12]
        trace = np.column_stack((times, rng.normal(size=(len(times), 10+len(SOURCES)))))
        labels = np.array([0, 1, 7])
        expected = score_trace(trace, labels, .03, .12, .03, .02)
        for rows in (1, 2, 7, 31, 1000):
            result = score_chunks((trace[i:i+rows] for i in range(0, len(trace), rows)), labels, .03, .12, .03, .02)
            self.assertEqual(result["confusion_matrix"], expected["confusion_matrix"])
            for phase, begin, end in (("startup", 0, .02), ("training", .02, .03), ("inference", .03, .12), ("total", 0, .12)):
                energy = integrate_power(times, trace[:, 11:], begin, end)
                for key, value in energy.items():
                    self.assertAlmostEqual(result["energy"][phase][key], value.sum(), places=12)

    def test_streaming_rejects_bad_boundaries_and_truncation(self):
        trace = np.zeros((5, 11+len(SOURCES)))
        trace[:, 0] = [0, .001, .002, .003, .004]
        for parts in ([trace[:3], trace[2:]], [trace[:2], trace[1:]], [trace[:2]], []):
            with self.assertRaises(ValueError):
                score_chunks(parts, np.array([0, 1, 7]), .001, .004, .001, .001)

    def test_text_chunks_ngspice_and_xyce(self):
        trace = np.arange(5*46, dtype=float).reshape(5, 46)
        with tempfile.TemporaryDirectory() as temporary:
            for backend, delimiter in (("ngspice", " "), ("xyce", ",")):
                path = Path(temporary) / backend
                np.savetxt(path, trace, delimiter=delimiter, header="header", comments="")
                if backend == "xyce":
                    with path.open("a") as output:
                        output.write("End of Xyce(TM) Simulation\n")
                np.testing.assert_array_equal(np.vstack(list(trace_chunks(path, backend, rows=2))), trace)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        np.savez(self.work / "data.npz", train_x=np.zeros((1, 16)), train_y=np.array([0]),
                 test_x=np.zeros((2, 16)), test_y=np.array([0, 0]))
        self.binary = self.work / "fake-ngspice"
        self.binary.write_text(f'''#!{sys.executable}
import os, sys, time
from pathlib import Path
if '--version' in sys.argv:
    print('** ngspice-46 : test double **')
else:
    mode = os.environ.get('TEST_SIMULATOR_MODE', 'ok')
    if mode == 'timeout':
        time.sleep(10)
    with Path('trace.txt').open('w') as out:
        out.write('header\\n')
        for t in [0, .001, .002, .003, .004]:
            out.write(' '.join(map(str, [t, 1] + [0]*44)) + '\\n')
    if mode == 'abort':
        print('tran simulation(s) aborted')
    if mode == 'bad-trace':
        Path('trace.txt').write_text('header\\nnot a numeric trace\\n')
''')
        self.binary.chmod(0o700)

    def run_cli(self, mode="ok", extra=()):
        env = dict(os.environ, TEST_SIMULATOR_MODE=mode)
        return subprocess.run([sys.executable, str(ROOT / "runner.py"), str(ROOT / "examples/constant_zero.cir"),
                               str(self.work / "data.npz"), "--binary", str(self.binary),
                               "--output", str(self.work / "result"), *extra], env=env,
                              capture_output=True, text=True, timeout=20)

    def test_success_and_opt_in_compressed_trace(self):
        result = self.run_cli(extra=("--keep-trace",))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.work / "result/report.json").read_text())
        self.assertEqual(report["accuracy"], 1)
        self.assertEqual(report["simulator"], "ngspice-46")
        self.assertEqual(report["dataset_sha256"], content_hash(load_data(self.work / "data.npz")))
        with gzip.open(self.work / "result/trace.txt.gz", "rt") as trace:
            self.assertEqual(trace.readline(), "header\n")
        self.assertFalse((self.work / "result/trace.txt").exists())

    def test_success_omits_trace_by_default(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.name for p in (self.work / "result").iterdir()}, {"report.json", "harness.cir", "simulator.log"})

    def test_zero_exit_abort_keeps_evidence_and_fails(self):
        result = self.run_cli("abort")
        self.assertEqual(result.returncode, 2)
        self.assertIn("aborted analysis", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.work / "result/report.json").exists())
        for name in ("failure.json", "harness.cir", "trace.partial.txt", "simulator.log"):
            self.assertTrue((self.work / "result" / name).exists(), name)

    def test_timeout_keeps_harness(self):
        result = self.run_cli("timeout", extra=("--timeout", "0.1"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("timed out", result.stderr)
        self.assertTrue((self.work / "result/harness.cir").exists())
        self.assertTrue((self.work / "result/failure.json").exists())

    def test_bad_trace_keeps_evidence(self):
        result = self.run_cli("bad-trace")
        self.assertEqual(result.returncode, 2)
        failure = json.loads((self.work / "result/failure.json").read_text())
        self.assertEqual(failure["stage"], "scoring")
        self.assertTrue((self.work / "result/trace.partial.txt").exists())

    def test_existing_output_has_actionable_error(self):
        (self.work / "result").mkdir()
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("choose a new --output", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_named_task_rejects_wrong_dataset_before_simulation(self):
        result = self.run_cli(extra=("--task", "mnist017-dev"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("dataset contents do not match", result.stderr)
        self.assertFalse((self.work / "result").exists())


@unittest.skipUnless(shutil.which("ngspice"), "ngspice unavailable")
class LongTransientTests(unittest.TestCase):
    def test_learning_circuit_crosses_four_seconds(self):
        # Same waveforms/devices as the failure reproducer, but start training
        # late to keep the fixture tiny. TSTART limits output, not integration.
        # Varying inputs are essential: a constant-input regression alone misses
        # the native-PWL failure even after redundant flat points are removed.
        data = dataset()
        deck = build_deck(circuit(2, [0, 1]), data, 1, 0, .001, 3.99, 1e-5)[0]
        deck = deck.split(".control")[0] + """.control
set numdgt=17
set wr_singlescale
set wr_vecnames
save v(out0)
tran 1e-5 4.13 3.989 1e-5 uic
wrdata trace.txt v(out0)
quit
.endc
.end
"""
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "harness.cir").write_text(deck)
            result = subprocess.run(["ngspice", "-n", "-b", "harness.cir"], cwd=work,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0)
            self.assertNotIn("aborted", result.stdout + result.stderr)
            trace = np.loadtxt(work / "trace.txt", skiprows=1)
            self.assertGreaterEqual(trace[-1, 0], 4.13-1e-12)
            self.assertTrue(np.all(np.diff(trace[:, 0]) > 0))


if __name__ == "__main__":
    unittest.main()
