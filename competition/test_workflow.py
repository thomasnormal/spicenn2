"""Submission scaffolding and simulator isolation tests."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from competition import new_submission
from competition.container_backend import ContainerBackend
from competition.runner import safe_trace
from competition.validate import validate_folder


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_scaffold_is_valid_and_retains_attribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "submissions").mkdir()
            (root / "competition/examples").mkdir(parents=True)
            shutil.copyfile(ROOT / "LICENSE", root / "LICENSE")
            shutil.copyfile(ROOT / "competition/examples/mnist_017_tuned.cir", root / "competition/examples/mnist_017_tuned.cir")
            with patch.object(new_submission, "ROOT", root), patch.object(sys, "argv", ["new_submission.py", "my-entry", "--author", "Example Author"]), contextlib.redirect_stdout(io.StringIO()):
                new_submission.main()
            folder = root / "submissions/my-entry"
            self.assertEqual(sum(validate_folder(folder)[1].values()), 546)
            license_text = (folder / "LICENSE").read_text()
            self.assertIn("Thomas Dybdahl Ahle", license_text)
            self.assertIn("Example Author", license_text)
            self.assertIn("MIT License", license_text)
            self.assertIn("--task mnist017-v0", (folder / "README.md").read_text())

    def test_artifact_links_are_not_followed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret = root / "outside"
            secret.write_text("not a trace")
            symbolic = root / "trace.txt"
            symbolic.symlink_to(secret)
            with self.assertRaisesRegex(ValueError, "not a link"):
                safe_trace(symbolic)
            hard = root / "hard.txt"
            os.link(secret, hard)
            with self.assertRaisesRegex(ValueError, "not a link"):
                safe_trace(hard)

    def test_scaffold_main_track_and_tutorial_presets(self):
        for baseline, task, components in (("mnist_10", "mnist10-v0", 1820),
                                           ("mnist_017", "mnist017-v0", 546),
                                           ("blobs", "blobs-dev", 84)):
            with self.subTest(baseline=baseline), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "submissions").mkdir()
                (root / "competition/examples").mkdir(parents=True)
                shutil.copyfile(ROOT / "LICENSE", root / "LICENSE")
                shutil.copyfile(ROOT / "competition/examples" / (baseline + ".cir"),
                                root / "competition/examples" / (baseline + ".cir"))
                argv = ["new_submission.py", "my-entry", "--author", "Example Author", "--baseline", baseline]
                with patch.object(new_submission, "ROOT", root), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    new_submission.main()
                folder = root / "submissions/my-entry"
                self.assertEqual(sum(validate_folder(folder)[1].values()), components)
                self.assertIn("--task " + task, (folder / "README.md").read_text())

    def test_container_uses_immutable_image_and_restricted_mounts(self):
        identity = "sha256:" + "a" * 64
        result = subprocess.CompletedProcess([], 0, identity+"\n", "")
        with patch("competition.container_backend.shutil.which", return_value="docker"), \
                patch("competition.container_backend.subprocess.run", return_value=result), \
                patch("competition.container_backend.os.getuid", return_value=1000), \
                patch("competition.container_backend.os.getgid", return_value=1000):
            backend = ContainerBackend("example:tag")
        command = backend.create_command(Path("/tmp/isolated-test"))
        for value in (identity, "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                      "--memory=4g", "--memory-swap=4g", "--pids-limit=64", "--cpus=2", "--pull=never"):
            self.assertIn(value, command)
        self.assertEqual(command.count("--mount"), 1)
        self.assertIn("type=bind,src=/tmp/isolated-test,dst=/work", command)
        self.assertNotIn("example:tag", command)
        self.assertTrue(backend.name.startswith("spicenn2-simulator-"))


@unittest.skipUnless(os.environ.get("TEST_DOCKER_IMAGE"), "set TEST_DOCKER_IMAGE for container integration tests")
class ContainerIntegrationTests(unittest.TestCase):
    def test_scoring_and_timeout_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            np.savez(work / "data.npz", train_x=np.zeros((1, 16)), train_y=np.array([0]),
                     test_x=np.zeros((2, 16)), test_y=np.array([0, 1]))
            command = [sys.executable, str(ROOT / "competition/runner.py"), str(ROOT / "competition/examples/constant_zero.cir"),
                       str(work / "data.npz"), "--docker-image", os.environ["TEST_DOCKER_IMAGE"]]
            result = subprocess.run(command + ["--output", str(work / "success")], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((work / "success/report.json").read_text())
            self.assertEqual(report["accuracy"], .5)
            self.assertTrue(report["docker_image_id"].startswith("sha256:"))
            self.assertAlmostEqual(report["energy"]["inference"]["delivered_j_per_image"], 4.5e-7, delta=1e-10)
            # Snapshot container names so this test does not make assumptions
            # about unrelated scoring jobs running at the same time.
            list_command = ["docker", "ps", "-a", "--filter", "name=spicenn2-simulator-", "--format", "{{.Names}}"]
            before = set(subprocess.check_output(list_command, text=True).splitlines())
            result = subprocess.run(command + ["--output", str(work / "timeout"), "--timeout", "0.001"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertTrue((work / "timeout/failure.json").exists())
            after = set(subprocess.check_output(list_command, text=True).splitlines())
            self.assertFalse(after - before, f"orphaned simulator containers: {after - before}")


if __name__ == "__main__":
    unittest.main()
