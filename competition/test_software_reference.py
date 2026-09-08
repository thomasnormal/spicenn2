"""The software reference must reject different data before training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from competition.data import content_hash
from competition import software_reference


class FrozenDatasetTests(unittest.TestCase):
    def test_checks_array_content_not_archive_encoding_or_sidecar(self):
        arrays = {
            "train_x": np.full((2, 16), 0.5), "train_y": np.array([0, 1]),
            "test_x": np.full((2, 16), 0.25), "test_y": np.array([1, 0]),
        }
        digest = content_hash(arrays)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "mnist017-v0.json").write_text(json.dumps({"dataset_sha256": digest}))
            data_path = root / "data.npz"
            with patch.object(software_reference, "TASK_DIRECTORY", root):
                # Both encodings of exactly the same values are accepted.
                for save in (np.savez, np.savez_compressed):
                    save(data_path, **arrays)
                    _, _, actual = software_reference.verified_dataset("mnist017-v0", data_path)
                    self.assertEqual(actual, digest)
                # A plausible sidecar cannot rescue altered features or labels.
                data_path.with_suffix(".json").write_text(json.dumps({"dataset_sha256": digest}))
                for key in ("train_x", "train_y", "test_x", "test_y"):
                    with self.subTest(key=key):
                        altered = {name: array.copy() for name, array in arrays.items()}
                        altered[key].flat[0] = 0
                        if np.array_equal(altered[key], arrays[key]):
                            altered[key].flat[0] = 1
                        np.savez(data_path, **altered)
                        with self.assertRaisesRegex(ValueError, "content hash does not match"):
                            software_reference.evaluate("mnist017-v0", data_path)

    def test_rejects_unfrozen_task(self):
        with self.assertRaisesRegex(ValueError, "frozen tasks"):
            software_reference.verified_dataset("mnist10-dev", "unused.npz")


if __name__ == "__main__":
    unittest.main()
