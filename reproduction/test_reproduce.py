"""Checkpoint provenance and output preservation checks."""

from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

from reproduction import reproduce


class ReproductionSafetyTests(unittest.TestCase):
    def test_historical_artifacts_match_provenance_hashes(self):
        folder = reproduce.ROOT / "reproduction/checkpoints/example2-d22b775"
        manifest = json.loads((folder / "SOURCE.json").read_text(encoding="utf-8"))
        for name, digest in manifest["files_sha256"].items():
            with self.subTest(file=name):
                self.assertEqual(
                    hashlib.sha256((folder / name).read_bytes()).hexdigest(), digest
                )

    def test_modified_checkpoint_is_rejected_before_loading(self):
        names = (
            "example_1/improved/figures/reference/models.pt",
            "reproduction/checkpoints/example2-d22b775/model_state.pt",
            "reproduction/checkpoints/SHA256.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(reproduce.ROOT / name, target)
            with (root / names[0]).open("ab") as file:
                file.write(b"changed")
            with patch.object(reproduce, "ROOT", root), patch.object(
                reproduce.torch, "load"
            ) as load:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    reproduce.load_models()
                load.assert_not_called()

    def test_existing_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "verification.json"
            marker.write_text("preserved", encoding="utf-8")
            with patch("sys.argv", ["reproduce.py", "--outdir", directory]):
                with self.assertRaises(FileExistsError):
                    reproduce.main()
            self.assertEqual(marker.read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()
