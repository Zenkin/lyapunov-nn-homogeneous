"""Output handling must preserve references and distinguish failed runs."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from launcher import run


class OutputTests(unittest.TestCase):
    def test_latest_gallery_keeps_reference_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "RESULTS.html"
            reference.write_text("preserved reference", encoding="utf-8")
            folder = root / "results/run/reproduction"
            folder.mkdir(parents=True)
            (folder / "RESULTS.html").write_text(
                '<a href="verification.json"><img src="figure.png"></a></html>',
                encoding="utf-8",
            )
            with patch.object(run, "ROOT", root):
                run.publish_gallery(folder)
            self.assertEqual(reference.read_text(), "preserved reference")
            latest = (root / "results/LATEST.html").read_text()
            self.assertIn('href="run/reproduction/verification.json"', latest)
            self.assertIn('src="run/reproduction/figure.png"', latest)

    def test_failed_reproduction_preserves_previous_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            latest = root / "results/LATEST.html"
            latest.write_text("previous successful run", encoding="utf-8")
            with patch.object(run, "ROOT", root), patch.object(
                run, "execute", return_value=1
            ), patch("sys.argv", ["run.py", "--action", "reproduce"]):
                self.assertEqual(run.main(), 1)
            self.assertEqual(latest.read_text(), "previous successful run")

    def test_unrelated_process_failure_is_not_an_audit_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "run.log"
            log.write_text("ImportError: unavailable dependency", encoding="utf-8")
            self.assertFalse(run.expected_audit_failure(root, 1, log))


if __name__ == "__main__":
    unittest.main()
