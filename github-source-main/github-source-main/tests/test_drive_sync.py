import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cloud" / "01_sync_drive_artifacts.py"
SPEC = importlib.util.spec_from_file_location("drive_sync", SCRIPT)
drive_sync = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(drive_sync)


class DriveSyncTests(unittest.TestCase):
    def make_args(self, **overrides):
        values = {
            "push_inputs": False,
            "publish": False,
            "all": False,
            "pull_results": False,
            "pull": False,
            "push_packages": False,
            "push_project_files": False,
            "push_models": False,
            "pull_models": False,
            "pull_latest_runs": False,
            "pull_latest_run": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_publish_alias_never_pushes_models(self):
        operations = drive_sync.requested_operations(self.make_args(publish=True))
        self.assertTrue(operations["push_packages"])
        self.assertTrue(operations["push_project_files"])
        self.assertFalse(operations["push_models"])

    def test_pull_results_fetches_models_and_all_run_kinds(self):
        operations = drive_sync.requested_operations(self.make_args(pull_results=True))
        self.assertTrue(operations["pull_models"])
        self.assertTrue(operations["pull_latest_runs"])

    def test_latest_model_run_is_scoped_by_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive_root = Path(tmp)
            older = drive_root / "runs" / "classifier" / "20260101_000000"
            newer = drive_root / "runs" / "classifier" / "20260102_000000"
            unrelated = drive_root / "runs" / "other" / "20260103_000000"
            for path in (older, newer, unrelated):
                path.mkdir(parents=True)
            self.assertEqual(drive_sync.latest_model_run(drive_root, "classifier"), newer)


if __name__ == "__main__":
    unittest.main()
