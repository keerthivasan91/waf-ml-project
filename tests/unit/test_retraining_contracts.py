import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestRetrainingContracts(unittest.TestCase):
    def test_query_imports_retrain_batch_collection(self):
        tree = ast.parse((ROOT / "app/db/queries.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "app.db.collections":
                imported.update(alias.name for alias in node.names)
        self.assertIn("retrain_batches", imported)

    def test_offline_script_has_required_cli_contract(self):
        source = (ROOT / "ml/retraining/offline_retrain.py").read_text(encoding="utf-8")
        compile(source, "offline_retrain.py", "exec")
        for flag in (
            "--batch",
            "--base-checkpoint",
            "--base-train-x",
            "--base-train-y",
            "--val-x",
            "--val-y",
            "--l2a-normal-val",
            "--l2a-attack-val",
            "--output-dir",
        ):
            self.assertIn(flag, source)

    def test_feedback_route_exposes_batch_export(self):
        source = (ROOT / "app/api/routes/feedback.py").read_text(encoding="utf-8")
        self.assertIn('"/retrain-batches/latest"', source)
        self.assertIn('"/retrain-batches/{batch_id}/export"', source)


if __name__ == "__main__":
    unittest.main()
