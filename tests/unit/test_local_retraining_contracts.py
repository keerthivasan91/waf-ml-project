import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestLocalRetrainingContracts(unittest.TestCase):
    def test_local_worker_compiles_without_training_dependencies_at_import_time(self):
        source = (ROOT / "app/services/local_retraining.py").read_text(encoding="utf-8")
        compile(source, "local_retraining.py", "exec")
        self.assertNotIn("import torch", source)

    def test_config_exposes_local_training_paths(self):
        source = (ROOT / "app/core/config.py").read_text(encoding="utf-8")
        for name in (
            "LOCAL_RETRAIN_ENABLED",
            "LOCAL_RETRAIN_AUTO_PROMOTE",
            "RETRAIN_BASE_CHECKPOINT",
            "RETRAIN_BASE_TRAIN_X",
            "RETRAIN_BASE_TRAIN_Y",
            "RETRAIN_VAL_X",
            "RETRAIN_VAL_Y",
            "RETRAIN_L2A_NORMAL_VAL",
            "RETRAIN_L2A_ATTACK_VAL",
            "RETRAIN_LOCAL_RUNS_DIR",
        ):
            self.assertIn(name, source)

    def test_feedback_route_exposes_local_start_and_status(self):
        source = (ROOT / "app/api/routes/feedback.py").read_text(encoding="utf-8")
        self.assertIn('"/local-retrain/start"', source)
        self.assertIn('"/local-retrain/status"', source)

    def test_feature_extractor_can_reload_scaler(self):
        source = (ROOT / "app/services/feature_extractor.py").read_text(encoding="utf-8")
        self.assertIn("def reload_normalizer()", source)


if __name__ == "__main__":
    unittest.main()
