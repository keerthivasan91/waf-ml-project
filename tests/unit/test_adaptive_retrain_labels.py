import unittest
from unittest.mock import patch

from app.services import adaptive_retrain


class TestAdaptiveRetrainLabels(unittest.TestCase):
    @patch("app.services.adaptive_retrain.reaudit")
    def test_false_positive_uses_normal_cross_agreement(self, reaudit):
        reaudit.return_value = {"label": "normal"}
        passed, reason = adaptive_retrain._cross_agreement_pass({
            "verified_label": "false_positive",
            "url": "/products",
            "method": "GET",
            "body": "",
        })
        self.assertTrue(passed)
        self.assertEqual(reason, "")

    @patch("app.services.adaptive_retrain.reaudit")
    def test_false_positive_rejects_when_current_model_still_flags_attack(self, reaudit):
        reaudit.return_value = {"label": "xss"}
        passed, reason = adaptive_retrain._cross_agreement_pass({
            "verified_label": "false_positive",
            "url": "/products",
            "method": "GET",
            "body": "",
        })
        self.assertFalse(passed)
        self.assertEqual(reason, "cross_agreement_failed_normal_flagged_as_attack")


if __name__ == "__main__":
    unittest.main()
