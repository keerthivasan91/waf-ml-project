import ast
import unittest
from pathlib import Path


class TestQueryCollectionImports(unittest.TestCase):
    def test_retrain_log_collection_is_imported(self):
        path = Path(__file__).resolve().parents[2] / "app" / "db" / "queries.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "app.db.collections":
                imported_names.update(alias.name for alias in node.names)

        self.assertIn("retrain_log", imported_names)


if __name__ == "__main__":
    unittest.main()
