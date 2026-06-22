import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "00_colab_kaggle_workflow.ipynb"


class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_all_code_cells_compile_and_have_no_saved_outputs(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_notebook_is_classifier_only(self):
        self.assertIn("CNN dish classifier", self.markdown)
        self.assertIn('BRANCH = "main"', self.code)
        self.assertIn("01_train_classifier.py", self.code)
        self.assertIn("dish_classifier.pt", self.code)
        self.assertIn("--patience", self.code)


if __name__ == "__main__":
    unittest.main()
