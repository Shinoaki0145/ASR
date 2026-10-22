import json
from pathlib import Path
import unittest


class NotebookTests(unittest.TestCase):
    def test_clean_and_compiles(self):
        nb = json.loads(Path("Run_Web_ASR_Colab.ipynb").read_text())
        self.assertEqual(nb["nbformat"], 4)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<cell>", "exec")
                self.assertEqual(cell["outputs"], [])
        source = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertNotIn("BUNDLE_B64", source)
        self.assertIn("'sync', '--locked'", source)
        self.assertEqual(source.count("subprocess.Popen("), 1)
