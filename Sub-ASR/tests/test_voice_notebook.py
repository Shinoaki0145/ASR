"""Check that the conversation notebook runs from the consolidated project layout."""
import ast
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class VoiceNotebookTests(unittest.TestCase):
    def test_bootstrap_finds_sources_from_repo_and_project(self):
        notebook = PROJECT / 'Hoi_Thoai_Ngat_Loi_Testing.ipynb'
        self.assertTrue(notebook.is_file(), 'Notebook must live next to voice.py')
        cells = {c['id']: ''.join(c['source']) for c in json.loads(notebook.read_text())['cells']}
        tree = ast.parse(cells['bootstrap'])
        # Display helpers need a notebook kernel; path resolution does not.
        tree.body = [n for n in tree.body if not (
            isinstance(n, ast.ImportFrom) and n.module == 'IPython.display')]
        for folder in (PROJECT.parent, PROJECT):
            with self.subTest(folder=folder), chdir(folder), redirect_stdout(io.StringIO()):
                scope = {}
                exec(compile(tree, 'bootstrap', 'exec'), scope)
                self.assertEqual(scope['WORKSPACE'], PROJECT)
                self.assertTrue((scope['WORKSPACE'] / 'voice.py').is_file())
                self.assertTrue((scope['WORKSPACE'] / 'test_voice.py').is_file())
                env_assignment = ast.parse(cells['config']).body[0]
                exec(compile(ast.Module(body=[env_assignment], type_ignores=[]), 'config', 'exec'), scope)
                self.assertEqual(scope['ENV_PATH'], PROJECT.parent / '.env')
        self.assertNotIn('notebook_runtime', '\n'.join(cells.values()))
        self.assertNotIn('BUNDLE_B64', '\n'.join(cells.values()))
