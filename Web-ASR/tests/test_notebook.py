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

    def test_uploaded_env_loaded_before_prompts(self):
        import os, tempfile, getpass, subprocess, sys, io
        from contextlib import redirect_stdout
        from unittest.mock import patch

        nb = json.loads(Path("Run_Web_ASR_Colab.ipynb").read_text())
        source = next(
            "".join(c["source"])
            for c in nb["cells"]
            if c["cell_type"] == "code" and "SERVER_ENV = " in "".join(c["source"])
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "ASR"
            root.mkdir()
            env = Path(folder) / ".env"
            env.write_text(
                'LLM_BASE_URL=https://example.test/v1\nLLM_MODEL=test\nLLM_API_KEY="test-secret"\nWEB_USERNAME=user\nWEB_PASSWORD="web-secret"\n'
            )
            scope = dict(
                os=os,
                Path=Path,
                REPO_DIR=root,
                FAKE=False,
                SHARE=True,
                getpass=getpass,
                subprocess=subprocess,
                sys=sys,
            )
            output = io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "builtins.input", side_effect=AssertionError("unexpected prompt")
                ),
                patch(
                    "getpass.getpass",
                    side_effect=AssertionError("unexpected password prompt"),
                ),
                redirect_stdout(output),
            ):
                exec(compile(source, "<config-cell>", "exec"), scope)
            self.assertEqual(scope["SERVER_ENV"]["ASR_ENV_PATH"], str(env))
            self.assertEqual(scope["SERVER_ENV"]["LLM_API_KEY"], "test-secret")
            self.assertEqual(scope["SERVER_ENV"]["WEB_PASSWORD"], "web-secret")
            self.assertNotIn("test-secret", output.getvalue())
            self.assertNotIn("web-secret", output.getvalue())
