import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock


class FormNotebookTests(unittest.TestCase):
    def test_cells_compile_and_config_runs_without_llm_keys(self):
        notebook = json.loads(Path("Run_Form_ASR_Colab.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<cell>", "exec")
                self.assertEqual(cell["outputs"], [])
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if "SERVER_ENV = " in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            scope = dict(Path=Path, os=os, sys=sys, subprocess=subprocess,
                         REPO_DIR=Path(folder), FAKE=False, SHARE=True)
            with patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
                exec(compile(source, "<config>", "exec"), scope)
            self.assertEqual(scope["SERVER_ENV"]["SEGMENT_SILENCE_SECONDS"], "0.7")
            self.assertNotIn("LLM_API_KEY", scope["SERVER_ENV"])

    def test_bootstrap_clones_github_and_reuses_existing_checkout(self):
        notebook = json.loads(Path("Run_Form_ASR_Colab.ipynb").read_text(encoding="utf-8"))
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["cell_type"] == "code" and "REPO_DIR = " in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            work = root / "notebook"
            work.mkdir()
            expected = root / "ASR"
            # Keep the bootstrap independent of whether /content exists on the test host.
            from unittest.mock import MagicMock
            path_factory = MagicMock(side_effect=lambda p: expected if p == "/content/ASR" else root if p == "/content" else Path(p))
            path_factory.cwd.return_value = work
            test_source = source.replace("from pathlib import Path", "")
            def clone(command, check):
                self.assertEqual(command[:2], ["git", "clone"])
                self.assertTrue(command[-2].startswith("https://github.com/"))
                checkout = Path(command[-1])
                (checkout / "Web-ASR").mkdir(parents=True)
                (checkout / "Web-ASR/form_app.py").touch()
                (checkout / "Web-ASR/pyproject.toml").touch()
                (checkout / "Sub-ASR").mkdir()
                (checkout / "Sub-ASR/voice.py").touch()
            with patch("subprocess.run", side_effect=clone) as run:
                scope = {"Path": path_factory}
                exec(compile(test_source, "<bootstrap>", "exec"), scope)
                self.assertEqual(scope["PROJECT"], expected / "Web-ASR")
                run.assert_called_once()
                exec(compile(test_source, "<bootstrap>", "exec"), {"Path": path_factory})
                run.assert_called_once()

    def test_form_launch_has_no_auth_even_with_saved_credentials(self):
        import form_app
        from web_asr.config import Settings
        for username, password in [("", ""), ("old-user", "old-password")]:
            with self.subTest(saved_credentials=bool(username)):
                settings = Settings(fake=True, mode="form", share=True, host="0.0.0.0",
                                    username=username, password=password)
                ui = Mock()
                with patch.object(form_app.Settings, "load", return_value=settings), \
                     patch.object(form_app, "FormController"), \
                     patch.object(form_app, "build_ui", return_value=ui), \
                     patch.object(form_app.atexit, "register"):
                    form_app.main()
                self.assertIsNone(ui.queue.return_value.launch.call_args.kwargs["auth"])
