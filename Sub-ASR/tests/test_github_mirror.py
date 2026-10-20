import ast
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def resolver(run):
    tree = ast.parse((ROOT/'molab_app.py').read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='resolve_project')
    namespace = {'Path':Path,'subprocess':Mock(run=run)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'mirror','exec'),namespace)
    return namespace['resolve_project']


def project_at(root):
    p = root/'sub_asr/upload.py'
    p.parent.mkdir(parents=True)
    p.write_text('# test source')


class MirrorTests(unittest.TestCase):
    def test_local_source_never_fetches_or_changes_git(self):
        run = Mock()
        with tempfile.TemporaryDirectory() as d:
            here = Path(d)/'local'
            project_at(here)
            self.assertEqual(resolver(run)(here,Path(d),update=True),here)
        run.assert_not_called()

    def test_notebook_only_clone_fetches_missing_modules(self):
        with tempfile.TemporaryDirectory() as d:
            cwd = Path(d)
            def clone(command,**kwargs):
                self.assertEqual(command[:3],['git','clone','--depth'])
                project_at(Path(command[-1])/'Sub-ASR')
            run = Mock(side_effect=clone)
            self.assertEqual(resolver(run)(cwd/'notebook',cwd),cwd/'.sub_asr_source/Sub-ASR')
            run.assert_called_once()

    def test_cached_source_is_updated_only_on_prepare(self):
        run = Mock()
        with tempfile.TemporaryDirectory() as d:
            cwd = Path(d)
            project_at(cwd/'.sub_asr_source/Sub-ASR')
            resolve = resolver(run)
            resolve(cwd/'notebook',cwd)
            run.assert_not_called()
            resolve(cwd/'notebook',cwd,update=True)
            self.assertEqual(run.call_count,2)
            self.assertIn('fetch',run.call_args_list[0].args[0])
            self.assertNotIn('--force',run.call_args_list[1].args[0])

    def test_no_archive_workflow_in_notebook_or_readme(self):
        for name in ('molab_app.py','README.md'):
            text = (ROOT/name).read_text()
            self.assertNotIn('Sub-ASR-molab.zip',text)
            self.assertNotIn('zipfile',text)
