"""CPU checks for the Gipformer notebook and its report writer."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

NOTEBOOK = Path(__file__).with_name('Gipformer15_68M_T4_Benchmark_DeepFilterNet3_PY312.ipynb')


class GipformerChecks(unittest.TestCase):
    def test_k2_optional_version_metadata_never_blocks_inference(self):
        from types import SimpleNamespace
        tree = ast.parse(NOTEBOOK.with_name('gipformer_benchmark.py').read_text())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and any(k.arg=='k2_version' for k in n.keywords))
        value = next(k.value for k in call.keywords if k.arg=='k2_version')
        expression = compile(ast.Expression(value), '<k2-version>', 'eval')
        for module, expected in [(SimpleNamespace(__version__='release'), 'release'),
                                 (SimpleNamespace(__dev_version__='development'), 'development'),
                                 (SimpleNamespace(), None)]:
            with self.subTest(expected=expected):
                self.assertEqual(eval(expression, {'k2':module}), expected)

    def test_icefall_owns_single_use_interop_thread_configuration(self):
        # Exercise the worker's actual thread/import statements against PyTorch's
        # single-call contract; importing Icefall train configures inter-op itself.
        from types import SimpleNamespace
        tree = ast.parse(NOTEBOOK.with_name('gipformer_benchmark.py').read_text())
        constructor = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name=='__init__')
        nodes = []
        for node in constructor.body:
            if isinstance(node, ast.ImportFrom) and node.module=='train':
                nodes.append(node)
            elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute)
                  and isinstance(node.value.func.value, ast.Name)
                  and node.value.func.value.id=='torch'
                  and node.value.func.attr in ('set_num_threads','set_num_interop_threads')):
                nodes.append(node)
        state = {'intra':None, 'interop':None}
        def set_interop(count):
            if state['interop'] is not None:
                raise RuntimeError('cannot set number of interop threads twice')
            state['interop']=count
        torch = SimpleNamespace(set_num_threads=lambda count:state.update(intra=count),
                                set_num_interop_threads=set_interop)
        def import_train(name, *args, **kwargs):
            self.assertEqual(name,'train')
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            return SimpleNamespace(add_model_arguments=None,get_model=None,get_params=None)
        scope = {'torch':torch,'__builtins__':{'__import__':import_train}}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<thread-startup>','exec'),scope)
        self.assertEqual(state,{'intra':2,'interop':1})

    def test_quality_reports_use_shared_metrics_and_separate_output(self):
        from gipformer_benchmark import write_quality_reports
        from quality_benchmark import error_counts
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root/'quality_results.json'
            existing.write_text('existing Zipformer results')
            rows = [dict(id='vi_0', kind='noisy', noise_type='knock', noise_style='events',
                         snr_db=5, branch=branch, audio_seconds=2, asr_seconds=.1,
                         enhancement_seconds=.02, resample_seconds=.01,
                         **error_counts('một hai', hyp))
                    for branch, hyp in [('raw','một'),('df3','một hai')]]
            target = root/'gipformer'
            write_quality_reports(target, rows, {'model': 'g-group-ai-lab/gipformer1.5-68M-rnnt'})
            self.assertEqual(existing.read_text(), 'existing Zipformer results')
            report = json.loads((target/'quality_results.json').read_text())
            self.assertEqual(report['model'], 'g-group-ai-lab/gipformer1.5-68M-rnnt')
            scores = {r['branch']:r for r in report['summary']}
            self.assertEqual(scores['raw']['wer'], .5)
            self.assertEqual(scores['df3']['wer'], 0)
            self.assertTrue((target/'quality_summary.csv').exists())
            self.assertTrue((target/'quality_per_clip.csv').exists())

    def test_audio_pair_validates_sample_rate_and_length(self):
        import numpy as np
        import soundfile as sf
        from gipformer_benchmark import load_pair
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = {'raw_16k':'raw.wav','df3_16k':'df3.wav'}
            wave = np.linspace(-.2,.2,1600,dtype=np.float32)
            sf.write(root/'raw.wav',wave,16000,subtype='FLOAT')
            sf.write(root/'df3.wav',wave*.5,16000,subtype='FLOAT')
            pair = load_pair(root,sample)
            np.testing.assert_array_equal(pair['raw'],wave)
            np.testing.assert_array_equal(pair['df3'],wave*.5)
            sf.write(root/'df3.wav',wave[:-1],16000,subtype='FLOAT')
            with self.assertRaisesRegex(RuntimeError,'lengths'):
                load_pair(root,sample)
            sf.write(root/'df3.wav',wave,8000,subtype='FLOAT')
            with self.assertRaisesRegex(RuntimeError,'16kHz'):
                load_pair(root,sample)

    def test_notebook_embeds_current_workers_and_compiles(self):
        notebook = json.loads(NOTEBOOK.read_text())
        literals = {}
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            compile(source, cell['id'], 'exec')
            self.assertNotIn('files.upload(', source)
            self.assertEqual(cell['outputs'], [])
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id.endswith('WORKER_SOURCE'):
                        literals[node.targets[0].id] = ast.literal_eval(node.value)
        for name, filename in [('QUALITY_WORKER_SOURCE','quality_benchmark.py'),
                               ('GIP_WORKER_SOURCE','gipformer_benchmark.py')]:
            self.assertEqual(literals[name].strip(), NOTEBOOK.with_name(filename).read_text().strip())
            compile(literals[name], name, 'exec')


if __name__ == '__main__':
    unittest.main()
