"""CPU-only regression checks; run: python3 -m unittest -v test_notebook.py."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter

NOTEBOOK = Path(__file__).with_name('Zipformer_30M_T4_Benchmark_DeepFilterNet3_PY312_CUDA_FIXED.ipynb')


def worker_source(name):
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell['cell_type'] != 'code' or name not in ''.join(cell['source']):
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f'Missing standalone worker: {name}')


class NotebookChecks(unittest.TestCase):
    def test_cuda_guard_requires_executed_cuda_nodes(self):
        tree = ast.parse(worker_source('ASR_WORKER_SOURCE'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'inspect_cuda_profiles')
        scope = dict(json=json, Path=Path, Counter=Counter)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<guard>', 'exec'), scope)
        inspect = scope['inspect_cuda_profiles']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            for events in [[], [{'cat': 'Node', 'args': {'provider': 'CPUExecutionProvider'}}],
                           [{'cat': 'Session', 'args': {'provider': 'CUDAExecutionProvider'}}]]:
                path.write_text(json.dumps(events))
                with self.assertRaises(RuntimeError):
                    inspect([path])
            path.write_text(json.dumps([
                {'cat': 'Node', 'args': {'provider': 'CUDAExecutionProvider'}},
                {'cat': 'Node', 'args': {'provider': 'CPUExecutionProvider'}}]))
            self.assertEqual(inspect([path]), {'CUDAExecutionProvider': 1, 'CPUExecutionProvider': 1})
            with self.assertRaises(RuntimeError):
                inspect([])

    def test_nvml_memory_without_optional_constant(self):
        from types import SimpleNamespace
        tree = ast.parse(worker_source('ASR_WORKER_SOURCE'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'memory')
        processes = []
        nvml = SimpleNamespace(
            nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=8 * 2**20),
            nvmlDeviceGetComputeRunningProcesses=lambda handle: processes)
        scope = dict(pynvml=nvml, handle=object(), os=SimpleNamespace(getpid=lambda: 42))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<nvml-memory>', 'exec'), scope)
        for used, expected in [(3 * 2**20, 3.0), (0, 0.0), (None, None), ((1 << 64)-1, None)]:
            with self.subTest(used=used):
                processes[:] = [SimpleNamespace(pid=99, usedGpuMemory=100 * 2**20),
                                SimpleNamespace(pid=42, usedGpuMemory=used)]
                self.assertEqual(scope['memory'](), {'total_mib': 8.0, 'process_mib': expected})
        processes.clear()
        self.assertIsNone(scope['memory']()['process_mib'])

    def test_native_log_distinguishes_node_placement_from_provider_failure(self):
        nb = json.loads(NOTEBOOK.read_text())
        tree = ast.parse(''.join(next(c for c in nb['cells'] if c['id'] == 'probe')['source']))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'native_cpu_fallback_lines']
        self.assertEqual(len(functions), 1, 'Need a log classifier that permits ORT CPU node placement')
        scope = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<log-classifier>', 'exec'), scope)
        classify = scope['native_cpu_fallback_lines']
        placement = ('[I:onnxruntime:, fallback_cpu_capability.cc:167 GetCpuPreferredNodes] '
                     'ORT optimization- Force fallback to CPU execution for node: Gather '
                     'because the CPU execution path is deemed faster')
        self.assertEqual(classify(placement), [])
        failure = 'session.cc: Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON. Fallback to cpu!'
        self.assertEqual(classify(placement + '\n' + failure), [failure])
        self.assertEqual(classify('Unsupported string: cuda:config. Fallback to cpu'),
                         ['Unsupported string: cuda:config. Fallback to cpu'])
        self.assertEqual(classify('Failed to enable CUDA. Fall back to CPU'),
                         ['Failed to enable CUDA. Fall back to CPU'])
        self.assertEqual(classify('Adding default CPU execution provider.'), [])

    def test_profile_config_has_actual_newlines(self):
        tree = ast.parse(worker_source('ASR_WORKER_SOURCE'))
        call = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'config'
                    and node.func.attr == 'write_text')
        value = eval(compile(ast.Expression(call.args[0]), '<config>', 'eval'),
                     {'profile_dir': Path('/tmp/probe')})
        self.assertEqual(value.splitlines(), ['ProfilingFilePrefix=/tmp/probe/sherpa', 'LogSeverityLevel=1'])

    def test_audio_branches_preserve_float_amplitude_and_length(self):
        try:
            import numpy as np
            import soundfile as sf
            from scipy.signal import resample_poly
        except ImportError:
            self.skipTest('Run with numpy, scipy, soundfile for audio verification')
        from types import SimpleNamespace
        upload = next(c for c in json.loads(NOTEBOOK.read_text())['cells'] if c['id'] == 'dataset-select')
        sources = [(''.join(upload['source']), 'raw_16k'),
                   (worker_source('DF3_WORKER_SOURCE'), 'cleaned_16k')]
        wave = np.full(48007, 1.5, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for source, name in sources:
                tree = ast.parse(source)
                processing = next(node for node in tree.body if isinstance(node, ast.Assign)
                                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
                writing = next(node for node in tree.body if isinstance(node, ast.Expr)
                               and isinstance(node.value, ast.Call)
                               and isinstance(node.value.func, ast.Attribute)
                               and isinstance(node.value.func.value, ast.Name)
                               and node.value.func.value.id == 'sf' and node.value.func.attr == 'write'
                               and len(node.value.args) > 1 and isinstance(node.value.args[1], ast.Name)
                               and node.value.args[1].id == name)
                path = str(Path(directory)/f'{name}.wav')
                scope = dict(np=np, sf=sf, resample_poly=resample_poly, wave_48k=wave,
                             cleaned_array=wave.copy(), audio_path=path, a=SimpleNamespace(output=path))
                exec(compile(ast.Module(body=[processing, writing], type_ignores=[]), '<audio>', 'exec'), scope)
                data, sr = sf.read(path, dtype='float32')
                self.assertEqual(sr, 16000)
                self.assertEqual(len(data), 16003)
                self.assertEqual(sf.info(path).subtype, 'FLOAT')
                self.assertGreater(float(data.max()), 1.4)
                outputs.append(data)
            np.testing.assert_array_equal(*outputs)

    def test_quality_worker_is_embedded_without_external_source_files(self):
        expected = NOTEBOOK.with_name('quality_benchmark.py').read_text()
        self.assertEqual(worker_source('QUALITY_WORKER_SOURCE').strip(), expected.strip())

    def test_dataset_only_flow_prepares_audio_before_probe(self):
        cells = json.loads(NOTEBOOK.read_text())['cells']
        ids = [c['id'] for c in cells]
        self.assertLess(ids.index('dataset-select'), ids.index('df-latency'))
        self.assertLess(ids.index('df-latency'), ids.index('probe'))
        self.assertLess(ids.index('probe'), ids.index('score'))
        sources = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
        self.assertNotIn('files.upload(', sources)
        self.assertNotIn('original = next(iter(uploaded))', sources)

    def test_cells_and_workers_compile_and_have_no_saved_results(self):
        nb = json.loads(NOTEBOOK.read_text())
        for index, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell-{index}', 'exec')
                self.assertEqual(cell.get('outputs'), [])
                self.assertIsNone(cell.get('execution_count'))
        for name in ['ASR_WORKER_SOURCE', 'DF3_WORKER_SOURCE', 'QUALITY_WORKER_SOURCE']:
            compile(worker_source(name), name, 'exec')


if __name__ == '__main__':
    unittest.main()
