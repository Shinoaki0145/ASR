"""Run: uv run --with numpy --with imageio-ffmpeg python -m unittest -v test_silero_notebook.py"""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave

import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent


def notebook_decoder():
    notebook = json.loads((ROOT / 'Silero_VAD_Test.ipynb').read_text())
    functions = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
                 and cell.get('id') != 'install'
                 for node in ast.parse(''.join(cell['source'])).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'decode_audio_memory']
    if len(functions) != 1:
        raise AssertionError('Notebook needs one decode_audio_memory function')
    scope = dict(Path=Path, np=np, subprocess=subprocess,
                 shutil=shutil, imageio_ffmpeg=imageio_ffmpeg)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<notebook-decoder>', 'exec'), scope)
    return scope['decode_audio_memory']


class DecoderChecks(unittest.TestCase):
    def test_problem_mp3_decodes_without_intermediate_files(self):
        decode = notebook_decoder()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input with spaces.mp3'
            shutil.copyfile(ROOT / 'test_1.mp3', path)
            samples, diagnostic = decode(path, 16000)
            self.assertEqual(samples.ndim, 1)
            self.assertEqual(samples.dtype, np.float32)
            self.assertTrue(samples.flags.writeable)
            self.assertTrue(np.isfinite(samples).all())
            self.assertTrue(13 < samples.size / 16000 < 15)
            self.assertIn('Header missing', diagnostic)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_stereo_resampling_and_invalid_input(self):
        decode = notebook_decoder()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stereo.wav'
            with wave.open(str(path), 'wb') as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(48000)
                output.writeframes(bytes(48000 * 2 * 2))
            samples, _ = decode(path, 16000)
            self.assertEqual(samples.shape, (16000,))
            self.assertTrue((samples == 0).all())
            path.write_bytes(b'not an audio file')
            with self.assertRaisesRegex(RuntimeError, 'FFmpeg'):
                decode(path, 16000)
            with self.assertRaises(FileNotFoundError):
                decode(Path(directory) / 'missing.mp3', 16000)


if __name__ == '__main__':
    unittest.main()
