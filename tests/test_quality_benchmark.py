"""Run with numpy/scipy/soundfile: python -m unittest -v test_quality_benchmark.py."""
import unittest
import numpy as np
from quality_benchmark import normalize_text, error_counts, mix_at_snr, summarize


class QualityChecks(unittest.TestCase):
    def test_download_recovers_from_http_500_and_rejects_bad_checksum(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path
        import hashlib
        import tempfile
        import threading
        from quality_benchmark import download
        payload = b'valid audio archive'
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append(self.path)
                if self.path == '/failed':
                    self.send_error(500)
                    return
                body = payload if self.path == '/valid' else b'invalid payload'
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}'
            md5 = hashlib.md5(payload).hexdigest()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'archive.zip'
                download(base+'/failed', path, fallback_urls=[base+'/valid'], expected_md5=md5)
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(calls, ['/failed', '/valid'])
                download(base+'/failed', path, expected_md5=md5)
                self.assertEqual(calls, ['/failed', '/valid'], 'Valid cache must skip network')
                bad = Path(directory)/'bad.zip'
                with self.assertRaises(RuntimeError):
                    download(base+'/bad', bad, expected_md5=md5, attempts=1)
                self.assertFalse(bad.exists())
                self.assertFalse(bad.with_suffix('.zip.part').exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_event_track_has_gaps_is_reproducible_and_keeps_full_impacts(self):
        from quality_benchmark import build_noise_track, frame_activity
        sr, length = 1000, 8000
        impact = np.hanning(200).astype(np.float32)
        bank = [dict(wave=impact, file='knock_00.wav')]
        speech = np.zeros(length, dtype=np.float32)
        speech[1000:3000] = .1
        speech[4500:6500] = .1
        activity = frame_activity(speech, sr)
        first = build_noise_track(bank, length, sr, np.random.default_rng(4), 'knock', 'events', activity)
        second = build_noise_track(bank, length, sr, np.random.default_rng(4), 'knock', 'events', activity)
        track, mask, events = first
        np.testing.assert_array_equal(track, second[0])
        self.assertEqual(events, second[2])
        self.assertTrue(mask.any())
        self.assertLess(mask.mean(), .5)
        self.assertGreaterEqual(len(events), 1)
        for event in events:
            start, stop = event['start_sample'], event['end_sample']
            np.testing.assert_array_equal(track[start:stop], impact)
            self.assertIn(event['onset_region'], ('speech', 'pause', 'unconstrained'))

    def test_event_snr_ignores_silent_padding(self):
        from quality_benchmark import mix_at_snr
        speech = np.full(4000, .1, dtype=np.float32)
        noise = np.zeros_like(speech)
        noise[1000:1200] = .5
        mask = noise != 0
        mixed, target, measured, gain = mix_at_snr(speech, noise, 5, active_mask=mask,
                                                  speech_mask=np.ones(len(speech), dtype=bool))
        actual = 20*np.log10(np.sqrt(np.mean(target.astype(float)**2))/
                            np.sqrt(np.mean((mixed-target)[mask].astype(float)**2)))
        self.assertAlmostEqual(actual, 5, places=4)
        np.testing.assert_array_equal(mixed[~mask], target[~mask])

    def test_summary_keeps_noise_classes_and_styles_separate(self):
        rows = [dict(kind='noisy', noise_type=noise, noise_style=style, snr_db=5,
                     branch='raw', audio_seconds=2, asr_seconds=.1,
                     **error_counts('một hai', 'một'))
                for noise, style in [('keyboard','continuous'),('keyboard','events'),('knock','events')]]
        result = summarize(rows)
        self.assertEqual(len(result), 3)
        self.assertEqual({(r['noise_type'],r['noise_style']) for r in result},
                         {('keyboard','continuous'),('keyboard','events'),('knock','events')})

    def test_snr_peak_and_reproducibility(self):
        rng = np.random.default_rng(42)
        speech = rng.normal(0, .1, 48000).astype(np.float32)
        noise = rng.normal(0, .2, 48000).astype(np.float32)
        for snr in (0, 5, 10):
            mixed, target, actual_snr, gain = mix_at_snr(speech, noise, snr)
            self.assertEqual(mixed.shape, speech.shape)
            self.assertEqual(mixed.dtype, np.float32)
            self.assertAlmostEqual(actual_snr, snr, places=4)
            measured = 10*np.log10(np.sum(target.astype(float)**2)/np.sum((mixed-target).astype(float)**2))
            self.assertAlmostEqual(measured, snr, places=4)
            self.assertLessEqual(float(np.max(np.abs(mixed))), .950001)
        with self.assertRaises(ValueError):
            mix_at_snr(speech, np.zeros_like(noise), 0)
        with self.assertRaises(ValueError):
            mix_at_snr(speech, noise[:-1], 0)
        with self.assertRaises(ValueError):
            mix_at_snr(speech*np.nan, noise, 0)

    def test_vietnamese_normalization_and_edit_counts(self):
        self.assertEqual(normalize_text('  XIN CHÀO, Việt Nam!  '), 'xin chào việt nam')
        counts = error_counts('Xin chào Việt Nam', 'xin chào việt')
        self.assertEqual(counts['word_errors'], 1)
        self.assertEqual(counts['reference_words'], 4)
        self.assertEqual(counts['wer'], .25)
        self.assertEqual(error_counts('má', 'ma')['char_errors'], 1)
        noise = error_counts('', 'xin chào')
        self.assertIsNone(noise['wer'])
        self.assertIsNone(noise['cer'])
        self.assertTrue(noise['has_text'])
        self.assertFalse(error_counts('', '...')['has_text'])

    def test_corpus_rates_are_weighted_and_noise_is_separate(self):
        rows = []
        for ref, hyp in [('một hai ba bốn', 'một hai ba'), ('năm', '')]:
            rows.append(dict(kind='noisy', snr_db=5, branch='raw', audio_seconds=2,
                             asr_seconds=.1, **error_counts(ref, hyp)))
        rows.append(dict(kind='noise_only', snr_db=None, branch='raw', audio_seconds=2,
                         asr_seconds=.1, **error_counts('', 'sai')))
        groups = summarize(rows)
        speech = next(x for x in groups if x['kind']=='noisy')
        self.assertAlmostEqual(speech['wer'], 2/5)
        noise = next(x for x in groups if x['kind']=='noise_only')
        self.assertIsNone(noise['wer'])
        self.assertEqual(noise['nonempty_transcript_rate'], 1)


if __name__ == '__main__':
    unittest.main()
