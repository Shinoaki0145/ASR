import io
import unittest
import numpy as np
import soundfile as sf
from sub_asr.upload import decode_audio, transcribe_upload


def wav_bytes(samples, sr):
    stream = io.BytesIO()
    sf.write(stream,samples,sr,format='WAV',subtype='FLOAT')
    return stream.getvalue()


class UploadTests(unittest.TestCase):
    def test_resamples_stereo_without_peak_normalization(self):
        t = np.arange(48000)/48000
        stereo = np.column_stack((0.2*np.sin(2*np.pi*440*t),np.zeros(len(t))))
        wave, metadata = decode_audio(wav_bytes(stereo,48000))
        self.assertEqual(wave.shape,(16000,))
        self.assertEqual(wave.dtype,np.float32)
        self.assertEqual(metadata['original_sample_rate'],48000)
        self.assertEqual(metadata['original_channels'],2)
        self.assertAlmostEqual(float(np.max(np.abs(wave))),0.1,delta=0.002)

    def test_rejects_too_long_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError,'duration'):
            decode_audio(wav_bytes(np.zeros(32000),16000),max_seconds=1)

    def test_rejects_empty_or_nonfinite_audio(self):
        for wave in (np.array([],dtype=np.float32),np.array([float('nan')],dtype=np.float32)):
            with self.assertRaises(ValueError):
                decode_audio(wav_bytes(wave,16000))

    def test_preserves_pause_and_entire_waveform_for_asr(self):
        wave = np.concatenate((np.ones(16000)*0.1,np.zeros(32000),np.ones(16000)*0.2)).astype(np.float32)
        class Backend:
            metadata = {'decoding_method':'greedy_search'}
            def recognize(self, audio):
                np.testing.assert_array_equal(audio,wave)
                return 'Tôi muốn đi Đà Nẵng',0.1
        result = transcribe_upload(wave,Backend(),lambda audio:[{'start':0.,'end':1.}, {'start':3.,'end':4.}])
        self.assertEqual(result['transcript'],'Tôi muốn đi Đà Nẵng')
        self.assertEqual(result['audio_seconds'],4)
        self.assertTrue(result['speech_detected'])

    def test_no_speech_skips_recognizer(self):
        class Backend:
            metadata = {}
            def recognize(self,audio):
                raise AssertionError('Noise-only must not be sent to ASR')
        result = transcribe_upload(np.zeros(16000,dtype=np.float32),Backend(),lambda audio:[])
        self.assertEqual(result['transcript'],'')
        self.assertFalse(result['speech_detected'])
