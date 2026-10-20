import unittest
import numpy as np
from sub_asr.turns import TurnBuffer, TurnConfig


class TurnTests(unittest.TestCase):
    def setUp(self):
        self.turn = TurnBuffer(TurnConfig(min_silence=0.32, incomplete_silence=0.96,
                                         max_wait=1.6, start_speech=0.096, max_turn=5))

    def feed(self, seconds, speech):
        for _ in range(round(seconds / 0.032)):
            self.turn.feed(np.ones(512, dtype=np.float32), 0.9 if speech else 0.01)

    def transcribe(self, text='Tôi muốn đi'):
        snapshot = self.turn.snapshot()
        self.assertTrue(self.turn.set_text(snapshot.turn_id, snapshot.revision, text))
        return snapshot

    def test_noise_does_not_start_turn(self):
        self.feed(2, False)
        self.assertFalse(self.turn.active)
        self.assertFalse(self.turn.should_finalize())

    def test_short_click_does_not_start_turn(self):
        self.feed(0.032, True)
        self.feed(0.32, False)
        self.assertFalse(self.turn.active)

    def test_pause_then_speech_retains_same_audio_and_context(self):
        self.feed(0.32, True)
        first = self.transcribe()
        self.feed(0.32, False)
        self.turn.set_endpoint(first.turn_id, first.revision, False)
        self.assertFalse(self.turn.should_finalize())
        self.feed(0.32, True)
        second = self.transcribe('Tôi muốn đi Đà Nẵng')
        self.assertEqual(first.turn_id, second.turn_id)
        self.assertGreater(len(second.audio), len(first.audio))
        self.assertEqual(self.turn.text, 'Tôi muốn đi Đà Nẵng')

    def test_complete_sentence_ends_after_shorter_silence(self):
        self.feed(0.32, True)
        s = self.transcribe('Xin chào')
        self.turn.set_endpoint(s.turn_id, s.revision, True)
        self.feed(0.352, False)
        self.assertTrue(self.turn.should_finalize())

    def test_incomplete_sentence_waits_but_not_forever(self):
        self.feed(0.32, True)
        s = self.transcribe()
        self.turn.set_endpoint(s.turn_id, s.revision, False)
        self.feed(0.352, False)
        self.assertFalse(self.turn.should_finalize())
        self.feed(0.672, False)
        self.assertTrue(self.turn.should_finalize())

    def test_semantic_failure_uses_max_wait(self):
        self.feed(0.32, True)
        self.transcribe()
        self.feed(1.0, False)
        self.assertFalse(self.turn.should_finalize())
        self.feed(0.672, False)
        self.assertTrue(self.turn.should_finalize())

    def test_old_decision_cannot_end_resumed_speech(self):
        self.feed(0.32, True)
        s = self.transcribe()
        self.feed(0.32, True)
        self.assertFalse(self.turn.set_endpoint(s.turn_id, s.revision, True))
        self.feed(0.352, False)
        self.assertFalse(self.turn.should_finalize())  # final audio not transcribed yet

    def test_reset_rejects_previous_turn_results(self):
        self.feed(0.32, True)
        old = self.transcribe()
        self.turn.reset()
        self.feed(0.32, True)
        self.assertFalse(self.turn.set_text(old.turn_id, old.revision, 'stale'))
        self.assertFalse(self.turn.set_endpoint(old.turn_id, old.revision, True))

    def test_long_speech_has_bounded_buffer(self):
        self.feed(5.2, True)
        self.assertTrue(self.turn.at_limit)
        s = self.transcribe()
        self.assertLessEqual(len(s.audio), int(5.032 * 16000))
        self.assertTrue(self.turn.should_finalize())

    def test_config_rejects_invalid_timeouts(self):
        with self.assertRaises(ValueError):
            TurnConfig(min_silence=2, incomplete_silence=1)
