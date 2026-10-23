import asyncio
import time
import unittest
import numpy as np
from web_asr.backend import Controller
from web_asr.config import Settings


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.c = Controller(Settings(fake=True))
        self.sid = self.c.dispatch(None, "open")

    def tearDown(self):
        self.c.close()

    def snapshot(self):
        return self.c.dispatch(self.sid, "snapshot")

    def wait(self, predicate):
        limit = time.monotonic() + 5
        while time.monotonic() < limit:
            s = self.snapshot()
            if predicate(s):
                return s
            time.sleep(0.025)
        self.fail(str(s))

    def test_muted_text_history_and_context(self):
        self.c.dispatch(self.sid, "speaker", False)
        self.c.dispatch(self.sid, "text", "Đặt vé Đà Nẵng")
        self.wait(
            lambda s: (
                s["output"].startswith("[DEMO]")
                and any(e["event"] == "playback_stopped" for e in s["events"])
            )
        )
        self.c.dispatch(self.sid, "text", "Ngày mai")
        s = self.wait(lambda s: "Ngày mai" in (s["request"] or ""))
        self.assertIn("Đà Nẵng", s["request"])
        c = self.c.clients[self.sid]
        self.assertTrue(any(x["role"] == "assistant" for x in c.session.history))
        self.assertEqual(c.session.last_reply["delivery"], "browser")

    def test_audio_and_two_texts_one_group(self):
        self.c.dispatch(self.sid, "speaker", False)
        self.c.dispatch(self.sid, "mic", True)
        self.c.dispatch(self.sid, "audio", (16000, np.ones(8192, np.float32) * 0.1))
        self.wait(lambda s: bool(s["groups"]))
        self.c.dispatch(self.sid, "text", "ngày 30/10")
        self.c.dispatch(self.sid, "text", "2 vé")
        self.assertIsNone(self.snapshot()["request"])
        self.c.dispatch(self.sid, "mic", False)
        s = self.wait(lambda s: s["request"] is not None)
        self.assertIn("âm thanh demo", s["request"])
        self.assertIn("ngày 30/10", s["request"])
        self.assertIn("2 vé", s["request"])
        self.assertEqual(sum(e["event"] == "transcript_final" for e in s["events"]), 1)

    def test_browser_ended_only_completion_and_stale_ack(self):
        self.c.dispatch(self.sid, "text", "Hello")
        s = self.wait(lambda s: s["command"] and s["command"]["kind"] == "play")
        cmd = s["command"]
        c = self.c.clients[self.sid]
        self.c.dispatch(self.sid, "ack", [{**cmd, "command_id": -1, "status": "ended"}])
        self.assertNotEqual(c.session.last_reply["status"], "completed")
        self.c.dispatch(self.sid, "ack", [{**cmd, "status": "ended"}])
        self.wait(lambda s: any(e["event"] == "playback_stopped" for e in s["events"]))
        self.assertEqual(c.session.last_reply["status"], "completed")

    def test_speaker_does_not_change_generation_or_replay(self):
        self.c.dispatch(self.sid, "text", "Hello")
        self.wait(lambda s: s["command"] and s["command"]["kind"] == "play")
        before = self.snapshot()
        self.c.dispatch(self.sid, "speaker", False)
        self.c.dispatch(self.sid, "speaker", True)
        after = self.snapshot()
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(after["command"]["kind"], "stop")
        self.assertGreater(after["audio_epoch"], before["audio_epoch"])

    def test_session_isolation_and_mic_limit(self):
        other = self.c.dispatch(None, "open")
        self.c.dispatch(self.sid, "mic", True)
        with self.assertRaises(ValueError):
            self.c.dispatch(other, "mic", True)
        self.c.dispatch(self.sid, "text", "hello")
        self.assertIsNone(self.c.dispatch(other, "snapshot")["request"])
        with self.assertRaises(ValueError):
            self.c.dispatch(None, "open")

    def test_asr_failure_preserves_typed_input(self):
        def fail(audio):
            raise RuntimeError("native ASR failed")

        self.c.pool.recognize = fail
        self.c.dispatch(self.sid, "speaker", False)
        self.c.dispatch(self.sid, "mic", True)
        self.c.dispatch(self.sid, "audio", (16000, np.ones(8192, np.float32) * 0.1))
        self.wait(lambda s: bool(s["groups"]))
        self.c.dispatch(self.sid, "text", "ngày mai 2 vé")
        self.c.dispatch(self.sid, "mic", False)
        s = self.wait(lambda s: s["request"] is not None)
        self.assertIn("ngày mai 2 vé", s["request"])
        self.assertTrue(any(e["event"] == "asr_failed" for e in s["events"]))

    def test_packet_order_and_rejected_mic(self):
        other = self.c.dispatch(None, "open")
        self.c.dispatch(other, "mic", True)

        def packet(seq, **kw):
            return dict(
                sequence=seq,
                recording_id=1,
                mic=True,
                rate=16000,
                audio="",
                acks=[],
                **kw,
            )

        self.c.dispatch(self.sid, "packet", packet(2, text="second"))
        self.assertIsNone(self.snapshot()["request"])
        self.c.dispatch(self.sid, "packet", packet(1, text="first"))
        self.wait(lambda s: s["request"] is not None)
        s = self.snapshot()
        self.assertFalse(s["mic"])
        self.assertTrue(s["capture_error"])
        users = [
            m["content"]
            for m in self.c.clients[self.sid].session.history
            if m["role"] == "user"
        ]
        self.assertEqual(users, ["first", "second"])

    def test_audio_overflow_keeps_control_text(self):
        import base64

        payload = base64.b64encode(np.ones(64000, dtype="<i2").tobytes()).decode()
        self.c.dispatch(
            self.sid,
            "packet",
            dict(
                sequence=1,
                recording_id=1,
                mic=True,
                rate=16000,
                audio=payload,
                text="giữ text",
                acks=[],
            ),
        )
        s = self.wait(lambda s: s["request"] is not None)
        self.assertIn("giữ text", s["request"])
        self.assertTrue(s["capture_error"])
        self.assertFalse(s["mic"])

    def test_native_recognize_serialized(self):
        import threading
        from web_asr.backend import ModelPool

        pool = ModelPool()
        active = 0
        peak = 0

        class Native:
            def recognize(self, wave):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                time.sleep(0.03)
                active -= 1
                return "x", 0.03

        pool.backend = Native()
        threads = [
            threading.Thread(target=pool.recognize, args=(np.zeros(512),))
            for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(peak, 1)

    def test_tts_exit_race_preserves_cancellation(self):
        import asyncio
        from unittest.mock import AsyncMock, Mock, patch

        process = Mock(returncode=None)
        process.communicate = AsyncMock(side_effect=asyncio.CancelledError)
        process.terminate = Mock(side_effect=ProcessLookupError)
        process.wait = AsyncMock(return_value=0)

        async def run():
            self.c.settings.fake = False
            self.c.settings.tts_voice = "espeak"
            with patch(
                "web_asr.backend.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await self.c.clients[self.sid].synthesize("xin chào")

        self.c.call(run())

    def test_edge_tts_synthesis_and_markdown_cleanup(self):
        from unittest.mock import patch

        captured = {}

        class FakeComm:
            def __init__(self, text, voice, rate):
                captured["text"] = text
                captured["voice"] = voice
                captured["rate"] = rate

            async def stream(self):
                yield {"type": "audio", "data": b"ID3-mp3-bytes"}

        async def run():
            self.c.settings.fake = False
            self.c.settings.tts_voice = "vi-VN-HoaiMyNeural"
            with patch("edge_tts.Communicate", FakeComm):
                data, mime, engine = await self.c.clients[self.sid].synthesize(
                    "Chào **bạn**, đặt [vé](https://x) #1"
                )
            self.assertEqual(mime, "audio/mpeg")
            self.assertEqual(data, b"ID3-mp3-bytes")
            self.assertIn("edge-tts", engine)
            self.assertEqual(captured["text"], "Chào bạn, đặt vé 1")
            self.assertEqual(captured["voice"], "vi-VN-HoaiMyNeural")

        self.c.call(run())

    def test_upload_audio_file_streams_through_vad_and_asr(self):
        import tempfile
        import soundfile as sf

        self.c.dispatch(self.sid, "speaker", False)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            speech = np.ones(16000, dtype=np.float32) * 0.2
            sf.write(f.name, speech, 16000)
            self.c.dispatch(self.sid, "audio_file", f.name, False)
            s = self.wait(lambda s: s["request"] is not None)
        self.assertIn("âm thanh demo", s["request"])
        self.assertTrue(any(e["event"] == "file_stream_started" for e in s["events"]))
        self.assertTrue(any(e["event"] == "file_stream_finished" for e in s["events"]))
