"""One background event loop; browser callbacks never wait for model inference."""

import asyncio
from contextlib import suppress
import copy
from collections import deque
import re
import sys
import threading
import time
import tomllib
import uuid
import base64
import io
import wave
import numpy as np
from .config import ROOT
from .audio import AudioIngress

sys.path.insert(0, str(ROOT / "Sub-ASR"))
from voice import Session, AudioRunner, LLM, Decision
from sub_asr.turns import TurnConfig


class FakeLLM:
    async def resolve(self, text, state, history, last_reply):
        await asyncio.sleep(0.03)
        return Decision(
            "append" if state.request else "new",
            ((state.request + "; ") if state.request else "") + text,
            None,
        )

    async def reply(self, state, history):
        for word in ("[DEMO] Đã nhận: " + state.request).split():
            yield word + " "
            await asyncio.sleep(0.015)

    async def close(self):
        pass


class ModelPool:
    def __init__(self, fake=False):
        self.fake = fake
        self.lock = threading.Lock()
        self.backend = None

    def load(self):
        with self.lock:
            if self.fake or self.backend:
                return
            from sub_asr.asr import Gipformer

            folder = ROOT / "Sub-ASR"
            cfg = tomllib.loads((folder / "config.toml").read_text())["asr"]
            self.backend = Gipformer(
                folder / cfg["repo"],
                folder / cfg["icefall"],
                folder / cfg["model_dir"],
                "greedy_search",
                4,
            )

    def recognize(self, audio):
        with self.lock:
            if self.fake:
                return ("âm thanh demo", 0.01)
            return self.backend.recognize(audio)

    def new_vad(self):
        if self.fake:
            return lambda x: float(np.max(np.abs(x)) > 0.015)
        import torch
        from silero_vad import load_silero_vad

        model = load_silero_vad().to("cuda").eval()

        def vad(frame):
            with torch.inference_mode():
                return model(torch.from_numpy(frame).to("cuda"), 16000).item()

        return vad


class WebSession(Session):
    recover_asr_error = True

    def __init__(self, owner, llm):
        self.owner = owner
        self.groups = {}
        self.current = None
        super().__init__(
            llm,
            emit=owner.event,
            speaker=owner.speak,
            speaker_delivery="browser",
            record_text=True,
        )

    def speech_started(self):
        super().speech_started()
        self.owner.stop_audio("interrupted")
        tid = self.owner.runner.turn.turn_id
        self.current = tid
        self.groups[tid] = {
            "typed": [],
            "generation": self.generation,
            "id": uuid.uuid4().hex[:8],
        }
        self.emit("input_group", group=self.groups[tid]["id"], turn_id=tid)

    def typed(self, text):
        if not text.strip():
            return
        # A pending frozen segment still owns its typed additions until final ASR submit.
        g = self.groups.get(self.current)
        if g:
            Session.speech_started(self)
            self.owner.stop_audio("interrupted")
            g["typed"].append(text.strip())
            g["generation"] = self.generation
            self.emit(
                "text_appended", group=g["id"], revision=len(g["typed"]), text=text
            )
        else:
            Session.speech_started(self)
            self.owner.stop_audio("interrupted")
            super().submit(text)

    def submit(self, text, utterance_generation=None, turn_id=None):
        g = self.groups.pop(turn_id, None)
        if g:
            text = "\n".join(
                ([text] if text.strip() else [])
                + [f"[Bổ sung bằng text] {x}" for x in g["typed"]]
            )
            utterance_generation = g["generation"]
            if self.current == turn_id:
                self.current = None
        super().submit(text, utterance_generation, turn_id)


class Client:
    def __init__(self, controller, vad):
        self.controller = controller
        self.events = deque(maxlen=200)
        self.output = ""
        self.output_generation = -1
        self.mic = False
        self.speaker = True
        self.epoch = 0
        self.command_id = 0
        self.command = None
        self.waiter = None
        self.tts = None
        self.packet_sequence = 1
        self.packets = {}
        self.last_seen = time.monotonic()
        self.closed = False
        self.ingress = AudioIngress()
        self.queue = asyncio.Queue()
        self.audio_queued = 0
        self.capture_error = None
        self.recording_id = None
        self.llm = (
            FakeLLM()
            if controller.settings.fake
            else LLM(
                controller.settings.base_url,
                controller.settings.model,
                controller.settings.api_key,
                resolve_timeout=controller.settings.timeout,
                turn_model=controller.settings.turn_model,
            )
        )
        self.session = WebSession(self, self.llm)
        self.runner = AudioRunner(
            self.session,
            controller.pool,
            vad,
            TurnConfig(),
            controller.settings.silence,
        )
        self.worker = asyncio.create_task(self.consume())

    def event(self, event):
        self.events.append(event)
        if event["event"] == "speech_started":
            self.output = ""
            self.output_generation = event["generation"]
        if event["event"] == "reply_chunk":
            self.output += event["text"]
        if event["event"] == "reply_final":
            self.output = event["text"]

    def command_for(self, kind, **extra):
        self.command_id += 1
        self.command = dict(
            command_id=self.command_id,
            generation=self.session.generation,
            audio_epoch=self.epoch,
            playback_id=uuid.uuid4().hex,
            kind=kind,
            **extra,
        )
        self.event(
            dict(
                event="audio_command",
                generation=self.session.generation,
                kind=kind,
                command_id=self.command_id,
            )
        )
        return self.command

    def stop_audio(self, status):
        self.command_for("stop")
        if self.waiter and not self.waiter.done():
            self.waiter.set_result(status)
        if self.tts and not self.tts.done():
            self.tts.cancel()

    @staticmethod
    def clean_tts_text(text):
        cleaned = re.sub(r"[*_`#~>]+", "", text)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        return cleaned.strip() or text

    async def synthesize_espeak(self, text):
        p = await asyncio.create_subprocess_exec(
            "espeak-ng",
            "-v",
            "vi",
            "-s",
            "165",
            "--stdout",
            "--stdin",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            data, _ = await asyncio.wait_for(p.communicate(text.encode()), 20)
            if p.returncode:
                raise RuntimeError("TTS failed")
            return data, "audio/wav"
        finally:
            if p.returncode is None:
                with suppress(ProcessLookupError):
                    p.terminate()
                try:
                    await asyncio.wait_for(p.wait(), 1)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        p.kill()
                    await p.wait()

    async def synthesize_edge(self, text):
        import edge_tts

        comm = edge_tts.Communicate(
            self.clean_tts_text(text),
            voice=self.controller.settings.tts_voice,
            rate=self.controller.settings.tts_rate,
        )
        chunks = bytearray()
        async with asyncio.timeout(20):
            async for chunk in comm.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    chunks.extend(chunk["data"])
        if not chunks:
            raise RuntimeError("Edge TTS returned empty audio")
        return bytes(chunks), "audio/mpeg"

    async def synthesize(self, text):
        if self.controller.settings.fake:
            x = (np.sin(np.arange(48000) * 2 * np.pi * 440 / 16000) * 3000).astype(
                "<i2"
            )
            b = io.BytesIO()
            with wave.open(b, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(x.tobytes())
            return b.getvalue(), "audio/wav"
        if self.controller.settings.tts_voice.lower() != "espeak":
            try:
                return await self.synthesize_edge(text)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        return await self.synthesize_espeak(self.clean_tts_text(text))

    async def speak(self, text):
        if not self.speaker:
            return "muted"
        epoch = self.epoch
        generation = self.session.generation
        self.tts = asyncio.create_task(self.synthesize(text))
        try:
            result = await self.tts
            data, mime = (
                result if isinstance(result, tuple) else (result, "audio/wav")
            )
        except asyncio.CancelledError:
            if epoch != self.epoch:
                return "muted"
            raise
        finally:
            self.tts = None
        if not self.speaker or epoch != self.epoch:
            return "muted"
        self.waiter = asyncio.get_running_loop().create_future()
        self.command_for(
            "play",
            audio_url=f"data:{mime};base64," + base64.b64encode(data).decode(),
        )
        try:
            bps = 6000 if mime == "audio/mpeg" else 32000
            return await asyncio.wait_for(self.waiter, max(len(data) / bps, len(text) / 10) + 15)
        except TimeoutError:
            self.event(dict(event="audio_timeout", generation=generation))
            self.stop_audio("error")
            return "error"
        finally:
            if generation != self.session.generation:
                self.stop_audio("interrupted")
            self.waiter = None

    def ack(self, ack):
        if not isinstance(ack, dict) or not self.command:
            return
        if any(
            ack.get(k) != self.command.get(k)
            for k in ("command_id", "generation", "audio_epoch", "playback_id")
        ):
            return
        self.event(
            dict(
                event="browser_ack",
                generation=self.session.generation,
                status=ack.get("status"),
            )
        )
        status = ack.get("status")
        if status in {"started", "blocked", "ended", "error"}:
            self.command.pop("audio_url", None)
        if (
            self.waiter
            and not self.waiter.done()
            and status in {"ended", "error", "stopped", "muted"}
        ):
            self.waiter.set_result("completed" if status == "ended" else status)

    async def consume(self):
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(self.queue.get(), 0.032)
                except TimeoutError:
                    if not self.runner.busy:
                        continue
                    frame = None if self.mic else np.zeros(512, np.float32)
                if not isinstance(frame, str):
                    self.audio_queued = max(0, self.audio_queued - 1)
                if isinstance(frame, str):
                    self.session.typed(frame)
                else:
                    await self.runner.consume(frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.mic = False
            self.capture_error = (
                f"Audio pipeline lỗi: {type(e).__name__}. Tải lại trang."
            )
            self.event(
                dict(
                    event="audio_error",
                    generation=self.session.generation,
                    error=type(e).__name__,
                )
            )

    def enqueue(self, frames):
        if self.audio_queued + len(frames) > 96:
            self.mic = False
            self.capture_error = "Audio queue đầy: tắt/bật mic để thử lại"
            self.event(dict(event="audio_overflow", generation=self.session.generation))
            return
        for frame in frames:
            self.queue.put_nowait(frame)
            self.audio_queued += 1

    async def close(self):
        self.closed = True
        self.mic = False
        self.stop_audio("interrupted")
        self.worker.cancel()
        await asyncio.gather(self.worker, return_exceptions=True)
        try:
            await self.session.close()
        finally:
            try:
                await self.runner.close()
            finally:
                await self.llm.close()


class Controller:
    def __init__(self, settings, pool=None):
        self.settings = settings
        self.pool = pool or ModelPool(settings.fake)
        self.pool.load()
        self.clients = {}
        self.opening = 0
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.call(self.reaper_start())

    def call(self, coro, timeout=10):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=timeout)

    async def reaper_start(self):
        self.reaper = asyncio.create_task(self.expire())

    async def expire(self):
        while True:
            await asyncio.sleep(10)
            for sid, c in list(self.clients.items()):
                if time.monotonic() - c.last_seen > 120:
                    try:
                        await c.close()
                    except Exception:
                        pass
                    self.clients.pop(sid, None)

    def dispatch(self, sid, action, *args):
        async def run():
            if action == "open":
                if len(self.clients) + self.opening >= 2:
                    raise ValueError("Đã đủ 2 phiên, đóng tab cũ hoặc đợi 120s")
                self.opening += 1
                try:
                    vad = await asyncio.to_thread(self.pool.new_vad)
                    key = uuid.uuid4().hex
                    self.clients[key] = Client(self, vad)
                    return key
                finally:
                    self.opening -= 1
            c = self.clients[sid]
            c.last_seen = time.monotonic()
            if action == "packet":
                packet = args[0]
                seq = packet["sequence"]
                if seq < c.packet_sequence:
                    return
                if seq > c.packet_sequence + 32:
                    raise ValueError(
                        "Audio transport mất thứ tự quá lâu, tải lại trang"
                    )
                c.packets[seq] = packet
                while c.packet_sequence in c.packets:
                    p = c.packets.pop(c.packet_sequence)
                    c.packet_sequence += 1
                    if p["mic"] and p.get("recording_id") != c.recording_id:
                        c.recording_id = p.get("recording_id")
                        c.capture_error = None
                        if any(x.mic for x in self.clients.values() if x is not c):
                            c.capture_error = "Một phiên khác đang dùng mic"
                        else:
                            c.ingress = AudioIngress()
                            c.mic = True
                    if p.get("audio") and c.mic:
                        x = np.frombuffer(base64.b64decode(p["audio"]), dtype="<i2")
                        c.enqueue(c.ingress.push(p["rate"], x))
                    if p.get("text"):
                        c.queue.put_nowait(p["text"])
                    for ack in p.get("acks", []):
                        c.ack(ack)
                    if not p["mic"] and c.mic:
                        c.enqueue(c.ingress.finish())
                        c.mic = False
            elif action == "text":
                c.queue.put_nowait(args[0])
            elif action == "mic":
                enabled = args[0]
                if enabled and any(x.mic for x in self.clients.values() if x is not c):
                    raise ValueError("Một phiên khác đang dùng mic")
                if enabled and not c.mic:
                    c.ingress = AudioIngress()
                if not enabled and c.mic:
                    c.enqueue(c.ingress.finish())
                c.mic = enabled
            elif action == "audio":
                if c.mic and args[0] is not None:
                    c.enqueue(c.ingress.push(*args[0]))
            elif action == "speaker":
                c.speaker = args[0]
                c.epoch += 1
                c.stop_audio("muted" if not c.speaker else "stopped")
            elif action == "ack":
                for ack in args[0]:
                    c.ack(ack)
            elif action == "snapshot":
                return dict(
                    output=c.output,
                    request=c.session.state.request,
                    generation=c.session.generation,
                    mic=c.mic,
                    capture_error=c.capture_error,
                    speaker=c.speaker,
                    audio_epoch=c.epoch,
                    command=c.command,
                    events=list(c.events),
                    groups={
                        str(k): copy.deepcopy(v) for k, v in c.session.groups.items()
                    },
                    queue=c.queue.qsize(),
                    asr_busy=self.runner_busy(c),
                    reply_status=c.session.last_reply,
                    mode="DEMO — không phải nhận dạng thật"
                    if self.settings.fake
                    else "REAL",
                )
            elif action == "close":
                await c.close()
                self.clients.pop(sid, None)

        return self.call(run(), timeout=60 if action == "open" else 10)

    @staticmethod
    def runner_busy(client):
        return client.runner.asr_task is not None

    def close(self):
        async def cleanup():
            self.reaper.cancel()
            await asyncio.gather(self.reaper, return_exceptions=True)
            await asyncio.gather(
                *(c.close() for c in list(self.clients.values())),
                return_exceptions=True,
            )
            self.clients.clear()

        self.call(cleanup())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()
