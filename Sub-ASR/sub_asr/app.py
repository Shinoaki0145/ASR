"""Capture never waits for ASR, HTTP requests, or TTS playback."""
import asyncio
from contextlib import suppress
import logging

from .services import speak
from .turns import TurnBuffer

log = logging.getLogger(__name__)


class Conversation:
    def __init__(self, backend, vad, turn_config, assistant_config, chat=None, tts=True,
                 transcribe_only=False):
        self.backend, self.vad, self.chat = backend, vad, chat
        self.turn = TurnBuffer(turn_config)
        self.config = assistant_config
        self.tts, self.transcribe_only = tts, transcribe_only
        self.history = []
        self.asr_task = self.semantic_task = self.response_task = None
        self.asr_snapshot = None
        self.semantic_key = self.last_semantic_key = None
        self.last_final_text = ''
        self.last_final_clock = None

    async def cancel(self, task):
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def respond(self):
        response = ''
        playback_started = False
        try:
            print('Trợ lý: ', end='', flush=True)
            messages = [{'role':'system', 'content':self.config['system_prompt']}, *self.history]
            async for chunk in self.chat.reply(messages):
                response += chunk
                print(chunk, end='', flush=True)
            print()
            if response and self.tts:
                playback_started = True
                await speak(response, command=self.config.get('tts_command','espeak-ng'),
                            voice=self.config.get('tts_voice','vi'), speed=self.config.get('tts_speed',165))
            if response:
                self.history.append({'role':'assistant','content':response})
        except asyncio.CancelledError:
            print('\n[Người dùng ngắt lời]', flush=True)
            # Do not add a generated but unspoken response to conversation context.
            if playback_started and response:
                self.history.append({'role':'assistant', 'content':response+' [Phản hồi bị người dùng ngắt.]'})
            raise
        except Exception as exc:
            log.error('Assistant response failed: %s', type(exc).__name__)

    async def consume(self, frame):
        probability = self.vad(frame)
        started = self.turn.feed(frame, probability)
        if started:
            # Speech must persist for start_speech, rather than a single noisy frame.
            await self.cancel(self.response_task)
            self.response_task = None
            print(f'\n[Lượt {self.turn.turn_id}: đang nghe]', flush=True)

        if self.semantic_task and self.semantic_key != (self.turn.turn_id, self.turn.revision):
            await self.cancel(self.semantic_task)
            self.semantic_task = None

        if self.asr_task and self.asr_task.done():
            text, seconds = self.asr_task.result()
            s = self.asr_snapshot
            self.asr_task = None
            if self.turn.set_text(s.turn_id, s.revision, text):
                print(f'Tạm: {self.turn.text}  [ASR {seconds*1000:.0f} ms]', flush=True)

        if self.semantic_task and self.semantic_task.done():
            decision = self.semantic_task.result()
            self.turn.set_endpoint(*self.semantic_key, decision)
            print(f'[Kết thúc lượt: {decision if decision is not None else "chưa biết; chờ dự phòng"}]', flush=True)
            self.semantic_task = None

        if self.turn.needs_asr() and self.asr_task is None:
            self.asr_snapshot = self.turn.snapshot()
            self.asr_task = asyncio.create_task(asyncio.to_thread(self.backend.recognize, self.asr_snapshot.audio))

        key = (self.turn.turn_id, self.turn.revision)
        if (self.chat and self.turn.active and self.turn.text and
                self.turn.text_revision == self.turn.revision and self.semantic_task is None and
                self.last_semantic_key != key and self.turn.silence >= self.turn.config.min_silence / 2):
            self.semantic_key = self.last_semantic_key = key
            self.semantic_task = asyncio.create_task(self.chat.end_of_turn(self.turn.text, self.history))

        if self.turn.should_finalize():
            text = self.turn.text
            self.last_final_text, self.last_final_clock = text, self.turn.clock
            print(f'Cuối: {text or "[không nhận ra lời nói]"}', flush=True)
            if self.turn.at_limit:
                log.warning('max_turn reached; force endpoint (may split a long utterance)')
            await self.cancel(self.semantic_task)
            self.semantic_task = None
            self.turn.reset()
            if text:
                self.history.append({'role':'user','content':text})
                # Trim whole exchanges rather than starting history with an assistant reply.
                limit = self.config.get('max_history_messages',20)
                while len(self.history) > limit:
                    self.history.pop(0)
                    while self.history and self.history[0]['role'] != 'user':
                        self.history.pop(0)
                if self.chat and not self.transcribe_only:
                    await self.cancel(self.response_task)
                    self.response_task = asyncio.create_task(self.respond())

    async def run(self, source):
        try:
            async for frame in source:
                await self.consume(frame)
            # File mode includes an audio silence tail; drain an outstanding response at EOF.
            if self.turn.active:
                log.warning('Input ended before finalization; no forced partial transcript was sent')
            if self.response_task:
                await self.response_task
        finally:
            for task in (self.semantic_task, self.response_task, self.asr_task):
                await self.cancel(task)
            if self.chat:
                await self.chat.close()
