import asyncio
from contextlib import redirect_stdout
import io
import unittest
import numpy as np

from sub_asr.app import Conversation
from sub_asr.turns import TurnConfig


class FakeASR:
    def recognize(self, wave):
        return ('Tôi muốn đi Đà Nẵng' if np.count_nonzero(wave) > 7000 else 'Tôi muốn đi', 0.001)


class FakeChat:
    def __init__(self):
        self.messages = []
        self.interrupted = asyncio.Event()
        self.response_started = asyncio.Event()
        self.hold_response = False

    async def end_of_turn(self, text, history):
        return text.endswith('Đà Nẵng')

    async def reply(self, messages):
        self.messages.append(messages)
        self.response_started.set()
        try:
            if self.hold_response:
                await asyncio.sleep(10)
            yield 'Bạn muốn đi khi nào?'
        finally:
            self.interrupted.set()

    async def close(self):
        pass


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    def make_session(self, chat=None):
        return Conversation(FakeASR(),lambda frame: float(frame[0]),
            TurnConfig(start_speech=0.096,min_silence=0.32,incomplete_silence=0.96,
                       max_wait=1.6,partial_interval=0.1),
            {'system_prompt':'Trợ lý','max_history_messages':20},chat=chat,tts=False)

    async def feed(self, session, seconds, speech):
        for _ in range(round(seconds/0.032)):
            await session.consume(np.full(512,0.9 if speech else 0,dtype=np.float32))
            await asyncio.sleep(0.001)

    async def test_thinking_pause_yields_one_complete_user_message(self):
        chat = FakeChat()
        session = self.make_session(chat)
        with redirect_stdout(io.StringIO()):
            await self.feed(session,0.32,True)
            first_id = session.turn.turn_id
            await self.feed(session,0.6,False)
            self.assertFalse(chat.messages)
            await self.feed(session,0.32,True)
            self.assertEqual(session.turn.turn_id,first_id)
            await self.feed(session,0.6,False)
            if session.response_task:
                await session.response_task
        users = [m for m in session.history if m['role']=='user']
        self.assertEqual(users,[{'role':'user','content':'Tôi muốn đi Đà Nẵng'}])
        self.assertEqual(len(chat.messages),1)

    async def test_noise_only_does_not_call_asr_or_llm(self):
        session = self.make_session(FakeChat())
        with redirect_stdout(io.StringIO()):
            await self.feed(session,2,False)
        self.assertIsNone(session.asr_task)
        self.assertFalse(session.history)
        self.assertFalse(session.chat.messages)

    async def test_new_speech_cancels_pending_response(self):
        chat = FakeChat()
        chat.hold_response = True
        session = self.make_session(chat)
        session.history = [{'role':'user','content':'Xin chào'}]
        with redirect_stdout(io.StringIO()):
            session.response_task = asyncio.create_task(session.respond())
            await chat.response_started.wait()
            await self.feed(session,0.16,True)
            self.assertTrue(chat.interrupted.is_set())
            self.assertIsNone(session.response_task)
            self.assertEqual(session.history,[{'role':'user','content':'Xin chào'}])
            await session.cancel(session.asr_task)
