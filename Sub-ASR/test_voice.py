import asyncio
import json
from pathlib import Path
import sys
import threading
from contextlib import suppress
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from voice import Decision, RequestState, parse_decision, apply_decision, Session, LLM, AudioRunner, speak_local, ResolveError


class StateTests(unittest.TestCase):
    def test_actions_and_schema(self):
        state = RequestState()
        for action, request in [('new','Đặt vé Đà Nẵng'),('append','Đặt vé Đà Nẵng ngày mai'),
                                ('update','Đặt vé Huế ngày mai'),('replace','Tìm khách sạn Hà Nội')]:
            state = apply_decision(state, 'nguyên văn', parse_decision(
                dict(action=action, request=request, question=None)))
            self.assertEqual(state.request, request)
        ambiguous = apply_decision(state,'Hà Nội',Decision('clarify',None,'Điểm đi hay đến?'))
        self.assertEqual(ambiguous.request,state.request)
        self.assertEqual(ambiguous.pending_text,'Hà Nội')
        self.assertIsNone(apply_decision(state,'Thôi',Decision('cancel',None,None)).request)
        for value in [dict(action='other',request='x',question=None),
                      dict(action=[],request='x',question=None),
                      dict(action='new',request='',question=None),
                      dict(action='cancel',request='null',question=None),
                      dict(action='clarify',request=None,question=None),
                      dict(action='new',request='x',question=None,extra=1)]:
            with self.assertRaises(ValueError): parse_decision(value)
        with self.assertRaises(ValueError): apply_decision(RequestState(),'x',Decision('append','x',None))
        with self.assertRaises(ValueError): apply_decision(state,'x',Decision('new','x',None))


class FakeLLM:
    def __init__(self):
        self.calls=[]
        self.hold=None
    async def resolve(self,text,state,history,last_reply):
        self.calls.append((text,state,history,last_reply))
        if self.hold and text=='old':
            try: await self.hold.wait()
            except asyncio.CancelledError: await self.hold.wait()
        if text=='hủy': return Decision('cancel',None,None)
        if text=='?': return Decision('clarify',None,'Bạn muốn đi từ đâu?')
        return Decision('new' if state.request is None else 'replace',text,None)
    async def reply(self,state,history):
        yield 'Phản hồi: '+state.request


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupt_and_stale_resolution(self):
        llm=FakeLLM(); llm.hold=asyncio.Event(); events=[]
        s=Session(llm,emit=events.append,simulated_seconds=0)
        s.speech_started(); s.submit('old'); await asyncio.sleep(0)
        old_task=s.response_task
        s.speech_started(); s.submit('new')
        await s.response_task
        llm.hold.set(); await old_task; await s.close()
        self.assertEqual(s.state.request,'new')
        self.assertTrue(any(e['event']=='stale_result_discarded' for e in events))
        self.assertFalse(any(m['role']=='assistant' for m in s.history))
        self.assertEqual(s.last_reply['text'],'Phản hồi: new')

    async def test_playback_cancel_does_not_block_capture_or_claim_heard(self):
        started=asyncio.Event(); release=asyncio.Event(); events=[]
        async def slow_speaker(text):
            started.set()
            try: await asyncio.Future()
            finally: await release.wait()
        s=Session(FakeLLM(),emit=events.append,speaker=slow_speaker)
        s.speech_started();s.submit('Đặt vé');await started.wait()
        s.speech_started()
        self.assertEqual(s.generation,2)
        self.assertEqual(s.state.request,'Đặt vé')
        self.assertEqual(s.last_reply['status'],'interrupted')
        self.assertFalse(any(m['role']=='assistant' for m in s.history))
        release.set(); await s.close()

    async def test_new_playback_waits_for_old_cleanup(self):
        started=asyncio.Event();release=asyncio.Event();active=[];overlap=[]
        async def speaker(text):
            if active: overlap.append(list(active))
            active.append(text);started.set()
            try:
                if 'old' in text: await asyncio.Future()
            finally:
                if 'old' in text: await release.wait()
                active.remove(text)
        s=Session(FakeLLM(),emit=lambda e:None,speaker=speaker)
        s.speech_started();s.submit('old');await started.wait()
        s.speech_started();s.submit('new');await asyncio.sleep(0);await asyncio.sleep(0)
        # Capture/resolve đã tiếp tục, nhưng không được phát câu mới trên loa khi câu cũ đang dọn.
        self.assertEqual(s.state.request,'new')
        release.set();await s.response_task;await s.close()
        self.assertEqual(overlap,[])
        self.assertNotEqual(s.llm.calls[-1][2][-1]['content'],'new')

    async def test_shutdown_order_and_client_closes_on_error(self):
        from voice import close_resources
        calls=[]
        class Resource:
            def __init__(self,name,fail=False): self.name=name;self.fail=fail
            async def close(self):
                calls.append(self.name)
                if self.fail: raise RuntimeError(self.name)
        with self.assertRaisesRegex(RuntimeError,'session'):
            await close_resources(Resource('session',True),Resource('asr'),Resource('http'))
        self.assertEqual(calls,['session','asr','http'])
        calls.clear()
        with self.assertRaisesRegex(RuntimeError,'asr'):
            await close_resources(Resource('session'),Resource('asr',True),Resource('http'))
        self.assertEqual(calls,['session','asr','http'])

    async def test_resolver_failure_is_distinct_from_valid_clarification(self):
        class Broken(FakeLLM):
            async def resolve(self,*args): raise ResolveError('http',400)
        events=[];s=Session(Broken(),emit=events.append,simulated_seconds=0)
        s.state=RequestState('đặt vé Đà Nẵng')
        s.speech_started();s.submit('ngày 30/10');await s.response_task
        self.assertEqual(s.state.request,'đặt vé Đà Nẵng')
        self.assertEqual(s.state.pending_text,'ngày 30/10')
        diagnostic=next(e for e in events if e['event']=='resolve_failed')
        self.assertEqual(diagnostic['reason'],'http')
        self.assertEqual(diagnostic['http_status'],400)
        await s.close()

    async def test_known_partial_request_then_date_does_not_require_completed_bot_question(self):
        class Chat(FakeLLM):
            async def resolve(self,text,state,history,last_reply):
                self.calls.append((text,state,history,last_reply))
                if state.request is None: return Decision('new','đặt vé tàu đi Đà Nẵng',None)
                return Decision('append','đặt vé tàu đi Đà Nẵng ngày 30/10',None)
        events=[];s=Session(Chat(),emit=events.append,simulated_seconds=4)
        s.speech_started();s.submit('đặt vé tàu đi Đà Nẵng')
        for _ in range(10):
            if any(e['event']=='playback_started' for e in events): break
            await asyncio.sleep(0)
        self.assertEqual(s.state.request,'đặt vé tàu đi Đà Nẵng')
        s.speech_started();s.submit('ừ vào ngày 30 tháng 10 nha')
        for _ in range(10):
            if s.state.request.endswith('30/10'): break
            await asyncio.sleep(0)
        self.assertEqual(s.state.request,'đặt vé tàu đi Đà Nẵng ngày 30/10')
        self.assertEqual([e['action'] for e in events if e['event']=='request_updated'],['new','append'])
        self.assertFalse(any(e['event']=='resolve_failed' for e in events))
        await s.close()

    async def test_empty_clarification_and_cancel(self):
        s=Session(FakeLLM(),emit=lambda e:None,simulated_seconds=0)
        s.speech_started();s.submit('a');await s.response_task
        s.speech_started();s.submit('');self.assertIsNone(s.response_task)
        self.assertEqual(s.state.request,'a')
        s.submit('?');await s.response_task
        self.assertEqual(s.state.pending_text,'?')
        s.speech_started();s.submit('hủy');await s.response_task
        self.assertIsNone(s.state.request)
        self.assertEqual(s.last_reply['text'],'Đã hủy yêu cầu.')
        await s.close()


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_resolution_request_contract(self):
        import httpx
        async def handler(req):
            payload=json.loads(req.content)
            self.assertEqual(payload['model'],'turn-small')
            self.assertFalse(payload['stream'])
            self.assertEqual(payload['temperature'],0)
            self.assertEqual(payload['max_tokens'],512)
            self.assertEqual(len(json.loads(payload['messages'][-1]['content'])['history']),6)
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(dict(
                action='append',request='Đặt vé Đà Nẵng ngày mai',question=None))}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm=LLM('http://test/v1','response-large',turn_model='turn-small',client=client)
            d=await llm.resolve('ngày mai',RequestState('Đặt vé Đà Nẵng'),
                                [dict(role='user',content=str(i)) for i in range(9)],None)
            self.assertEqual(d.action,'append')

    async def test_invalid_response_timeout_and_context(self):
        import httpx
        async def handler(req):
            p=json.loads(req.content)
            data=json.loads(p['messages'][-1]['content'])
            self.assertEqual(data['state']['pending_text'],'Hà Nội')
            self.assertIsNone(data['last_reply']['text'])
            return httpx.Response(200,json={'choices':[{'message':{'content':'{"action":"new"}'}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm=LLM('http://test/v1','model',client=client)
            with self.assertRaises(ResolveError) as error:
                await llm.resolve('x',RequestState(pending_text='Hà Nội'),[],
                                  dict(text='Câu chưa nghe',status='interrupted'))
            self.assertEqual(error.exception.reason,'invalid_decision')
        async def slow(req):
            await asyncio.sleep(1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
            llm=LLM('http://test/v1','model',client=client,resolve_timeout=.01)
            with self.assertRaises(ResolveError) as error:
                await llm.resolve('x',RequestState(),[],None)
            self.assertEqual(error.exception.reason,'timeout')

class AudioRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_asr_single_flight_and_stale_revision(self):
        from sub_asr.turns import TurnConfig
        import numpy as np
        class ASR:
            def __init__(self):
                self.entered=threading.Event();self.release=threading.Event();self.calls=0
            def recognize(self,audio):
                self.calls+=1;self.entered.set();self.release.wait(2)
                return ('kết quả cũ',.01)
        backend=ASR();events=[]
        s=Session(FakeLLM(),emit=events.append,simulated_seconds=0)
        runner=AudioRunner(s,backend,lambda frame:.9,TurnConfig(start_speech=.032,partial_interval=.01))
        try:
            await runner.consume(np.ones(512,dtype=np.float32))
            await asyncio.to_thread(backend.entered.wait,1)
            for _ in range(8): await runner.consume(np.ones(512,dtype=np.float32))
            self.assertEqual(backend.calls,1)
            backend.release.set()
            await runner.asr_task
            await runner.consume(np.ones(512,dtype=np.float32))
            self.assertEqual(runner.turn.text,'')
            self.assertTrue(any(e['event']=='stale_asr_discarded' for e in events))
        finally:
            backend.release.set();await runner.close();await s.close()

    async def test_two_speech_spans_update_same_request_after_interruption(self):
        from sub_asr.turns import TurnConfig
        import numpy as np
        class ASR:
            def recognize(self,audio):
                return ('đặt vé' if np.max(audio)>.85 else 'ngày mai',.001)
        class Chat(FakeLLM):
            async def endpoint(self,text,history): return True
            async def resolve(self,text,state,history,last_reply):
                return Decision('new' if state.request is None else 'append',
                                'đặt vé' if state.request is None else 'đặt vé ngày mai',None)
        events=[];s=Session(Chat(),emit=events.append,simulated_seconds=4)
        runner=AudioRunner(s,ASR(),lambda frame:float(frame[0]),TurnConfig())
        async def feed(n,value):
            for _ in range(n):
                await runner.consume(np.full(512,value,dtype=np.float32));await asyncio.sleep(.001)
        try:
            await feed(32,.9);await feed(94,0)
            self.assertEqual(s.state.request,'đặt vé')
            await feed(32,.8);await feed(94,0)
            # resolve_and_respond đã cập nhật trước khi playback giả lập hoàn tất.
            self.assertEqual(s.state.request,'đặt vé ngày mai')
            self.assertEqual([m['content'] for m in s.history if m['role']=='user'],['đặt vé','ngày mai'])
            self.assertTrue(any(e['event']=='cancel_requested' for e in events))
        finally: await runner.close();await s.close()

    async def test_one_second_silence_closes_without_semantic_llm(self):
        from sub_asr.turns import TurnConfig
        import numpy as np
        class ASR:
            def recognize(self,audio): return ('đặt vé',.001)
        class Chat(FakeLLM):
            async def endpoint(self,*args):
                raise AssertionError('Không được gọi LLM endpoint trong chế độ chốt theo im lặng')
        events=[];s=Session(Chat(),emit=events.append,simulated_seconds=0)
        runner=AudioRunner(s,ASR(),lambda frame:float(frame[0]),TurnConfig())
        try:
            for value,count in [(.9,32),(0,33)]:
                for _ in range(count):
                    await runner.consume(np.full(512,value,dtype=np.float32));await asyncio.sleep(.001)
            self.assertEqual(len([e for e in events if e['event']=='segment_closed']),1)
            self.assertEqual([m['content'] for m in s.history if m['role']=='user'],['đặt vé'])
        finally: await runner.close();await s.close()

    async def test_slow_asr_keeps_two_segments_separate_and_defers_old_response(self):
        from sub_asr.turns import TurnConfig
        import numpy as np
        class ASR:
            def __init__(self): self.release=threading.Event();self.entered=threading.Event();self.active=0;self.peak=0
            def recognize(self,audio):
                self.active+=1;self.peak=max(self.peak,self.active);self.entered.set()
                self.release.wait(3)
                self.active-=1
                return ('đặt vé' if np.max(audio)>.85 else 'ngày mai',.001)
        backend=ASR();events=[];s=Session(FakeLLM(),emit=events.append,simulated_seconds=0)
        runner=AudioRunner(s,backend,lambda frame:float(frame[0]),TurnConfig())
        try:
            for value,count in [(.9,32),(0,94),(.8,32),(0,33)]:
                for _ in range(count):
                    await runner.consume(np.full(512,value,dtype=np.float32));await asyncio.sleep(0)
            self.assertEqual(len([e for e in events if e['event']=='segment_closed']),2)
            backend.release.set()
            for _ in range(1000):
                await runner.consume(np.zeros(512,dtype=np.float32));await asyncio.sleep(.001)
                if not runner.busy: break
            if s.response_task: await s.response_task
            self.assertFalse(runner.busy)
            self.assertEqual(backend.peak,1)
            self.assertEqual([m['content'] for m in s.history if m['role']=='user'],['đặt vé','ngày mai'])
            self.assertEqual(len(s.llm.calls),1)
            self.assertEqual(s.llm.calls[0][0],'ngày mai')
            self.assertEqual(s.llm.calls[0][2][-1]['content'],'đặt vé')
            self.assertTrue(any(e['event']=='response_deferred' for e in events))
        finally: backend.release.set();await runner.close();await s.close()

    async def test_pause_under_one_second_keeps_same_audio_turn(self):
        from sub_asr.turns import TurnConfig
        import numpy as np
        class ASR:
            def recognize(self,audio): return ('câu đang nói',.001)
        events=[];s=Session(FakeLLM(),emit=events.append,simulated_seconds=0)
        runner=AudioRunner(s,ASR(),lambda frame:float(frame[0]),TurnConfig())
        try:
            for value,count in [(.9,32),(0,16),(.9,32)]:
                for _ in range(count):
                    await runner.consume(np.full(512,value,dtype=np.float32));await asyncio.sleep(.001)
            self.assertEqual(s.generation,1)
            self.assertFalse(any(e['event']=='segment_closed' for e in events))
        finally: await runner.close();await s.close()

    async def test_tts_kills_after_terminate_timeout(self):
        from unittest.mock import AsyncMock,patch
        class Process:
            returncode=None
            def terminate(self): self.terminated=True
            def kill(self): self.killed=True;self.returncode=-9
            async def communicate(self,data): await asyncio.Future()
            async def wait(self):
                if self.returncode is None: await asyncio.Future()
                return self.returncode
        process=Process()
        with patch('voice.shutil.which',return_value='espeak-ng'), patch(
                'voice.asyncio.create_subprocess_exec',AsyncMock(return_value=process)):
            task=asyncio.create_task(speak_local('Xin chào'))
            await asyncio.sleep(0);task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        self.assertTrue(process.terminated);self.assertTrue(process.killed)

if __name__=='__main__': unittest.main()
