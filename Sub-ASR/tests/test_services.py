import asyncio
import json
import unittest
import httpx
from sub_asr.services import ChatService


class ServicesTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_validates_boolean_not_truthy_string(self):
        async def handler(request):
            self.assertEqual(request.url.path, '/v1/chat/completions')
            payload = json.loads(request.content)
            self.assertIn('Tôi muốn', payload['messages'][-1]['content'])
            return httpx.Response(200, json={'choices':[{'message':{'content':'{"complete":"false"}'}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = ChatService('http://test/v1','model',client=client)
            self.assertIsNone(await service.end_of_turn('Tôi muốn'))

    async def test_incomplete_sentence_is_not_complete(self):
        async def handler(request):
            return httpx.Response(200, json={'choices':[{'message':{'content':'{"complete":false}'}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = ChatService('http://test/v1','model',client=client)
            self.assertIs(await service.end_of_turn('Tôi muốn đi'), False)

    async def test_endpoint_failure_returns_unknown(self):
        async def handler(request):
            return httpx.Response(503)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = ChatService('http://test/v1','model',client=client)
            self.assertIsNone(await service.end_of_turn('Xin chào'))

    async def test_streamed_reply_and_context(self):
        async def handler(request):
            p = json.loads(request.content)
            self.assertEqual(p['messages'][-1], {'role':'user','content':'Xin chào'})
            self.assertTrue(p['stream'])
            body = 'data: '+json.dumps({'choices':[{'delta':{'content':'Chào bạn'}}]})+'\n\ndata: [DONE]\n\n'
            return httpx.Response(200, text=body)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = ChatService('http://test/v1','model',client=client)
            text = ''.join([part async for part in service.reply([{'role':'user','content':'Xin chào'}])])
            self.assertEqual(text,'Chào bạn')

    async def test_semantic_timeout_is_bounded(self):
        async def handler(request):
            await asyncio.sleep(1)
            raise AssertionError('request should have been cancelled')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = ChatService('http://test/v1','model',client=client,semantic_timeout=0.01)
            self.assertIsNone(await service.end_of_turn('Tôi muốn'))

    async def test_tts_uses_stdin_and_terminates_on_cancel(self):
        from unittest.mock import AsyncMock, patch
        from sub_asr.services import speak
        started = asyncio.Event()
        class Process:
            returncode = None
            terminated = False
            async def communicate(self, data):
                self.data = data
                started.set()
                await asyncio.sleep(10)
            def terminate(self):
                self.terminated = True
                self.returncode = -15
            async def wait(self):
                return self.returncode
        process = Process()
        spawn = AsyncMock(return_value=process)
        with patch('asyncio.create_subprocess_exec',spawn):
            task = asyncio.create_task(speak('Xin chào; $(không chạy lệnh)'))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(process.terminated)
        self.assertEqual(process.data.decode(),'Xin chào; $(không chạy lệnh)')
        self.assertEqual(spawn.call_args.args,('espeak-ng','-v','vi','-s','165','--stdin'))
