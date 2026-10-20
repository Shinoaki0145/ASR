import argparse
import asyncio
import logging
import os
from pathlib import Path
import shutil
import tomllib

from dotenv import load_dotenv

from .turns import TurnConfig


def demo():
    """Deterministic simulation of two speech spans separated by a thinking pause."""
    import numpy as np
    from .turns import TurnBuffer, FRAME
    turn = TurnBuffer(TurnConfig())
    def feed(seconds, speech):
        for _ in range(round(seconds * 16000 / FRAME)):
            turn.feed(np.zeros(FRAME,dtype=np.float32), 0.9 if speech else 0.01)
    feed(1,True)
    first = turn.snapshot()
    turn.set_text(first.turn_id, first.revision, 'Tôi muốn đặt vé đi')
    turn.set_endpoint(first.turn_id, first.revision, False)
    feed(1.2,False)
    assert not turn.should_finalize()
    print('Tạm: Tôi muốn đặt vé đi → nghỉ 1,2 giây → giữ cùng lượt')
    feed(1,True)
    second = turn.snapshot()
    assert first.turn_id == second.turn_id
    turn.set_text(second.turn_id,second.revision,'Tôi muốn đặt vé đi Đà Nẵng vào sáng mai')
    turn.set_endpoint(second.turn_id,second.revision,True)
    feed(0.9,False)
    assert turn.should_finalize()
    print('Cuối:', turn.text)
    print('Demo dùng tín hiệu VAD, ASR và quyết định ngữ nghĩa giả lập; không chạy các model.')


async def run(args, config):
    from .app import Conversation
    from .audio import microphone, wav_frames
    from .asr import Gipformer
    from .services import ChatService
    try:
        import torch
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise RuntimeError('Install CUDA stack first: uv sync --extra cuda') from exc
    for value in config['asr'].values():
        if not (args.config.resolve().parent/value).exists():
            raise RuntimeError('Sources/model missing. Run: uv run python -m sub_asr.setup')
    base = args.config.resolve().parent
    cfg = config['asr']
    # Import Icefall before any neural inference: it owns the single inter-op setter.
    backend = Gipformer(base/cfg['repo'], base/cfg['icefall'], base/cfg['model_dir'], 'greedy_search', 4)
    vad_model = load_silero_vad().eval()
    torch.set_num_threads(2)
    def vad(frame):
        with torch.inference_mode():
            return float(vad_model(torch.from_numpy(frame),16000).item())
    chat = None
    settings = config['assistant']
    llm_model = os.getenv('LLM_MODEL','')
    if llm_model:
        chat = ChatService(os.environ.get('LLM_BASE_URL',''),llm_model,
                           api_key=os.getenv('LLM_API_KEY',''),turn_model=os.getenv('TURN_MODEL',''),
                           semantic_timeout=settings.get('semantic_timeout',1.5),
                           response_timeout=settings.get('response_timeout',45))
    elif not args.transcribe_only:
        raise RuntimeError('Set LLM_BASE_URL and LLM_MODEL in .env, or use --transcribe-only')
    else:
        logging.warning('No semantic model configured: use max_wait silence endpoint only')
    if not args.no_tts and not args.transcribe_only and not shutil.which(settings.get('tts_command','espeak-ng')):
        if chat:
            await chat.close()
        raise RuntimeError('Install espeak-ng, or pass --no-tts')
    turns = TurnConfig(**config['turn'])
    source = wav_frames(args.file, turns.max_wait + 1) if args.file else microphone(args.device)
    session = Conversation(backend,vad,turns,settings,chat=chat,tts=not args.no_tts,
                           transcribe_only=args.transcribe_only)
    print('Sẵn sàng. Ctrl+C để dừng. Dùng tai nghe: bản này chưa có AEC.', flush=True)
    await session.run(source)


def main():
    p = argparse.ArgumentParser(description='Silero VAD + Gipformer greedy voice assistant (no DF3)')
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parent.parent/'config.toml')
    p.add_argument('--file',type=Path,help='Paced mono/downmixed 16 kHz WAV instead of microphone')
    p.add_argument('--device',help='Microphone device ID or name')
    p.add_argument('--list-devices',action='store_true')
    p.add_argument('--no-tts',action='store_true')
    p.add_argument('--transcribe-only',action='store_true')
    p.add_argument('--demo',action='store_true',help='Offline deterministic logic demo, no models')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    # HTTP logs can contain endpoint URLs; application logs avoid transcript storage to disk.
    logging.getLogger('httpx').setLevel(logging.WARNING)
    if args.demo:
        demo()
        return
    if args.list_devices:
        try:
            import sounddevice as sd
        except OSError:
            p.exit(1, 'PortAudio unavailable. Install libportaudio2.\n')
        try:
            print(sd.query_devices())
        except sd.PortAudioError:
            p.exit(1, 'Cannot enumerate audio devices; check microphone access.\n')
        return
    load_dotenv(args.config.resolve().parent/'.env')
    config = tomllib.loads(args.config.read_text())
    TurnConfig(**config['turn'])
    if args.device and args.device.isdecimal():
        args.device = int(args.device)
    try:
        asyncio.run(run(args,config))
    except KeyboardInterrupt:
        print('\nĐã dừng.')
    except Exception as exc:
        p.exit(1,f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
