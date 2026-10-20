"""Offline upload: one file is one turn; pauses never split ASR context."""
import argparse
import asyncio
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import tomllib

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .turns import RATE

MAX_BYTES = 25_000_000


def decode_audio(data, max_seconds=60):
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Audio upload must be nonempty and at most 25 MB')
    try:
        info = sf.info(io.BytesIO(data))
    except (sf.LibsndfileError, RuntimeError):
        if not shutil.which('ffmpeg'):
            raise ValueError('Unsupported audio codec. Upload WAV/FLAC or install ffmpeg') from None
        # Bound decoded output before allocating; reject overlong input, never truncate silently.
        process = subprocess.run(['ffmpeg','-v','error','-i','pipe:0','-t',str(max_seconds+0.1),
            '-ac','1','-ar',str(RATE),'-f','f32le','pipe:1'],input=data,capture_output=True,timeout=30)
        if process.returncode:
            raise ValueError('Cannot decode audio. Try a valid WAV/FLAC file')
        wave = np.frombuffer(process.stdout,dtype='<f4').copy()
        original_sr = original_channels = None
    else:
        if info.samplerate <= 0 or info.frames / info.samplerate > max_seconds:
            raise ValueError(f'Audio duration must be at most {max_seconds} seconds')
        samples, original_sr = sf.read(io.BytesIO(data),dtype='float32',always_2d=True)
        original_channels = samples.shape[1]
        if not np.isfinite(samples).all():
            raise ValueError('Audio has nonfinite samples')
        wave = samples.mean(axis=1)
        if original_sr != RATE:
            divisor = math.gcd(original_sr,RATE)
            wave = resample_poly(wave,RATE//divisor,original_sr//divisor)
    if not wave.size or not np.isfinite(wave).all():
        raise ValueError('Audio must contain finite samples')
    if len(wave) / RATE > max_seconds:
        raise ValueError(f'Audio duration must be at most {max_seconds} seconds')
    return np.ascontiguousarray(wave,dtype=np.float32), dict(
        original_sample_rate=original_sr,original_channels=original_channels,
        sample_rate=RATE,audio_seconds=len(wave)/RATE)


def transcribe_upload(wave, backend, detect_speech):
    start = time.perf_counter()
    segments = detect_speech(wave)
    vad_seconds = time.perf_counter()-start
    # VAD gates noise-only input. Keep the full clip if speech exists, including pauses.
    text, seconds = backend.recognize(wave) if segments else ('',0.0)
    return dict(transcript=text,speech_detected=bool(segments),speech_segments=segments,
                audio_seconds=len(wave)/RATE,vad_seconds=vad_seconds,asr_seconds=seconds,
                model=backend.metadata)


async def assistant_reply(text, history, settings):
    from .services import ChatService
    service = ChatService(os.getenv('LLM_BASE_URL',''),os.getenv('LLM_MODEL',''),
        api_key=os.getenv('LLM_API_KEY',''),response_timeout=settings.get('response_timeout',45))
    try:
        history = [*history, {'role':'user','content':text}]
        messages = [{'role':'system','content':settings['system_prompt']},*history]
        reply = ''.join([chunk async for chunk in service.reply(messages)])
        if not reply.strip():
            raise ValueError('LLM returned no streamed text; check server SSE support')
        history.append({'role':'assistant','content':reply})
        return reply,history[-20:]
    finally:
        await service.close()


def main():
    from dotenv import load_dotenv
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parent.parent/'config.toml')
    p.add_argument('--history',type=Path)
    p.add_argument('--reply',action='store_true')
    args = p.parse_args()
    wave, audio_metadata = decode_audio(args.input_file.read_bytes())
    base = args.config.resolve().parent
    cfg = tomllib.loads(args.config.read_text())
    load_dotenv(base/'.env')
    import torch
    from .asr import Gipformer
    model_cfg = cfg['asr']
    backend = Gipformer(base/model_cfg['repo'],base/model_cfg['icefall'],base/model_cfg['model_dir'],
                        'greedy_search',4)
    from silero_vad import load_silero_vad, get_speech_timestamps
    vad_model = load_silero_vad().eval()
    torch.set_num_threads(2)
    def detect(wave):
        return get_speech_timestamps(torch.from_numpy(wave),vad_model,sampling_rate=RATE,
                    return_seconds=True,min_silence_duration_ms=200,speech_pad_ms=100)
    report = transcribe_upload(wave,backend,detect)
    report['audio'] = audio_metadata
    report['reply'] = ''
    history = json.loads(args.history.read_text()) if args.history else []
    report['history'] = history
    if args.reply and report['transcript'].strip():
        try:
            report['reply'],report['history'] = asyncio.run(assistant_reply(
                report['transcript'],history,cfg['assistant']))
        except Exception as exc:
            # Preserve successful ASR output if the optional LLM is unavailable.
            report['reply_error'] = f'LLM failed ({type(exc).__name__}); check .env and server settings'
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
