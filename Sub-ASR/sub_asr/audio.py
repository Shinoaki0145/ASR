"""Bounded microphone capture and paced 16 kHz WAV simulation."""
import asyncio
import time

import numpy as np
import soundfile as sf

from .turns import FRAME, RATE


async def microphone(device=None):
    import sounddevice as sd
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=100)
    errors = []

    def enqueue(frame, status):
        if status:
            errors.append(str(status))
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            errors.append('Audio queue overflow: input processing cannot keep up')

    def callback(indata, frames, timing, status):
        loop.call_soon_threadsafe(enqueue, indata[:, 0].copy(), status)

    with sd.InputStream(samplerate=RATE, blocksize=FRAME, channels=1, dtype='float32',
                        device=device, callback=callback):
        while True:
            frame = await queue.get()
            if errors:
                raise RuntimeError('; '.join(errors))
            yield frame


async def wav_frames(path, tail_seconds):
    wave, sr = sf.read(path, dtype='float32', always_2d=True)
    if sr != RATE:
        raise ValueError('File must be 16 kHz. Convert with ffmpeg -i input.wav -ac 1 -ar 16000 output.wav')
    mono = wave.mean(axis=1)
    # Pace playback in input time, including a silence tail to allow endpointing.
    deadline = time.monotonic()
    count = len(mono) + int(tail_seconds * RATE)
    for offset in range(0, count, FRAME):
        frame = np.zeros(FRAME, dtype=np.float32)
        segment = mono[offset:offset+FRAME]
        frame[:len(segment)] = segment
        yield frame
        deadline += FRAME / RATE
        await asyncio.sleep(max(0, deadline-time.monotonic()))
