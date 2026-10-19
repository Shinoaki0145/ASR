
"""Vietnamese speech benchmark with continuous and intermittent office noise; also embedded in the Colab notebook."""
import argparse
import csv
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import tarfile
import time
import unicodedata
import urllib.request
import urllib.error
import zipfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FLEURS_REVISION = '70bb2e84b976b7e960aa89f1c648e09c59f894dd'
FLEURS_BASE = f'https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REVISION}/data/vi_vn'
DCASE_URL = 'https://archive.org/download/dcase2016_task2_train_dev/dcase2016_task2_train_dev.zip'
DCASE_MD5 = '4e1b5e8887159193e8624dec801eb9e7'
# Storage hosts returned by archive.org/metadata/dcase2016_task2_train_dev.
DCASE_MIRRORS = [
    f'https://{host}/17/items/dcase2016_task2_train_dev/dcase2016_task2_train_dev.zip'
    for host in ('ia801900.us.archive.org', 'ia601900.us.archive.org')
]


NOISE_PREFIXES = {'keyboard': 'keyboard', 'knock': 'knock', 'doorslam': 'doorslam',
                  'keys': 'keysDrop', 'pageturn': 'pageturn', 'phone': 'phone'}
NOISE_CONDITIONS = [('keyboard', 'continuous')] + [(name, 'events') for name in NOISE_PREFIXES]


def normalize_text(text):
    text = unicodedata.normalize('NFC', text).casefold()
    return ' '.join(''.join(c if c.isalnum() or c.isspace() else ' ' for c in text).split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


def error_counts(reference, hypothesis):
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    words, chars = ref.split(), ref.replace(' ', '')
    word_errors = edit_distance(words, hyp.split())
    char_errors = edit_distance(chars, hyp.replace(' ', ''))
    return dict(reference_normalized=ref, hypothesis_normalized=hyp,
                word_errors=word_errors, reference_words=len(words),
                char_errors=char_errors, reference_chars=len(chars),
                wer=word_errors/len(words) if words else None,
                cer=char_errors/len(chars) if chars else None, has_text=bool(hyp))


def mix_at_snr(speech, noise, snr_db, *, active_mask=None, speech_mask=None):
    speech, noise = np.asarray(speech, dtype=np.float64), np.asarray(noise, dtype=np.float64)
    if speech.ndim != 1 or speech.shape != noise.shape or not speech.size:
        raise ValueError('Speech/noise must be nonempty mono arrays of identical length')
    if not np.isfinite(speech).all() or not np.isfinite(noise).all() or not math.isfinite(snr_db):
        raise ValueError('Nonfinite audio or SNR')
    active_mask = np.ones(len(noise), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    speech_mask = np.ones(len(speech), dtype=bool) if speech_mask is None else np.asarray(speech_mask, dtype=bool)
    if (active_mask.shape != speech.shape or speech_mask.shape != speech.shape
            or not active_mask.any() or not speech_mask.any()):
        raise ValueError('Invalid or empty activity mask')
    speech_rms = np.sqrt(np.mean(speech[speech_mask]**2))
    noise_rms = np.sqrt(np.mean(noise[active_mask]**2))
    if min(speech_rms, noise_rms) < 1e-10:
        raise ValueError('Cannot mix silent speech/noise at a defined SNR')
    scaled_noise = noise * (speech_rms / noise_rms) * 10**(-snr_db/20)
    mixed = speech + scaled_noise
    gain = min(1., .95/max(float(np.max(np.abs(mixed))), 1e-10))
    target = speech * gain
    mixed = mixed * gain
    actual_snr = 10*np.log10(np.mean(target[speech_mask]**2)/np.mean((scaled_noise[active_mask]*gain)**2))
    return mixed.astype(np.float32), target.astype(np.float32), float(actual_snr), gain


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['kind'], row.get('noise_type'), row.get('noise_style'),
                           row['snr_db'], row['branch']), []).append(row)
    result = []
    for (kind, noise_type, noise_style, snr_db, branch), items in groups.items():
        words, chars = sum(x['reference_words'] for x in items), sum(x['reference_chars'] for x in items)
        seconds = sum(x['audio_seconds'] for x in items)
        group = dict(kind=kind, noise_type=noise_type, noise_style=noise_style,
                     snr_db=snr_db, branch=branch, clips=len(items),
                     reference_words=words, reference_chars=chars,
                     wer=sum(x['word_errors'] for x in items)/words if words else None,
                     cer=sum(x['char_errors'] for x in items)/chars if chars else None,
                     nonempty_transcript_rate=sum(x['has_text'] for x in items)/len(items),
                     median_asr_seconds=float(np.median([x['asr_seconds'] for x in items])),
                     aggregate_asr_rtf=sum(x['asr_seconds'] for x in items)/seconds)
        if branch == 'df3':
            estimates = [x['asr_seconds'] + x['enhancement_seconds'] + x['resample_seconds'] for x in items]
            group['median_pipeline_estimate_seconds'] = float(np.median(estimates))
        result.append(group)
    return result


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_audio(source, target_sr=48000):
    wave, sr = sf.read(source, dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    if wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError('Empty or nonfinite audio')
    if sr != target_sr:
        divisor = math.gcd(sr, target_sr)
        wave = resample_poly(wave, target_sr//divisor, sr//divisor).astype(np.float32)
    return wave, sr


def download(url, path, *, fallback_urls=(), expected_md5=None, attempts=3):
    path = Path(path)
    if attempts < 1:
        raise ValueError('attempts must be positive')
    def checksum_ok(candidate):
        if expected_md5 is None:
            return True
        digest = hashlib.md5()
        with candidate.open('rb') as f:
            while chunk := f.read(1024*1024):
                digest.update(chunk)
        return digest.hexdigest() == expected_md5
    if path.exists() and checksum_ok(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    urls = list(dict.fromkeys([url, *fallback_urls]))
    last_error = None
    for attempt in range(attempts):
        for source_url in urls:
            print(f'Downloading ({attempt+1}/{attempts}): {source_url}', flush=True)
            try:
                request = urllib.request.Request(source_url, headers={'User-Agent': 'ZipformerBenchmark/1.0'})
                with urllib.request.urlopen(request, timeout=60) as source, temporary.open('wb') as destination:
                    expected_size = source.headers.get('Content-Length')
                    received = 0
                    while chunk := source.read(1024*1024):
                        destination.write(chunk)
                        received += len(chunk)
                if expected_size is not None and received != int(expected_size):
                    raise OSError(f'Incomplete download: {received}/{expected_size} bytes')
                if not checksum_ok(temporary):
                    raise OSError('Downloaded file checksum mismatch')
                temporary.replace(path)
                return path
            except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
                last_error = error
                temporary.unlink(missing_ok=True)
                print(f'Download failed: {error}', flush=True)
        if attempt + 1 < attempts:
            time.sleep(2**attempt)
    raise RuntimeError(f'Cannot download {url} after {attempts} attempts across {len(urls)} URLs. '
                       f'Last error: {last_error}. You may place the original file at {path}; '
                       'it will be reused only if its checksum is valid.') from last_error


def noise_segment(wave, length, rng):
    offset = int(rng.integers(len(wave)))
    # Circular crop; reuses short isolated keyboard recordings without introducing zero pads.
    segment = wave[(offset + np.arange(length)) % len(wave)]
    if np.sqrt(np.mean(segment.astype(float)**2)) < 1e-9:
        raise ValueError('Selected noise segment is silent')
    return segment, offset


def frame_activity(wave, sr, threshold_db=-35):
    """20ms RMS gate, an energy proxy rather than a speech VAD."""
    wave = np.asarray(wave, dtype=float)
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all() or sr < 1:
        raise ValueError('Invalid waveform/sample rate')
    hop = max(1, round(.02*sr))
    padded = np.pad(wave, (0, (-len(wave)) % hop))
    rms = np.sqrt(np.mean(padded.reshape(-1, hop)**2, axis=1))
    active = rms > max(float(rms.max())*10**(threshold_db/20), 1e-10)
    return np.repeat(active, hop)[:len(wave)]


def activity_intervals(mask, sr):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [[int(start)/sr, int(stop)/sr] for start, stop in
            zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def build_noise_track(bank, length, sr, rng, noise_type, style, speech_mask):
    if not bank or style not in ('events', 'continuous') or length < 1:
        raise ValueError('Invalid noise bank, style, or track length')
    speech_mask = np.asarray(speech_mask, dtype=bool)
    if speech_mask.shape != (length,):
        raise ValueError('Speech activity length mismatch')
    if style == 'continuous':
        source = bank[int(rng.integers(len(bank)))]
        track, offset = noise_segment(source['wave'], length, rng)
        return track, frame_activity(track, sr, -45), [dict(
            start_sample=0, end_sample=length, source_file=source['file'], source_offset=offset,
            onset_region='unconstrained', requested_onset='unconstrained',
            speech_overlap_fraction=float(speech_mask.mean()))]
    track = np.zeros(length, dtype=np.float32)
    events = []
    gap, margin, hop = round(.3*sr), round(.1*sr), max(1, round(.02*sr))
    count = int(rng.integers(2, 4)) if noise_type == 'keyboard' else int(rng.integers(1, 4))
    if noise_type == 'phone':
        count = 1
    first_region = int(rng.integers(2))
    for index in range(count):
        source = bank[int(rng.integers(len(bank)))]
        offset = 0
        if noise_type == 'keyboard':
            size = min(round(float(rng.uniform(.5, 2))*sr), length-2*margin-gap)
            if size < 1:
                break
            event_wave, offset = noise_segment(source['wave'], size, rng)
            # Fade only synthetic keyboard burst boundaries, not transient source events.
            event_wave = event_wave.copy()
            fade = min(round(.005*sr), size//2)
            if fade:
                event_wave[:fade] *= np.linspace(0, 1, fade)
                event_wave[-fade:] *= np.linspace(1, 0, fade)
        elif noise_type == 'phone':
            # Preserve recorded rings; separate repeated rings by a natural quiet gap.
            pieces = []
            for ring in range(int(rng.integers(2, 4))):
                if ring:
                    pieces.append(np.zeros(round(.6*sr), dtype=np.float32))
                pieces.append(source['wave'])
            event_wave = np.concatenate(pieces)
            if len(event_wave) > min(5*sr, length-2*margin-gap):
                event_wave = source['wave']  # Use one complete ring instead of cutting its tail.
        else:
            event_wave = source['wave']  # Keep the full impact/page-turn and its decay.
        size = len(event_wave)
        if size > length-2*margin-gap:
            continue
        candidates = np.arange(margin, length-margin-size+1, hop)
        for previous in events:
            candidates = candidates[(candidates+size+gap <= previous['start_sample']) |
                                    (candidates >= previous['end_sample']+gap)]
        if not candidates.size:
            continue
        requested = 'speech' if (index+first_region) % 2 == 0 else 'pause'
        preferred = candidates[speech_mask[candidates] == (requested == 'speech')]
        onset_region = requested if preferred.size else 'unconstrained'
        start = int(rng.choice(preferred if preferred.size else candidates))
        stop = start+size
        track[start:stop] = event_wave
        event_mask = frame_activity(event_wave, sr, -45)
        events.append(dict(start_sample=start, end_sample=stop, source_file=source['file'],
                           source_offset=offset, requested_onset=requested, onset_region=onset_region,
                           speech_overlap_fraction=float(speech_mask[start:stop][event_mask].mean())))
    if not events:
        raise ValueError(f'No complete {noise_type} event fits this utterance')
    return track, frame_activity(track, sr, -45), sorted(events, key=lambda e: e['start_sample'])


def prepare(root, count=20, seed=20261019):
    if count < 1:
        raise ValueError('count must be positive')
    root = Path(root)
    cache = root/'cache'
    cache.mkdir(parents=True, exist_ok=True)
    for folder in ('input_48k', 'raw_16k', 'df3_16k', 'noise_sources'):
        (root/folder).mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    archive = download(DCASE_URL, cache/'dcase2016_task2_train_dev.zip',
                       fallback_urls=DCASE_MIRRORS, expected_md5=DCASE_MD5)
    banks = {name: [] for name in NOISE_PREFIXES}
    with zipfile.ZipFile(archive) as z:
        members = sorted(n for n in z.namelist() if n.lower().endswith('.wav') and 'train/' in n)
        for noise_type, prefix in NOISE_PREFIXES.items():
            selected = [n for n in members if Path(n).stem.lower().startswith(prefix.lower())]
            for index, member in enumerate(selected):
                payload = z.read(member)
                wave, native_sr = read_audio(io.BytesIO(payload))
                if np.sqrt(np.mean(wave.astype(float)**2)) < 1e-9:
                    continue
                name = f'{noise_type}_{index:02d}.wav'
                sf.write(root/'noise_sources'/name, wave, 48000, subtype='FLOAT')
                banks[noise_type].append(dict(wave=wave, member=member, native_sr=native_sr,
                      sha256=hashlib.sha256(payload).hexdigest(), file=name, noise_type=noise_type))
            if not banks[noise_type]:
                raise RuntimeError(f'No valid {noise_type} recordings in DCASE archive')
    tsv = download(FLEURS_BASE+'/test.tsv', cache/'fleurs_vi_test.tsv')
    references = {}
    with tsv.open(encoding='utf-8') as f:
        for fields in csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            if len(fields) < 7:
                raise ValueError('Unexpected FLEURS TSV schema')
            references[fields[1]] = dict(prompt_id=fields[0], text=fields[2], gender=fields[6])
    speech = []
    used_prompts = set()
    # Stream only the required prefix of the 544 MB archive instead of downloading all speech.
    print('Streaming FLEURS Vietnamese test audio...', flush=True)
    with urllib.request.urlopen(FLEURS_BASE+'/audio/test.tar.gz', timeout=60) as response:
        with tarfile.open(fileobj=response, mode='r|gz') as tar:
            for member in tar:
                name = Path(member.name).name
                meta = references.get(name)
                if not member.isfile() or meta is None or meta['prompt_id'] in used_prompts:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                wave, native_sr = read_audio(io.BytesIO(payload))
                if not 3 <= len(wave)/48000 <= 15:
                    continue  # Never truncate speech: transcript stays aligned with the full recording.
                wave = wave * (.05/max(float(np.sqrt(np.mean(wave.astype(float)**2))), 1e-10))
                wave *= min(1., .95/max(float(np.max(np.abs(wave))), 1e-10))
                speech.append(dict(meta, wave=wave.astype(np.float32), native_sr=native_sr,
                                   source_file=name, sha256=hashlib.sha256(payload).hexdigest()))
                used_prompts.add(meta['prompt_id'])
                print(f'Selected {len(speech)}/{count}: {name} ({len(wave)/48000:.1f}s, native {native_sr}Hz)', flush=True)
                if len(speech) == count:
                    break
    if len(speech) != count:
        raise RuntimeError(f'Only found {len(speech)} valid unique FLEURS prompts')
    rows = []
    def write_input(identifier, wave, **metadata):
        input_file = f'input_48k/{identifier}.wav'
        raw_file = f'raw_16k/{identifier}.wav'
        sf.write(root/input_file, wave, 48000, subtype='FLOAT')
        sf.write(root/raw_file, resample_poly(wave, 1, 3), 16000, subtype='FLOAT')
        rows.append(dict(id=identifier, input_48k=input_file, raw_16k=raw_file,
                         df3_16k=f'df3_16k/{identifier}.wav', audio_seconds=len(wave)/48000, **metadata))
    for index, sample in enumerate(speech):
        identifier = f'vi_{index:03d}'
        speech_mask = frame_activity(sample['wave'], 48000)
        common = {k: v for k, v in sample.items() if k != 'wave'}
        common['speech_active_intervals'] = activity_intervals(speech_mask, 48000)
        write_input(identifier+'_clean', sample['wave'], kind='clean', snr_db=None,
                    noise_type=None, noise_style='clean', events=[], noise_active_intervals=[],
                    achieved_snr_db=None, mixture_gain=1., **common)
        for noise_type, style in NOISE_CONDITIONS:
            track, mask, events = build_noise_track(banks[noise_type], len(sample['wave']), 48000,
                                                    rng, noise_type, style, speech_mask)
            for snr in (0, 5, 10):
                mixed, _, actual_snr, gain = mix_at_snr(sample['wave'], track, snr,
                                                       active_mask=mask, speech_mask=speech_mask)
                write_input(identifier+f'_{noise_type}_{style}_snr{snr:02d}', mixed,
                            kind='noisy', snr_db=snr, noise_type=noise_type, noise_style=style,
                            events=events, noise_active_intervals=activity_intervals(mask, 48000),
                            achieved_snr_db=actual_snr, mixture_gain=gain, **common)
    for noise_type, style in NOISE_CONDITIONS:
        for index in range(3):
            wave, mask, events = build_noise_track(banks[noise_type], 10*48000, 48000, rng,
                                      noise_type, style, np.zeros(10*48000, dtype=bool))
            wave = wave * (.05/float(np.sqrt(np.mean(wave[mask].astype(float)**2))))
            wave *= min(1., .95/max(float(np.max(np.abs(wave))), 1e-10))
            write_input(f'{noise_type}_{style}_only_{index:02d}', wave, kind='noise_only',
                        snr_db=None, text='', prompt_id=None, noise_type=noise_type, noise_style=style,
                        events=events, noise_active_intervals=activity_intervals(mask, 48000),
                        speech_active_intervals=[], achieved_snr_db=None, mixture_gain=None)
    manifest = dict(schema_version=2, seed=seed, speech_clips=count, snrs_db=[0, 5, 10], samples=rows,
                    noise_conditions=NOISE_CONDITIONS,
                    fleurs_revision=FLEURS_REVISION, fleurs_source=FLEURS_BASE,
                    noise_source=DCASE_URL, noise_md5=DCASE_MD5,
                    noise_recordings=[{k:v for k,v in item.items() if k!='wave'}
                                      for bank in banks.values() for item in bank],
                    licenses={'FLEURS': 'CC BY 4.0', 'DCASE2016 archive': 'CC BY 3.0'},
                    snr_definition='RMS of active speech across utterance / RMS of active noise events; '
                        '20ms energy gates: speech -35dB, noise -45dB relative to each track peak. '
                        'Not a true VAD. Same event positions across SNRs, joint anti-clipping gain.',
                    notes='First unique full 3-15s prompts in archive order. Event onset labels use '
                          'energy activity; events may extend into speech from a pause onset. '
                          'Inspect speech_overlap_fraction. Continuous keyboard is a separate condition.')
    save_json(root/'manifest.json', manifest)
    print('Prepared', len(rows), 'inputs:', count, 'clean,',
          count*len(NOISE_CONDITIONS)*3, 'noisy,', 3*len(NOISE_CONDITIONS), 'noise-only', flush=True)


def enhance_dataset(root):
    import torch
    from df.enhance import init_df, enhance
    from importlib.metadata import version
    if not torch.cuda.is_available():
        raise RuntimeError('DF3 requires CUDA')
    root = Path(root)
    manifest = read_json(root/'manifest.json')
    model, state, _, _ = init_df(model_base_dir='DeepFilterNet3', log_file=None)
    if next(model.parameters()).device.type != 'cuda' or state.sr() != 48000:
        raise RuntimeError('DF3 model must be CUDA / 48kHz')
    model.eval()
    first, _ = read_audio(root/manifest['samples'][0]['input_48k'])
    with torch.inference_mode():
        for _ in range(2):
            enhance(model, state, torch.from_numpy(first).unsqueeze(0), pad=True)
        torch.cuda.synchronize()
        metrics = {}
        for index, sample in enumerate(manifest['samples']):
            wave, _ = read_audio(root/sample['input_48k'])
            tensor = torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            cleaned = enhance(model, state, tensor, pad=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            array = cleaned.squeeze(0).detach().cpu().numpy()
            if array.shape != wave.shape or not np.isfinite(array).all():
                raise RuntimeError('DF3 returned invalid waveform/length')
            start = time.perf_counter()
            output = resample_poly(array, 1, 3).astype(np.float32)
            resample_seconds = time.perf_counter()-start
            sf.write(root/sample['df3_16k'], output, 16000, subtype='FLOAT')
            input_rms = float(np.sqrt(np.mean(wave.astype(float)**2)))
            output_rms = float(np.sqrt(np.mean(array.astype(float)**2)))
            metrics[sample['id']] = dict(enhancement_seconds=seconds, resample_seconds=resample_seconds,
                output_peak=float(np.max(np.abs(output))),
                rms_attenuation_db=20*math.log10(max(input_rms, 1e-12)/max(output_rms, 1e-12)))
            print(f'DF3 {index+1}/{len(manifest["samples"])}: {sample["id"]}', flush=True)
    save_json(root/'enhancement.json', dict(device='cuda', gpu=torch.cuda.get_device_name(0),
              torch_version=torch.__version__, deepfilternet_version=version('deepfilternet'),
              manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
              samples=metrics, notes='Two warmups; one timed enhancement/clip, CPU frontend/synthesis included; resampling separate.'))


def recognize_dataset(root, model_dir, tokens, profile_dir):
    import ctypes
    import sherpa_onnx
    from importlib.metadata import version
    # Reuse the exact tested CUDA profile guard from the original worker.
    from asr_worker import inspect_cuda_profiles
    evidence = inspect_cuda_profiles(sorted(Path(profile_dir).glob('sherpa*.json')))
    root, model_dir = Path(root), Path(model_dir)
    manifest = read_json(root/'manifest.json')
    enhancement = read_json(root/'enhancement.json')
    expected = {sample['id'] for sample in manifest['samples']}
    if (set(enhancement['samples']) != expected or enhancement.get('manifest_sha256') !=
            hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()):
        raise RuntimeError('Enhancement/manifest IDs differ; rerun DF3 for this dataset')
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=str(tokens), encoder=str(model_dir/'encoder-epoch-20-avg-10.onnx'),
        decoder=str(model_dir/'decoder-epoch-20-avg-10.onnx'),
        joiner=str(model_dir/'joiner-epoch-20-avg-10.onnx'), num_threads=2,
        sample_rate=16000, feature_dim=80, decoding_method='greedy_search', provider='cuda')
    cudart = ctypes.CDLL('libcudart.so.12')
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int
    def recognize(wave):
        if cudart.cudaDeviceSynchronize():
            raise RuntimeError('CUDA synchronization failed')
        start = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, wave)
        recognizer.decode_stream(stream)
        text = stream.result.text
        if cudart.cudaDeviceSynchronize():
            raise RuntimeError('CUDA synchronization failed')
        return text, time.perf_counter()-start
    for _ in range(5):
        for name in ('raw_16k', 'df3_16k'):
            wave, _ = read_audio(root/manifest['samples'][0][name], 16000)
            recognize(wave)
    rows = []
    for index, sample in enumerate(manifest['samples']):
        pair = {}
        for branch in ('raw', 'df3'):
            wave, native_sr = read_audio(root/sample[branch+'_16k'], 16000)
            if native_sr != 16000:
                raise RuntimeError('ASR input must already be 16 kHz')
            pair[branch] = wave
        if len(pair['raw']) != len(pair['df3']):
            raise RuntimeError('Raw/DF3 lengths differ')
        for branch in (('raw', 'df3') if index % 2 == 0 else ('df3', 'raw')):
            text, seconds = recognize(pair[branch])
            row = dict(sample, branch=branch, transcript=text, asr_seconds=seconds,
                       **error_counts(sample['text'], text))
            if branch == 'df3':
                row.update(enhancement['samples'][sample['id']])
            rows.append(row)
        print(f'ASR {index+1}/{len(manifest["samples"])}: {sample["id"]}', flush=True)
    summary = summarize(rows)
    save_json(root/'quality_results.json', dict(samples=rows, summary=summary,
        cuda_probe_executed_node_counts=evidence, sherpa_version=version('sherpa-onnx'),
        model_revision=model_dir.name, df3_gpu=enhancement['gpu'],
        notes='Corpus WER/CER sum edit counts before division; casefold+NFC, punctuation removed, tones retained, digits not verbalized. Vietnamese WER uses whitespace tokens; CER excludes spaces. Empty-reference noise clips excluded from WER/CER, reported by nonempty transcript rate. One ASR run/clip; latency aggregates different clips, not repeated performance runs. Pipeline latency is a sum across warm separate workers, not measured end-to-end.'))
    for name, data in [('quality_per_clip.csv', rows), ('quality_summary.csv', summary)]:
        keys = list(dict.fromkeys(k for row in data for k in row))
        with (root/name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'df3', 'asr'])
    p.add_argument('--root', required=True)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261019)
    p.add_argument('--model-dir')
    p.add_argument('--tokens')
    p.add_argument('--profile-dir')
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(args.root, args.count, args.seed)
    elif args.mode == 'df3':
        enhance_dataset(args.root)
    else:
        if not all((args.model_dir, args.tokens, args.profile_dir)):
            p.error('asr requires --model-dir, --tokens, --profile-dir')
        recognize_dataset(args.root, args.model_dir, args.tokens, args.profile_dir)


if __name__ == '__main__':
    main()
