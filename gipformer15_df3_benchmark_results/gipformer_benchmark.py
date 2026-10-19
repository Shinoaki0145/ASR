
"""Gipformer 1.5 CUDA benchmark, using the official PyTorch inference recipe."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from quality_benchmark import read_json, save_json, read_audio, error_counts, summarize

MODEL_ID = 'g-group-ai-lab/gipformer1.5-68M-rnnt'


def write_quality_reports(output, rows, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    save_json(output/'quality_results.json', dict(metadata, samples=rows, summary=summary,
        notes='Shared normalization and corpus WER/CER with Zipformer. NFC/casefold, punctuation removed, '
              'Vietnamese tones retained; digits not verbalized. WER uses whitespace tokens; CER excludes spaces. '
              'Noise-only clips have no WER/CER. Pipeline sums use separate warm DF3 and ASR workers, '
              'not measured end-to-end. Quality ASR timing is one run per variable-length clip.'))
    for filename, data in [('quality_per_clip.csv', rows), ('quality_summary.csv', summary)]:
        keys = list(dict.fromkeys(k for row in data for k in row))
        with (output/filename).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)


class Gipformer:
    def __init__(self, repo, icefall_dir, model_dir, decoding_method, beam_size):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Gipformer requires CUDA; CPU fallback is not allowed')
        self.torch = torch
        self.device = torch.device('cuda:0')
        sys.path.insert(0, str(repo))
        from infer_pytorch import setup_icefall, load_model_files
        setup_icefall(Path(icefall_dir))
        import k2
        import kaldifeat
        import sentencepiece as spm
        from beam_search import greedy_search_batch, modified_beam_search
        from torch.nn.utils.rnn import pad_sequence
        from train import add_model_arguments, get_model, get_params
        # Icefall sets inter-op once at import. Only intra-op can be adjusted here.
        torch.set_num_threads(2)
        self.greedy = greedy_search_batch
        self.beam_search = modified_beam_search
        self.pad_sequence = pad_sequence
        parser = argparse.ArgumentParser()
        parser.add_argument('--context-size', type=int, default=2)
        add_model_arguments(parser)
        old_git_dir = os.environ.get('GIT_DIR')
        try:
            os.environ['GIT_DIR'] = str(Path(icefall_dir)/'.git')
            params = get_params()
        finally:
            if old_git_dir is None:
                os.environ.pop('GIT_DIR', None)
            else:
                os.environ['GIT_DIR'] = old_git_dir
        params.update(vars(parser.parse_args([])))
        params.decoding_method, params.beam_size = decoding_method, beam_size
        paths = load_model_files(version='1.5', model_dir=str(model_dir))
        tokens = k2.SymbolTable.from_file(paths['tokens'])
        params.blank_id, params.unk_id = tokens['<blk>'], tokens['<unk>']
        count = sum(not symbol.startswith('#') for symbol in tokens.symbols)
        if params.blank_id == 0:
            count -= 1
        params.vocab_size = count+1  # Same vocabulary calculation as the official script.
        load_start = time.perf_counter()
        self.model = get_model(params)
        checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
        state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(f'Checkpoint is missing model parameters: {incompatible.missing_keys[:20]}')
        self.model.to(self.device).eval()
        if not all(p.device.type == 'cuda' for p in self.model.parameters()):
            raise RuntimeError('Some Gipformer parameters are not on CUDA')
        torch.cuda.synchronize()
        self.model_load_seconds = time.perf_counter()-load_start
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(paths['bpe_model'])
        opts = kaldifeat.FbankOptions()
        opts.device = self.device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400
        self.fbank = kaldifeat.Fbank(opts)
        self.decoding_method, self.beam_size = decoding_method, beam_size
        self.metadata = dict(model=MODEL_ID, model_version='1.5', runtime='official PyTorch CUDA recipe',
            model_revision=Path(model_dir).name, torch_version=torch.__version__, torch_cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), k2_version=getattr(k2, '__version__', getattr(k2, '__dev_version__', None)),
            decoding_method=decoding_method, beam_size=beam_size if decoding_method=='modified_beam_search' else None,
            context_size=2, num_threads=torch.get_num_threads(),
            num_interop_threads=torch.get_num_interop_threads(), model_parameters=sum(p.numel() for p in self.model.parameters()),
            model_load_seconds=self.model_load_seconds,
            checkpoint_unexpected_keys=list(incompatible.unexpected_keys),
            cuda_checks=dict(model_parameters=True, features=False, encoder_output=False),
            repo_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip(),
            icefall_commit=subprocess.check_output(['git','-C',str(icefall_dir),'rev-parse','HEAD'], text=True).strip())
        print(json.dumps(self.metadata, ensure_ascii=False, indent=2), flush=True)

    def recognize(self, wave):
        torch = self.torch
        if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
            raise ValueError('Invalid waveform')
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)).to(self.device)
            features = self.fbank([tensor])
            lengths = torch.tensor([features[0].size(0)], device=self.device)
            features = self.pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
            if features.device.type != 'cuda':
                raise RuntimeError('Fbank features are not on CUDA')
            encoded, encoded_lengths = self.model.forward_encoder(features, lengths)
            if encoded.device.type != 'cuda':
                raise RuntimeError('Encoder output is not on CUDA')
            if self.decoding_method == 'greedy_search':
                hypotheses = self.greedy(model=self.model, encoder_out=encoded,
                                        encoder_out_lens=encoded_lengths)
            else:
                hypotheses = self.beam_search(model=self.model, encoder_out=encoded,
                                  encoder_out_lens=encoded_lengths, beam=self.beam_size)
            text = self.sp.decode(hypotheses[0])
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        self.metadata['cuda_checks'].update(features=True, encoder_output=True)
        return text, elapsed


def load_pair(root, sample):
    pair = {}
    for branch in ('raw', 'df3'):
        wave, sr = read_audio(Path(root)/sample[branch+'_16k'], 16000)
        if sr != 16000:
            raise RuntimeError('ASR files must already be 16kHz')
        pair[branch] = wave
    if len(pair['raw']) != len(pair['df3']):
        raise RuntimeError('Raw/DF3 lengths differ')
    return pair


def warmup(backend, pair):
    for _ in range(5):
        for wave in pair.values():
            backend.recognize(wave)


def performance(backend, root, output, sample, enhancement):
    pair = load_pair(root, sample)
    warmup(backend, pair)
    torch = backend.torch
    torch.cuda.reset_peak_memory_stats()
    results = {'raw': [], 'df3': []}
    for index in range(20):
        for branch in (('raw','df3') if index % 2 == 0 else ('df3','raw')):
            results[branch].append(backend.recognize(pair[branch]))
    for branch, runs in results.items():
        latencies = [elapsed for text, elapsed in runs]
        seconds = len(pair[branch])/16000
        median = float(np.median(latencies))
        p95 = float(np.percentile(latencies, 95))
        report = dict(backend.metadata, sample_id=sample['id'], audio_variant=branch,
            audio_seconds=seconds, warmup_runs=5, measured_runs=20,
            median_latency_seconds=median, p95_latency_seconds=p95,
            median_rtf=median/seconds, p95_rtf=p95/seconds,
            latencies_seconds=latencies, text=runs[0][0],
            transcript_consistent=len({text for text, elapsed in runs})==1,
            torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            df3=enhancement if branch=='df3' else None,
            notes='Warm synchronized waveform transfer + Fbank + encoder + decode + text. '
                  'Excludes file I/O/startup/model load. Torch memory includes allocator tensors, '
                  'not all CUDA context/driver allocations. Memory peak shared across both variants.')
        save_json(Path(output)/f'gipformer15_{branch}_benchmark.json', report)
        with (Path(output)/f'gipformer15_{branch}_benchmark.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=['run','latency_seconds','rtf'])
            writer.writeheader()
            writer.writerows(dict(run=i+1,latency_seconds=t,rtf=t/seconds) for i,t in enumerate(latencies))
        print(f'{branch}: median={median:.4f}s, P95={p95:.4f}s, RTF={median/seconds:.5f}',flush=True)


def quality(backend, root, output, manifest, enhancement):
    rows = []
    warmup(backend, load_pair(root, manifest['samples'][0]))
    for index, sample in enumerate(manifest['samples']):
        pair = load_pair(root, sample)
        for branch in (('raw','df3') if index % 2 == 0 else ('df3','raw')):
            text, seconds = backend.recognize(pair[branch])
            row = dict(sample, branch=branch, transcript=text, asr_seconds=seconds,
                       **error_counts(sample['text'], text))
            if branch=='df3':
                row.update(enhancement['samples'][sample['id']])
            rows.append(row)
        print(f'Gipformer {index+1}/{len(manifest["samples"])}: {sample["id"]}',flush=True)
    write_quality_reports(output, rows, dict(backend.metadata,
        manifest_sha256=hashlib.sha256((Path(root)/'manifest.json').read_bytes()).hexdigest(),
        df3_gpu=enhancement['gpu']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--icefall-dir', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--sample-id', default='vi_000_keyboard_events_snr05')
    p.add_argument('--decoding-method', choices=['modified_beam_search','greedy_search'], default='modified_beam_search')
    p.add_argument('--beam-size', type=int, default=4)
    args=p.parse_args()
    if args.beam_size < 1:
        p.error('beam-size must be positive')
    root, output=Path(args.root),Path(args.output)
    if root.resolve()==output.resolve():
        p.error('output must be a different directory from the shared dataset')
    output.mkdir(parents=True,exist_ok=True)
    manifest, enhancement=read_json(root/'manifest.json'),read_json(root/'enhancement.json')
    sha=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    ids={sample['id'] for sample in manifest['samples']}
    if enhancement.get('manifest_sha256')!=sha or set(enhancement['samples'])!=ids:
        raise RuntimeError('DF3 outputs do not match the dataset; rerun enhancement')
    sample=next((sample for sample in manifest['samples'] if sample['id']==args.sample_id),None)
    if sample is None:
        raise ValueError('Unknown performance sample ID')
    backend=Gipformer(args.repo,args.icefall_dir,args.model_dir,args.decoding_method,args.beam_size)
    performance(backend,root,output,sample,enhancement['samples'][sample['id']])
    quality(backend,root,output,manifest,enhancement)


if __name__=='__main__':
    main()
