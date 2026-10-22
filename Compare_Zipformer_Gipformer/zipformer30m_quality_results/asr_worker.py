
import argparse, csv, ctypes, gc, json, os, time, threading
from pathlib import Path
from collections import Counter
from importlib.metadata import version
import numpy as np
import soundfile as sf
import pynvml
import sherpa_onnx


def inspect_cuda_profiles(paths):
    counts = Counter()
    for path in paths:
        events = json.loads(Path(path).read_text())
        for event in events:
            provider = event.get('args', {}).get('provider')
            if event.get('cat') == 'Node' and provider:
                counts[provider] += 1
    if counts['CUDAExecutionProvider'] == 0:
        raise RuntimeError('Không có node CUDA thực thi trong profile Sherpa; dừng benchmark.')
    return dict(counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['probe', 'benchmark'], required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--tokens', required=True)
    p.add_argument('--profile-dir', required=True)
    a = p.parse_args()
    model_dir = Path(a.model_dir)
    profile_dir = Path(a.profile_dir)
    if a.mode == 'probe':
        config = profile_dir / 'provider.conf'
        config.write_text(f'ProfilingFilePrefix={profile_dir}/sherpa\nLogSeverityLevel=1\n')
        provider = f'cuda:{config}'
    else:
        evidence = inspect_cuda_profiles(sorted(profile_dir.glob('sherpa*.json')))
        provider = 'cuda'

    # CUDA_VISIBLE_DEVICES is set to physical GPU 0 by the notebook.
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    def memory():
        total = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        used = [x.usedGpuMemory for x in processes if x.pid == os.getpid()]
        pid_mib = sum(used) / 2**20 if used and all(x is not None and x != (1 << 64) - 1 for x in used) else None
        return {'total_mib': total, 'process_mib': pid_mib}

    before = memory()
    start = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=a.tokens, encoder=str(model_dir/'encoder-epoch-20-avg-10.onnx'),
        decoder=str(model_dir/'decoder-epoch-20-avg-10.onnx'),
        joiner=str(model_dir/'joiner-epoch-20-avg-10.onnx'),
        num_threads=2, sample_rate=16000, feature_dim=80,
        decoding_method='greedy_search', provider=provider)
    # Only the CUDA runtime is loaded; do not create a Torch context for measurement.
    cudart = ctypes.CDLL('libcudart.so.12')
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int
    def synchronize():
        status = cudart.cudaDeviceSynchronize()
        if status:
            raise RuntimeError(f'cudaDeviceSynchronize failed: {status}')
    synchronize()
    load_seconds = time.perf_counter() - start
    after = memory()
    print('Sherpa:', version('sherpa-onnx'), '| GPU:', gpu_name, flush=True)
    print('Memory before/after load:', before, after, flush=True)

    def recognize_once(signal):
        synchronize()
        start = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, signal)
        recognizer.decode_stream(stream)
        text = stream.result.text
        synchronize()
        return text, time.perf_counter() - start

    if a.mode == 'probe':
        signal, sr = sf.read('/content/asr_test_16k.wav', dtype='float32')
        assert sr == 16000
        print('Probe:', recognize_once(signal), flush=True)
        del recognizer
        gc.collect()
        return  # Process exit also destroys ORT sessions and flushes profiling.

    class GPUMonitor:
        def __init__(self):
            self.samples = []
            self.errors = []
            self.stop = threading.Event()
        def __enter__(self):
            self.samples.append(memory())
            self.thread = threading.Thread(target=self.poll, daemon=True)
            self.thread.start()
            return self
        def poll(self):
            while not self.stop.wait(0.02):
                try:
                    self.samples.append(memory())
                except Exception as error:
                    self.errors.append(str(error))
                    return
        def __exit__(self, *args):
            self.stop.set()
            self.thread.join()
            self.samples.append(memory())

    audios = {}
    for name, file in [('raw', 'asr_test_16k.wav'), ('df3', 'asr_test_df3_16k.wav')]:
        signal, sr = sf.read('/content/'+file, dtype='float32')
        assert sr == 16000 and signal.ndim == 1 and signal.size > 0
        assert np.isfinite(signal).all()
        audios[name] = signal
    assert len(audios['raw']) == len(audios['df3']), 'Raw/DF3 khác độ dài'
    results = {name: [] for name in audios}
    n_warmup, n_runs = 5, 20
    baseline = memory()
    with GPUMonitor() as monitor:
        # Warm both variants with monitoring already enabled.
        for _ in range(n_warmup):
            for signal in audios.values():
                recognize_once(signal)
        # Alternate order to reduce systematic raw-first drift; retain all runs.
        for index in range(n_runs):
            order = ('raw', 'df3') if index % 2 == 0 else ('df3', 'raw')
            for name in order:
                results[name].append(recognize_once(audios[name]))
    df_metrics = json.loads(Path('/content/df3_metrics.json').read_text())
    pid_samples = [s['process_mib'] for s in monitor.samples if s['process_mib'] is not None]
    for name, runs in results.items():
        latencies = [r[1] for r in runs]
        texts = [r[0] for r in runs]
        seconds = len(audios[name]) / 16000
        median = float(np.median(latencies))
        p95 = float(np.percentile(latencies, 95))
        report = {
            'model': 'hynt/Zipformer-30M-RNNT-6000h', 'model_revision': model_dir.name,
            'audio_variant': name, 'runtime': 'sherpa-onnx ONNX FP32',
            'sherpa_version': version('sherpa-onnx'), 'gpu': gpu_name,
            'driver': pynvml.nvmlSystemGetDriverVersion(), 'requested_provider': 'cuda',
            'cuda_probe_executed_node_counts': evidence,
            'decoding_method': 'greedy_search', 'num_threads': 2,
            'model_load_seconds': load_seconds,
            'warmup_runs': n_warmup, 'measured_runs': n_runs, 'audio_seconds': seconds,
            'median_latency_seconds': median, 'p95_latency_seconds': p95,
            'median_rtf': median/seconds, 'p95_rtf': p95/seconds,
            'gpu_memory_before_load': before, 'gpu_memory_after_load': after,
            'gpu_memory_baseline': baseline,
            'gpu_memory_peak_total_mib': max(s['total_mib'] for s in monitor.samples),
            'gpu_memory_peak_process_mib': max(pid_samples) if pid_samples else None,
            'memory_monitor_errors': monitor.errors,
            'text': texts[0], 'transcript_consistent': len(set(texts)) == 1,
            'transcripts': texts, 'latencies_seconds': latencies,
            'audio_peak': float(np.max(np.abs(audios[name]))),
            'audio_fraction_above_one': float(np.mean(np.abs(audios[name]) > 1)),
            'deepfilternet3': df_metrics if name == 'df3' else None,
            'approx_pipeline_median_seconds': median + df_metrics['median_enhancement_seconds'] + df_metrics['single_resample_seconds'] if name == 'df3' else None,
            'notes': 'Latency: stream + waveform + decode + text + CUDA sync; excludes model load/I/O. NVML 20ms sampled peak includes warmup; PID memory includes CUDA context/allocator. Pipeline sum is an estimate across separate warm workers, not measured end-to-end or a median of sums.'}
        # Ensure old drivers returning bytes still produce valid JSON.
        if isinstance(report['driver'], bytes):
            report['driver'] = report['driver'].decode()
        prefix = Path(f'/content/zipformer30m_{name}_benchmark')
        prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        with prefix.with_suffix('.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['run', 'latency_seconds', 'rtf'])
            writer.writeheader()
            writer.writerows({'run': i+1, 'latency_seconds': t, 'rtf': t/seconds}
                             for i, t in enumerate(latencies))
        print(f'{name}: median={median:.6f}s; p95={p95:.6f}s; RTF={median/seconds:.6f}', flush=True)
        print('Transcript:', texts[0], '| consistent:', report['transcript_consistent'], flush=True)
    pynvml.nvmlShutdown()


if __name__ == '__main__':
    main()
