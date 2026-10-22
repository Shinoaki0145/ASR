import numpy as np
import soxr


class AudioIngress:
    def __init__(self):
        self.rate = None
        self.resampler = None
        self.tail = np.empty(0, np.float32)
        self.finished = False

    def frames(self, wave):
        self.tail = np.concatenate((self.tail, wave))
        count = len(self.tail) // 512
        result = [x.copy() for x in self.tail[: count * 512].reshape(-1, 512)]
        self.tail = self.tail[count * 512 :]
        return result

    def push(self, rate, samples):
        if self.finished:
            raise ValueError("Recording đã đóng")
        if self.rate is not None and self.rate != rate:
            raise ValueError("Sample rate thay đổi giữa recording")
        x = np.asarray(samples)
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / max(
                abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max
            )
        else:
            x = x.astype(np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("Audio không hợp lệ")
        if self.rate is None:
            self.rate = rate
            if rate != 16000:
                self.resampler = soxr.ResampleStream(rate, 16000, 1, dtype="float32")
        if self.resampler:
            x = self.resampler.resample_chunk(x)
        return self.frames(x)

    def finish(self):
        if self.finished:
            return []
        self.finished = True
        result = (
            self.frames(
                self.resampler.resample_chunk(np.empty(0, np.float32), last=True)
            )
            if self.resampler
            else []
        )
        if self.tail.size:
            result.append(np.pad(self.tail, (0, 512 - len(self.tail))))
            self.tail = np.empty(0, np.float32)
        return result
