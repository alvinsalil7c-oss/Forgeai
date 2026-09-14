"""
Stage 10 — Microphone capture.

Requires (not yet installed per your status doc):
    pip install sounddevice scipy

sounddevice needs PortAudio, which ships bundled in its Windows wheel, so
no separate PortAudio install should be needed on Windows.
"""

import queue
import numpy as np
import sounddevice as sd


class MicCapture:
    def __init__(self, samplerate=16000, channels=1, blocksize=1024, device=None):
        """
        samplerate=16000 matches what webrtcvad (Stage 11) and most speech
        feature pipelines expect — resample later only if you need higher
        fidelity for something else.
        """
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device
        self._q = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[MicCapture] status: {status}")
        self._q.put(indata.copy())

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            device=self.device,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_available(self):
        """Drain whatever blocks have arrived since the last call."""
        chunks = []
        while not self._q.empty():
            chunks.append(self._q.get_nowait())
        if not chunks:
            return np.zeros((0, self.channels), dtype="float32")
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def list_devices():
        return sd.query_devices()


if __name__ == "__main__":
    import time

    print(MicCapture.list_devices())
    mic = MicCapture()
    mic.start()
    print("Recording 5s test... speak into the mic.")
    levels = []
    t0 = time.time()
    while time.time() - t0 < 5:
        chunk = mic.read_available()
        if len(chunk):
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            levels.append(rms)
        time.sleep(0.05)
    mic.stop()
    print(f"Captured {len(levels)} chunks, avg RMS: {np.mean(levels) if levels else 0:.5f}")
