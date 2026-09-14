"""
Stage 11 — Audio features.

Requires (not yet installed per your status doc):
    pip install librosa webrtcvad-wheels

Reminder from your decision log: `webrtcvad-wheels` is a drop-in for
`webrtcvad` on Windows (same `import webrtcvad`), needed because plain
webrtcvad fails to build without a C++ compiler.

webrtcvad requires 16-bit PCM mono audio at 8/16/32/48 kHz in frames of
exactly 10/20/30 ms — MicCapture above is already set to 16000 Hz mono,
so we just need the float32 -> int16 conversion and correct frame sizing
here.
"""

import numpy as np
import webrtcvad
import librosa


class AudioFeatureExtractor:
    def __init__(self, samplerate=16000, vad_aggressiveness=2, frame_ms=30):
        """
        vad_aggressiveness: 0 (least aggressive, more false positives on
            speech) to 3 (most aggressive, filters more non-speech).
        frame_ms: must be 10, 20, or 30 for webrtcvad.
        """
        assert frame_ms in (10, 20, 30)
        self.samplerate = samplerate
        self.frame_ms = frame_ms
        self.frame_len = int(samplerate * frame_ms / 1000)
        self.vad = webrtcvad.Vad(vad_aggressiveness)

    @staticmethod
    def _float_to_pcm16(chunk_float32):
        clipped = np.clip(chunk_float32, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)

    def voice_activity(self, chunk_float32):
        """
        chunk_float32: mono float32 array from MicCapture, any length.
        Returns fraction of frame_ms sub-frames flagged as speech (0-1),
        and the number of full frames it could evaluate.
        """
        if chunk_float32.ndim > 1:
            chunk_float32 = chunk_float32.mean(axis=1)
        pcm16 = self._float_to_pcm16(chunk_float32)
        pcm_bytes = pcm16.tobytes()

        bytes_per_frame = self.frame_len * 2  # int16 = 2 bytes/sample
        n_frames = len(pcm_bytes) // bytes_per_frame
        if n_frames == 0:
            return {"voice_fraction": 0.0, "frames_evaluated": 0}

        speech_count = 0
        for i in range(n_frames):
            frame = pcm_bytes[i * bytes_per_frame:(i + 1) * bytes_per_frame]
            if self.vad.is_speech(frame, self.samplerate):
                speech_count += 1

        return {
            "voice_fraction": speech_count / n_frames,
            "frames_evaluated": n_frames,
        }

    def energy_and_pitch(self, chunk_float32):
        """
        RMS energy (loudness proxy) + fundamental frequency estimate
        (pitch proxy for vocal stress/excitement), via librosa.
        """
        if chunk_float32.ndim > 1:
            chunk_float32 = chunk_float32.mean(axis=1)
        if len(chunk_float32) < 512:
            return {"rms": 0.0, "pitch_hz": 0.0}

        rms = float(np.sqrt(np.mean(chunk_float32 ** 2)))

        try:
            f0, voiced_flag, _ = librosa.pyin(
                chunk_float32,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=self.samplerate,
            )
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
            pitch = float(np.nanmean(voiced_f0)) if len(voiced_f0) else 0.0
        except Exception:
            pitch = 0.0

        return {"rms": rms, "pitch_hz": pitch}


if __name__ == "__main__":
    import time
    from mic_capture import MicCapture

    mic = MicCapture()
    extractor = AudioFeatureExtractor()
    mic.start()
    print("Testing VAD + energy/pitch for 5s...")
    t0 = time.time()
    while time.time() - t0 < 5:
        chunk = mic.read_available()
        if len(chunk):
            vad_result = extractor.voice_activity(chunk)
            ep_result = extractor.energy_and_pitch(chunk)
            print(f"voice: {vad_result['voice_fraction']:.2f}  "
                  f"rms: {ep_result['rms']:.4f}  pitch: {ep_result['pitch_hz']:.1f}Hz")
        time.sleep(0.2)
    mic.stop()
