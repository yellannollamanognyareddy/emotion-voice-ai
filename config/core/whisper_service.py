"""
EMORA — Sarvam AI Telugu Transcription Service
"""

import numpy as np
import scipy.io.wavfile as wavfile
import httpx
import tempfile
import os

SARVAM_API_KEY  = "sk_ukcn7ekp_M91Ye44c6lj9Qeo3VRobUdeB"
SAMPLE_RATE     = 16000
SARVAM_ENDPOINT = "https://api.sarvam.ai/speech-to-text"


def transcribe_audio_array(audio: np.ndarray) -> str:
    tmp_path = None
    try:
        print("Sending audio to Sarvam AI...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wavfile.write(
                tmp.name,
                SAMPLE_RATE,
                (audio * 32767).astype(np.int16)
            )
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            response = httpx.post(
                SARVAM_ENDPOINT,
                headers={"api-subscription-key": SARVAM_API_KEY},
                data={
                    "model":         "saaras:v3",
                    "language_code": "te-IN",
                    "mode":          "transcribe"
                },
                files={"file": ("audio.wav", f, "audio/wav")},
                timeout=30
            )

        print(f"Sarvam status: {response.status_code}")
        result     = response.json()
        transcript = result.get("transcript", "").strip()
        print(f"Transcript ({len(transcript)} chars): {transcript}")
        return transcript

    except Exception as e:
        print("Transcription Error:", str(e))
        return ""

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except:
                pass


def transcribe_audio_file(audio_path: str) -> str:
    import soundfile as sf
    import librosa

    try:
        audio, sr = sf.read(audio_path)
        print(f"Audio loaded: {len(audio)} samples @ {sr}Hz")

        if sr != SAMPLE_RATE:
            print(f"Resampling {sr}Hz → {SAMPLE_RATE}Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return transcribe_audio_array(audio)

    except Exception as e:
        print("File Transcription Error:", str(e))
        return ""