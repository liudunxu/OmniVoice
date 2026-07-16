import io
import unittest

import numpy as np
import soundfile as sf

import api


def _tone_wav(frequency: float, seconds: float = 1.6, sample_rate: int = 16000) -> bytes:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    waveform = 0.25 * np.sin(2.0 * np.pi * frequency * t)
    output = io.BytesIO()
    sf.write(output, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


class ReferenceIdentityQcTest(unittest.TestCase):
    def test_reference_quality_reports_gender_profile_and_speaker_check(self):
        result = api._reference_quality_legacy(_tone_wav(110.0), enable_speaker_check=True)

        self.assertTrue(result["ok"])
        self.assertIn("speaker_count", result)
        self.assertIn("gender", result)
        self.assertIn("gender_confidence", result)

    def test_reference_similarity_is_high_for_same_audio(self):
        audio = _tone_wav(170.0)
        comparison = api._compare_speaker_audio(audio, audio)

        self.assertTrue(comparison["ok"])
        self.assertGreater(comparison["similarity"], 0.99)

    def test_music_leakage_is_high_for_shared_signal(self):
        audio = _tone_wav(220.0)
        leakage = api._estimate_music_leakage(audio, audio)

        self.assertTrue(leakage["ok"])
        self.assertGreater(leakage["score"], 0.9)

    def test_prosody_similarity_is_high_for_same_audio(self):
        audio = _tone_wav(180.0)
        waveform, sample_rate = api._decode_audio_bytes_mono(audio, 16000)
        comparison = api._compare_prosody(
            waveform, sample_rate, waveform, sample_rate
        )

        self.assertTrue(comparison["ok"])
        self.assertGreater(comparison["similarity"], 0.99)


if __name__ == "__main__":
    unittest.main()
