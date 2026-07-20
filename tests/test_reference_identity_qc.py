import io
import unittest
from unittest.mock import AsyncMock, patch

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

    def test_prosody_warning_requires_multiple_deviant_axes(self):
        one_axis = {
            "ok": True,
            "similarity": 0.10,
            "source": {
                "f0_range_octaves": 0.2,
                "energy_range_db": 10.0,
                "activity_ratio": 0.6,
            },
            "output": {
                "f0_range_octaves": 1.2,
                "energy_range_db": 11.0,
                "activity_ratio": 0.58,
            },
        }
        two_axes = {
            **one_axis,
            "source": dict(one_axis["source"]),
            "output": {**one_axis["output"], "activity_ratio": 0.2},
        }

        self.assertFalse(api._prosody_mismatch_is_corroborated(one_axis, 3.0))
        self.assertTrue(api._prosody_mismatch_is_corroborated(two_axes, 3.0))
        self.assertEqual(
            two_axes["mismatch_assessment"]["deviant_axes"],
            ["pitch_range", "activity"],
        )

    def test_refresh_waveform_qc_drops_resolved_signal_issue(self):
        refreshed = api._refresh_waveform_quality_issues(
            ["too_much_silence", "prompt_leak", "duration_off_reference"],
            [],
        )

        self.assertEqual(refreshed, ["prompt_leak"])

    def test_reference_endpoint_guard_ignores_octave_gender_flip(self):
        body = {"gender": "male", "confidence": 0.9, "f0": 120.0}
        edge = {"gender": "female", "confidence": 0.9, "f0": 240.0}

        self.assertFalse(api._reference_endpoint_gender_conflict(body, edge))

    def test_contaminated_current_text_qc_can_be_replaced_at_equal_coverage(self):
        current = {"coverage": 1.0, "source_script_residue": True}
        candidate = {"coverage": 1.0, "source_script_residue": False}

        self.assertTrue(
            api._should_accept_text_qc_candidate(current, candidate, 0, 0, False)
        )


class OutputTextQcTest(unittest.IsolatedAsyncioTestCase):
    async def test_cross_language_prompt_leak_uses_unbiased_second_pass(self):
        forced = {
            "duration": 11.43,
            "language": "tl",
            "language_probability": 1.0,
            "segments": [{"text": "Kuya anong nangyari"}],
        }
        automatic = {
            "duration": 11.43,
            "language": "zh",
            "language_probability": 0.99,
            "segments": [{"text": "大师兄你怎么了"}],
        }
        with (
            patch.object(api, "_ensure_whisper_model", new=AsyncMock(return_value=object())),
            patch.object(api, "_transcribe_whisper_sync", side_effect=[forced, automatic]),
        ):
            result = await api._build_output_text_qc(
                "/tmp/not-read-by-mock.wav",
                "Kuya, anong nangyari?",
                "fil",
                {
                    "prompt_text": "大师兄。你怎么了？",
                    "target_duration_ms": 2580,
                },
            )

        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(result["source_script_residue"])
        self.assertTrue(result["prompt_leak_detected"])
        self.assertEqual(result["prompt_leak_audit"]["language"], "zh")

    async def test_reference_only_retry_drops_continuation_prompt_cache(self):
        class FakeTtsModel:
            def build_prompt_cache(self, **kwargs):
                return kwargs

        class FakeModel:
            tts_model = FakeTtsModel()

        result = await api._voxcpm_reference_only_retry_kwargs(
            FakeModel(),
            {
                "prompt_path": "/tmp/source.wav",
                "prompt_text": "大师兄。你怎么了？",
                "prompt_cache": {"continuation": True},
                "trim_silence_vad": True,
            },
            "/tmp/reference.wav",
        )

        self.assertIsNone(result["prompt_path"])
        self.assertEqual(result["prompt_text"], "")
        self.assertEqual(
            result["prompt_cache"],
            {"trim_silence_vad": True, "reference_wav_path": "/tmp/reference.wav"},
        )


if __name__ == "__main__":
    unittest.main()
