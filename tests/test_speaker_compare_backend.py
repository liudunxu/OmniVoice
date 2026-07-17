import os
import unittest
from unittest import mock

import numpy as np

import api


def _tone(seconds: float = 1.6, sample_rate: int = 16000, frequency: float = 170.0) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return 0.25 * np.sin(2.0 * np.pi * frequency * t)


class SpeakerCompareBackendDispatchTest(unittest.TestCase):
    def setUp(self):
        api._speaker_compare_fallback_warned = False

    def test_wavlm_backend_used_when_embeddings_available(self):
        with mock.patch.dict(os.environ, {"SPEAKER_COMPARE_BACKEND": "wavlm"}), \
            mock.patch.object(api.speaker_embedding, "available", return_value=True), \
            mock.patch.object(
                api.speaker_embedding,
                "embedding",
                side_effect=[np.array([1.0, 0.0]), np.array([0.5, 0.5])],
            ) as embed, \
            mock.patch.object(api.speaker_embedding, "cosine", return_value=0.87):
            result = api._compare_speaker_waveforms(_tone(), 16000, _tone(), 16000)

        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "wavlm_base_sv")
        self.assertAlmostEqual(result["similarity"], 0.87, places=4)
        self.assertEqual(embed.call_count, 2)
        self.assertIn("gender", result["left_gender"])
        self.assertIn("gender", result["right_gender"])

    def test_fallback_to_mfcc_when_backend_unavailable(self):
        with mock.patch.dict(os.environ, {"SPEAKER_COMPARE_BACKEND": "wavlm"}), \
            mock.patch.object(api.speaker_embedding, "available", return_value=False), \
            mock.patch.object(api.speaker_embedding, "embedding") as embed:
            result = api._compare_speaker_waveforms(_tone(), 16000, _tone(), 16000)

        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "mfcc_v1")
        self.assertGreater(result["similarity"], 0.99)
        embed.assert_not_called()

    def test_fallback_to_mfcc_when_embedding_returns_none(self):
        with mock.patch.dict(os.environ, {"SPEAKER_COMPARE_BACKEND": "wavlm"}), \
            mock.patch.object(api.speaker_embedding, "available", return_value=True), \
            mock.patch.object(api.speaker_embedding, "embedding", return_value=None):
            result = api._compare_speaker_waveforms(_tone(), 16000, _tone(), 16000)

        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "mfcc_v1")

    def test_fallback_to_mfcc_on_embedding_exception(self):
        with mock.patch.dict(os.environ, {"SPEAKER_COMPARE_BACKEND": "wavlm"}), \
            mock.patch.object(api.speaker_embedding, "available", return_value=True), \
            mock.patch.object(
                api.speaker_embedding, "embedding", side_effect=RuntimeError("boom")
            ):
            result = api._compare_speaker_waveforms(_tone(), 16000, _tone(), 16000)

        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "mfcc_v1")

    def test_mfcc_backend_skips_embedding_entirely(self):
        with mock.patch.dict(os.environ, {"SPEAKER_COMPARE_BACKEND": "mfcc"}), \
            mock.patch.object(api.speaker_embedding, "available") as available:
            result = api._compare_speaker_waveforms(_tone(), 16000, _tone(), 16000)

        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "mfcc_v1")
        available.assert_not_called()


class AppendGenderMismatchIssueTest(unittest.TestCase):
    def test_appends_on_confident_mismatch(self):
        identity_qc = {"right_gender": {"gender": "male", "confidence": 0.9}}
        issues = api._append_gender_mismatch_issue([], "女", identity_qc)
        self.assertEqual(issues, ["gender_mismatch"])

    def test_no_append_below_confidence_threshold(self):
        identity_qc = {"right_gender": {"gender": "male", "confidence": 0.4}}
        issues = api._append_gender_mismatch_issue([], "female", identity_qc)
        self.assertEqual(issues, [])

    def test_no_append_for_unknown_declared_or_output(self):
        self.assertEqual(
            api._append_gender_mismatch_issue(
                [], "unknown", {"right_gender": {"gender": "male", "confidence": 0.9}}
            ),
            [],
        )
        self.assertEqual(
            api._append_gender_mismatch_issue(
                [], "male", {"right_gender": {"gender": "unknown", "confidence": 0.9}}
            ),
            [],
        )

    def test_no_duplicate_and_none_safe(self):
        identity_qc = {"right_gender": {"gender": "female", "confidence": 0.95}}
        issues = api._append_gender_mismatch_issue(["gender_mismatch"], "m", identity_qc)
        self.assertEqual(issues, ["gender_mismatch"])
        self.assertIsNone(api._append_gender_mismatch_issue(None, "male", None))


if __name__ == "__main__":
    unittest.main()
