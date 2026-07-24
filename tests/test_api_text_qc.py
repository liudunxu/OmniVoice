import unittest

import numpy as np

from api import (
    _detect_metallic_resonance_artifact,
    _quality_candidate_score,
    _should_accept_text_qc_candidate,
    _text_qc_tokens,
)


class ApiTextQcTokenTest(unittest.TestCase):
    def test_cjk_uses_character_tokens(self):
        self.assertEqual(_text_qc_tokens("你好，world 123"), ["你", "好", "world", "123"])

    def test_latin_languages_keep_word_tokens(self):
        self.assertEqual(
            _text_qc_tokens("Kumusta, kaibigan! Hindi 123."),
            ["kumusta", "kaibigan", "hindi", "123"],
        )

    def test_thai_does_not_require_spaces(self):
        tokens = _text_qc_tokens("สวัสดี")
        self.assertGreaterEqual(len(tokens), 3)
        self.assertEqual("".join(tokens), "สวัสดี")

    def test_candidate_must_improve_text_without_worse_audio(self):
        current = {"coverage": 0.40}
        candidate = {"coverage": 0.80}
        self.assertTrue(_should_accept_text_qc_candidate(current, candidate, 1, 0, False))
        self.assertFalse(_should_accept_text_qc_candidate(current, candidate, 0, 1, False))
        self.assertFalse(_should_accept_text_qc_candidate(current, candidate, 1, 0, True))
        self.assertFalse(
            _should_accept_text_qc_candidate({"coverage": 0.60}, {"coverage": 0.63}, 1, 0, False)
        )

    def test_quality_score_prioritizes_metallic_resonance(self):
        clean = _quality_candidate_score([], 0.0, 1.0)
        metallic = _quality_candidate_score(["metallic_resonance"], 0.0, 1.0)
        self.assertGreater(metallic, clean)

    def test_metallic_detector_requires_sustained_narrow_band(self):
        sample_rate = 24000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        tone = 0.2 * np.sin(2 * np.pi * 4200 * t)
        self.assertEqual(
            _detect_metallic_resonance_artifact(tone, sample_rate),
            ["metallic_resonance"],
        )
        rng = np.random.default_rng(3)
        broadband = 0.12 * rng.normal(size=t.size).astype(np.float32)
        self.assertEqual(_detect_metallic_resonance_artifact(broadband, sample_rate), [])


if __name__ == "__main__":
    unittest.main()
