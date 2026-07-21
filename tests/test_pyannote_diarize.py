import unittest

from api import _assign_speakers, _best_overlap_speaker


TURNS = [
    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
    {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
]


class BestOverlapSpeakerTest(unittest.TestCase):
    def test_max_overlap_wins(self):
        # 0.5s overlap with SPEAKER_00, 1.5s with SPEAKER_01.
        self.assertEqual(_best_overlap_speaker(1.5, 3.5, TURNS), "SPEAKER_01")

    def test_exact_tie_returns_none(self):
        self.assertIsNone(_best_overlap_speaker(1.0, 3.0, TURNS))

    def test_no_overlap_returns_none(self):
        self.assertIsNone(_best_overlap_speaker(5.0, 6.0, TURNS))

    def test_empty_turns_returns_none(self):
        self.assertIsNone(_best_overlap_speaker(0.0, 1.0, []))
        self.assertIsNone(_best_overlap_speaker(0.0, 1.0, None))

    def test_invalid_span_returns_none(self):
        self.assertIsNone(_best_overlap_speaker(None, 1.0, TURNS))
        self.assertIsNone(_best_overlap_speaker(2.0, 2.0, TURNS))


class AssignSpeakersTest(unittest.TestCase):
    def test_words_assigned_and_segment_uses_dominant_word_speaker(self):
        segments = [
            {
                "start": 0.0,
                "end": 4.0,
                "text": "hello world again",
                "words": [
                    {"start": 0.0, "end": 1.0, "word": "hello"},
                    {"start": 1.0, "end": 1.8, "word": "world"},
                    {"start": 2.2, "end": 3.0, "word": "again"},
                ],
            }
        ]
        _assign_speakers(segments, TURNS)
        words = segments[0]["words"]
        self.assertEqual(words[0]["speaker"], "SPEAKER_00")
        self.assertEqual(words[1]["speaker"], "SPEAKER_00")
        self.assertEqual(words[2]["speaker"], "SPEAKER_01")
        # SPEAKER_00 dominates 2 of 3 words.
        self.assertEqual(segments[0]["speaker"], "SPEAKER_00")

    def test_wordless_segment_falls_back_to_segment_overlap(self):
        segments = [{"start": 2.5, "end": 3.5, "text": "again", "words": []}]
        _assign_speakers(segments, TURNS)
        self.assertEqual(segments[0]["speaker"], "SPEAKER_01")

    def test_no_overlap_words_yield_none_speakers(self):
        segments = [
            {
                "start": 5.0,
                "end": 6.0,
                "text": "later",
                "words": [{"start": 5.0, "end": 6.0, "word": "later"}],
            }
        ]
        _assign_speakers(segments, TURNS)
        self.assertIsNone(segments[0]["words"][0]["speaker"])
        self.assertIsNone(segments[0]["speaker"])

    def test_empty_turns_assign_none_everywhere(self):
        segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "words": [{"start": 0.0, "end": 1.0, "word": "hello"}],
            }
        ]
        _assign_speakers(segments, [])
        self.assertIsNone(segments[0]["words"][0]["speaker"])
        self.assertIsNone(segments[0]["speaker"])

    def test_empty_segments_is_noop(self):
        self.assertEqual(_assign_speakers([], TURNS), [])
        self.assertEqual(_assign_speakers(None, TURNS), None)


if __name__ == "__main__":
    unittest.main()
