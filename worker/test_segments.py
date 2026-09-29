import tempfile
import unittest

import numpy as np

from analyze import (
    DurationError,
    assert_duration,
    court_mask,
    detect_whistles,
    propose,
    propose_for_file,
)


def kept(segments, second: float) -> bool:
    return any(segment["keep"] and segment["start"] <= second < segment["end"] for segment in segments)


class SegmentTests(unittest.TestCase):
    def test_whistle_on_a_tone(self) -> None:
        rate = 16000
        samples = np.zeros(rate * 20, dtype=np.float32)
        count = int(0.3 * rate)
        offset = 5 * rate
        ticks = np.arange(count, dtype=np.float32) / rate
        samples[offset : offset + count] = 0.8 * np.sin(2 * np.pi * 3500 * ticks)
        times = detect_whistles(samples, rate)
        self.assertTrue(any(abs(stamp - 5.15) < 0.4 for stamp in times), times)

    def test_rallies_drop_the_gaps(self) -> None:
        duration = 60
        scores = np.zeros(duration)
        scores[10:18] = 50
        scores[28:36] = 50
        scores[48:55] = 50
        segments, warning = propose(scores, duration, had_audio=True, used_court=True)
        self.assertIsNone(warning)
        for second, keep in ((5, False), (14, True), (22, False), (32, True), (42, False), (51, True), (58, False)):
            self.assertEqual(kept(segments, second), keep, second)
        self.assertEqual(len(segments), 3)

    def test_short_touch_is_dropped(self) -> None:
        scores = np.zeros(30)
        scores[10:12] = 80
        segments, _warning = propose(scores, 30, had_audio=True, used_court=True)
        self.assertEqual(segments, [])

    def test_audio_alone_is_not_a_rally(self) -> None:
        scores = np.zeros(30)
        audio = np.zeros(30)
        audio[12] = 5
        segments, _warning = propose(scores, 30, audio_energy=audio, had_audio=True, used_court=True)
        self.assertEqual(segments, [])

    def test_quiet_contact_with_some_motion_counts(self) -> None:
        scores = np.zeros(30)
        scores[10:16] = 20
        audio = np.zeros(30)
        audio[10:16] = 5
        segments, _warning = propose(scores, 30, audio_energy=audio, had_audio=True, used_court=True)
        self.assertTrue(kept(segments, 12))

    def test_court_mask_keeps_the_near_court(self) -> None:
        mask = court_mask(20, 20, [(0.0, 0.05), (0.45, 0.05), (0.45, 0.95), (0.0, 0.95)])
        self.assertGreater(int(mask[:, 2].sum()), 10)
        self.assertEqual(int(mask[:, 18].sum()), 0)

    def test_silent_file_warns_and_uses_motion(self) -> None:
        scores = np.zeros(20)
        scores[4:10] = 40
        segments, warning = propose(scores, 20, had_audio=False, used_court=True)
        self.assertIn("No soundtrack", warning or "")
        self.assertTrue(kept(segments, 6))
        self.assertFalse(kept(segments, 0))

    def test_over_duration(self) -> None:
        with self.assertRaises(DurationError):
            assert_duration(10, limit=1)


class FixtureTests(unittest.TestCase):
    def test_fixture_file(self) -> None:
        from make_fixture import make_fixture

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/fixture.mov"
            make_fixture(path)
            segments, warning, duration = propose_for_file(path)
        self.assertIn("whole frame", warning or "")
        self.assertGreater(duration, 55)
        self.assertLess(duration, 65)
        for second, keep in ((1, False), (14, True), (24, False), (32, True), (43, False), (51, True)):
            self.assertEqual(kept(segments, second), keep, (second, segments))
        self.assertLess(
            sum(item["end"] - item["start"] for item in segments if item["keep"]),
            duration * 0.85,
        )


if __name__ == "__main__":
    unittest.main()
