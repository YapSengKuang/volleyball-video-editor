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
from ball import Hit, detect_rallies, moving_hits, rallies_from_hits


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
        mask = court_mask(
            20,
            20,
            [
                {"x": 0.0, "y": 0.05},
                {"x": 0.45, "y": 0.05},
                {"x": 0.45, "y": 0.95},
                {"x": 0.0, "y": 0.95},
            ],
        )
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


class BallStateTests(unittest.TestCase):
    def test_gap_ends_the_rally_and_pads_it(self) -> None:
        fps = 10.0
        mask = np.zeros(200, dtype=bool)
        mask[30:80] = True
        clips = detect_rallies(mask, fps, duration=20.0)
        self.assertEqual(len(clips), 1)
        start, end = clips[0]
        self.assertAlmostEqual(start, 3.0 - 1.5)
        self.assertAlmostEqual(end, 7.9 + 2.0, places=1)

    def test_three_second_gap_splits_rallies(self) -> None:
        fps = 10.0
        mask = np.zeros(300, dtype=bool)
        mask[10:40] = True
        mask[80:120] = True
        clips = detect_rallies(mask, fps, duration=30.0)
        self.assertEqual(len(clips), 2)

    def test_slow_ball_does_not_extend_the_point(self) -> None:
        fast = [
            Hit(time_s=1.0 + index * 0.2, cx=0.1 + index * 0.08, cy=0.4, width_px=20, conf=0.8)
            for index in range(12)
        ]
        slow = [
            Hit(time_s=6.0 + index * 0.2, cx=0.5 + index * 0.004, cy=0.6, width_px=20, conf=0.8)
            for index in range(15)
        ]
        clips = rallies_from_hits(fast + slow, duration=12.0, corners=None)
        self.assertEqual(len(clips), 1)
        self.assertLess(clips[0][1], 5.5)

    def test_ball_outside_the_court_is_ignored(self) -> None:
        corners = [(0.0, 0.1), (0.4, 0.1), (0.4, 0.9), (0.0, 0.9)]
        outside = [
            Hit(time_s=1.0 + index * 0.25, cx=0.75 + (index % 2) * 0.12, cy=0.45, width_px=20, conf=0.9)
            for index in range(10)
        ]
        self.assertEqual(len(rallies_from_hits(outside, duration=8.0, corners=None)), 1)
        self.assertEqual(rallies_from_hits(outside, duration=8.0, corners=corners), [])


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
