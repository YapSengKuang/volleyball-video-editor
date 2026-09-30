import unittest

from rallycondense.court import point_in_polygon, side_of_net
from rallycondense.evaluate import report
from rallycondense.segment import Second, Segment, clean_segments, detect_rallies


def burst(start: int, end: int, length: int = 40) -> list[Second]:
    samples = [Second() for _ in range(length)]
    for index in range(start, end):
        samples[index] = Second(motion=50, ball_count=1, ball_speed=0.8)
    return samples


class SegmentTests(unittest.TestCase):
    def test_motion_burst_becomes_one_padded_rally(self) -> None:
        raw = detect_rallies(burst(10, 18))
        clips = clean_segments(raw, duration=40)
        self.assertEqual(len(clips), 1)
        self.assertAlmostEqual(clips[0].start, 8.5)
        self.assertGreater(clips[0].end, 18)

    def test_short_touch_is_dropped(self) -> None:
        self.assertEqual(detect_rallies(burst(10, 12)), [])

    def test_small_gap_is_merged(self) -> None:
        raw = [(10.0, 16.0), (18.0, 24.0)]
        clips = clean_segments(raw, duration=40, merge_gap=3)
        self.assertEqual(len(clips), 1)

    def test_serve_ready_waits_for_play(self) -> None:
        samples = [Second() for _ in range(20)]
        samples[5] = Second(server_ready=True, motion=1)
        for index in range(8, 14):
            samples[index] = Second(motion=40, ball_count=1, ball_speed=0.6)
        raw = detect_rallies(samples)
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0][0], 5)


class CourtTests(unittest.TestCase):
    def test_point_inside_polygon(self) -> None:
        corners = [(0.1, 0.1), (0.4, 0.1), (0.4, 0.9), (0.1, 0.9)]
        self.assertTrue(point_in_polygon(0.2, 0.5, corners))
        self.assertFalse(point_in_polygon(0.8, 0.5, corners))

    def test_net_splits_the_court(self) -> None:
        net = [(0.5, 0.1), (0.5, 0.9)]
        self.assertEqual(side_of_net(0.2, 0.5, net), -side_of_net(0.8, 0.5, net))
        self.assertNotEqual(side_of_net(0.2, 0.5, net), 0)


class EvalTests(unittest.TestCase):
    def test_perfect_labels_meet_the_bar(self) -> None:
        rallies = [Segment(10, 20), Segment(30, 40)]
        result = report(rallies, rallies, duration=50)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["typical_boundary_error"], 0)
        self.assertTrue(result["meets_exit_bar"])

    def test_late_boundaries_are_measured(self) -> None:
        truth = [Segment(10, 20)]
        predicted = [Segment(12, 22)]
        result = report(predicted, truth, duration=30)
        self.assertAlmostEqual(result["mean_start_error"], 2)
        self.assertFalse(result["meets_exit_bar"])


if __name__ == "__main__":
    unittest.main()
