"""
The shuttle's own measurements: the smoothed direction, and finding the cork in
the detection box from the pixels.
"""

import numpy as np
import pytest

from sokil.shuttle import DirectionTracker, estimate_cork_position


class TestDirectionTracker:
    def test_the_first_update_has_nothing_to_compare_against(self):
        assert DirectionTracker().update(10, 10) == (0.0, 0.0)

    def test_alpha_one_follows_the_latest_step_exactly(self):
        tracker = DirectionTracker(alpha=1.0)
        tracker.update(0, 0)

        assert tracker.update(5, -3) == (5.0, -3.0)

    def test_smoothing_lags_a_sudden_reversal(self):
        tracker = DirectionTracker(alpha=0.5)
        tracker.update(0, 0)
        tracker.update(10, 0)  # vx_ema = 5
        vx, _ = tracker.update(0, 0)  # raw -10, so 0.5 * -10 + 0.5 * 5

        assert vx == pytest.approx(-2.5)

    def test_a_steady_flight_converges_on_its_true_velocity(self):
        tracker = DirectionTracker(alpha=0.5)
        for i in range(30):
            tracker.update(4 * i, 3 * i)

        assert tracker.velocity() == pytest.approx((4.0, 3.0), abs=1e-6)

    def test_the_direction_is_a_unit_vector(self):
        tracker = DirectionTracker(alpha=1.0)
        tracker.update(0, 0)
        tracker.update(4, 3)

        assert tracker.direction() == pytest.approx((0.8, 0.6), abs=1e-6)

    def test_a_resting_shuttle_has_a_direction_rather_than_a_zero_division(self):
        tracker = DirectionTracker()
        tracker.update(5, 5)
        tracker.update(5, 5)

        assert tracker.direction() == (0.0, 0.0)


class TestEstimateCorkPosition:
    HEIGHT, WIDTH = 80, 120

    def background(self, value: int = 60) -> np.ndarray:
        return np.full((self.HEIGHT, self.WIDTH), value, dtype=np.uint8)

    def frame_with_shuttle(self, x1, y1, x2, y2, brightness=230) -> np.ndarray:
        frame = self.background()
        frame[y1:y2, x1:x2] = brightness
        return frame

    def test_finds_the_leading_edge_of_the_shuttle(self):
        # A bright blob at x 40..60, travelling right: the cork is the right
        # end of it, not its centre.
        history = [self.background() for _ in range(5)]
        frame = self.frame_with_shuttle(40, 35, 60, 45)

        cork = estimate_cork_position(
            frame, history, (38, 33, 62, 47), velocity=(10.0, 0.0)
        )

        assert cork is not None
        assert cork[0] > 55
        assert 35 <= cork[1] <= 45

    def test_the_leading_edge_follows_the_flight_direction(self):
        history = [self.background() for _ in range(5)]
        frame = self.frame_with_shuttle(40, 35, 60, 45)
        box = (38, 33, 62, 47)

        rightwards = estimate_cork_position(frame, history, box, velocity=(10.0, 0.0))
        leftwards = estimate_cork_position(frame, history, box, velocity=(-10.0, 0.0))

        assert rightwards[0] > leftwards[0]

    def test_a_shuttle_over_a_painted_line_is_still_found(self):
        # the background carries a bright stripe; the median votes it into the
        # background, so only the shuttle is left as a difference
        marked = self.background()
        marked[38:42, :] = 200
        history = [marked.copy() for _ in range(5)]
        frame = marked.copy()
        frame[35:45, 40:60] = 250

        cork = estimate_cork_position(
            frame, history, (38, 33, 62, 47), velocity=(10.0, 0.0)
        )

        assert cork is not None
        assert cork[0] > 55

    def test_too_little_history_gives_no_measurement(self):
        frame = self.frame_with_shuttle(40, 35, 60, 45)

        assert (
            estimate_cork_position(
                frame, [self.background()], (38, 33, 62, 47), velocity=(10.0, 0.0)
            )
            is None
        )

    def test_a_box_smaller_than_three_pixels_gives_no_measurement(self):
        frame = self.frame_with_shuttle(40, 35, 60, 45)
        history = [self.background() for _ in range(5)]

        assert (
            estimate_cork_position(frame, history, (40, 40, 42, 42), (10.0, 0.0))
            is None
        )

    def test_a_shuttle_indistinguishable_from_the_background_gives_nothing(self):
        history = [self.background() for _ in range(5)]

        assert (
            estimate_cork_position(
                self.background(), history, (38, 33, 62, 47), (10.0, 0.0)
            )
            is None
        )

    def test_a_resting_shuttle_has_no_direction_to_project_along(self):
        history = [self.background() for _ in range(5)]
        frame = self.frame_with_shuttle(40, 35, 60, 45)

        assert (
            estimate_cork_position(frame, history, (38, 33, 62, 47), (0.0, 0.0)) is None
        )

    def test_history_frames_of_another_size_are_ignored(self):
        # a resized clip mid-history would otherwise blow up the median stack
        history = [np.zeros((10, 10), dtype=np.uint8) for _ in range(5)]
        frame = self.frame_with_shuttle(40, 35, 60, 45)

        assert (
            estimate_cork_position(frame, history, (38, 33, 62, 47), (10.0, 0.0))
            is None
        )

    def test_a_box_reaching_past_the_frame_edge_is_clamped(self):
        history = [self.background() for _ in range(5)]
        frame = self.frame_with_shuttle(100, 35, 119, 45)

        cork = estimate_cork_position(
            frame, history, (98, 33, 200, 47), velocity=(10.0, 0.0)
        )

        assert cork is not None
        assert cork[0] < self.WIDTH
