"""
The geometry and motion helpers in sokil.util.

These are the arithmetic the whole pipeline stands on — where the cork is, how
far the shuttle turned, how fast it was going — so they are tested against
hand-computable inputs rather than recorded output.
"""

import cv2
import numpy as np
import pytest

from sokil.util import (
    Undistorter,
    angle_between,
    cross_2d,
    draw_dashed_line,
    extract_contours,
    fig_to_numpy,
    find_contour_intersection,
    fit_velocity,
    infinite_line_points,
    is_significant_direction_change,
    is_significant_speed_change,
    is_speed_decreasing,
    laplacian_blur_metric,
    line_intersection,
    signed_angle_change,
    speed_estimation,
    zoom_inset,
    zoom_on_object,
)


class TestCross2d:
    def test_perpendicular_unit_vectors_give_one(self):
        assert cross_2d((1, 0), (0, 1)) == 1.0

    def test_sign_flips_with_the_order(self):
        assert cross_2d((0, 1), (1, 0)) == -1.0

    def test_parallel_vectors_give_zero(self):
        assert cross_2d((2, 4), (1, 2)) == 0.0


class TestLineIntersection:
    def test_crossing_segments_meet_in_the_middle(self):
        assert line_intersection((0, 0), (10, 10), (0, 10), (10, 0)) == (5.0, 5.0)

    def test_parallel_segments_do_not_meet(self):
        assert line_intersection((0, 0), (10, 0), (0, 5), (10, 5)) is None

    def test_intersection_beyond_the_segment_ends_is_rejected(self):
        # The infinite lines cross at (10, 0), but that is past the end of both.
        assert line_intersection((0, 0), (5, 0), (10, -5), (10, -1)) is None


class TestFindContourIntersection:
    def test_returns_the_crossing_nearest_the_direction_end(self):
        # A square, with a ray shot from its centre out to the right: the ray
        # crosses the right edge, and that is the crossing nearest its end.
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)

        crossing = find_contour_intersection(square, (5, 5), (100, 5))

        assert crossing == (10, 5)

    def test_returns_none_when_the_ray_misses(self):
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)

        assert find_contour_intersection(square, (50, 50), (100, 50)) is None


class TestSignedAngleChange:
    def test_straight_on_is_no_change(self):
        assert signed_angle_change((0, 0), (1, 0), (2, 0)) == pytest.approx(0.0)

    def test_a_left_turn_is_positive(self):
        assert signed_angle_change((0, 0), (1, 0), (1, 1)) == pytest.approx(90.0)

    def test_a_right_turn_is_negative(self):
        assert signed_angle_change((0, 0), (1, 0), (1, -1)) == pytest.approx(-90.0)

    def test_a_repeated_point_reports_no_change_rather_than_dividing_by_zero(self):
        assert signed_angle_change((5, 5), (5, 5), (9, 9)) == 0.0


class TestFitVelocity:
    def test_recovers_a_constant_velocity(self):
        times = [0, 1, 2, 3]
        points = [(0, 0), (3, -2), (6, -4), (9, -6)]

        assert fit_velocity(times, points) == pytest.approx([3.0, -2.0])

    def test_uses_frame_numbers_not_sample_index(self):
        # A gap in the track: frames 0, 1 and 5. Fitting against the index would
        # report 1 px/sample; against the frame axis it is the true 1 px/frame.
        assert fit_velocity([0, 1, 5], [(0, 0), (1, 0), (5, 0)]) == pytest.approx(
            [1.0, 0.0]
        )

    def test_a_single_sample_has_no_velocity(self):
        assert fit_velocity([0], [(0, 0)]) is None

    def test_repeated_timestamps_have_no_velocity(self):
        assert fit_velocity([4, 4, 4], [(0, 0), (1, 1), (2, 2)]) is None


class TestAngleBetween:
    def test_same_direction_is_zero(self):
        assert angle_between((1, 0), (5, 0)) == pytest.approx(0.0)

    def test_reversal_is_one_hundred_and_eighty(self):
        assert angle_between((1, 0), (-1, 0)) == pytest.approx(180.0)

    def test_perpendicular_is_ninety(self):
        assert angle_between((0, 3), (7, 0)) == pytest.approx(90.0)

    def test_a_zero_vector_is_degenerate_not_an_error(self):
        assert angle_between((0, 0), (1, 1)) == 0.0


class TestSpeedHeuristics:
    def test_a_short_history_is_never_a_significant_change(self):
        assert is_significant_speed_change([10.0, 200.0]) is False

    def test_an_outlier_last_frame_is_significant(self):
        assert is_significant_speed_change([10.0, 10.2, 9.9, 10.1, 400.0]) is True

    def test_a_steady_history_is_not(self):
        assert is_significant_speed_change([10.0, 11.0, 9.0, 10.5, 10.0]) is False

    def test_a_constant_history_has_no_spread_to_judge_against(self):
        assert is_significant_speed_change([10.0, 10.0, 10.0, 10.0]) is False

    def test_speed_decreasing_only_on_the_last_frame(self):
        assert is_speed_decreasing([1.0, 2.0, 3.0, 2.0]) is True

    def test_an_earlier_dip_disqualifies_the_last_only_check(self):
        assert is_speed_decreasing([3.0, 1.0, 2.0, 1.5]) is False

    def test_without_last_only_any_dip_counts(self):
        assert is_speed_decreasing([3.0, 1.0, 2.0, 5.0], last_only=False) is True

    def test_direction_change_needs_one_candidate_beyond_the_threshold(self):
        assert (
            is_significant_direction_change(10.0, [12.0, 15.0], threshold=20) is False
        )
        assert is_significant_direction_change(10.0, [12.0, 90.0], threshold=20) is True


class TestSpeedEstimation:
    def test_a_track_shorter_than_two_points_has_no_speed(self):
        assert speed_estimation([(0, 0)], [100.0], 100.0, 30.0) == 0.0

    def test_scales_the_pixel_distance_by_the_area_ratio(self):
        # 1000 px over 2 frames at 1 fps, with the box exactly at the reference
        # area, is 1000 * 0.001 m in 2 s -> 0.5 m/s -> 1.8 km/h.
        speed = speed_estimation([(0, 0), (1000, 0)], [50.0, 50.0], 50.0, 1.0)

        assert speed == pytest.approx(1.8)

    def test_a_smaller_box_reads_as_further_away_and_therefore_faster(self):
        near = speed_estimation([(0, 0), (1000, 0)], [50.0, 50.0], 50.0, 1.0)
        far = speed_estimation([(0, 0), (1000, 0)], [25.0, 25.0], 50.0, 1.0)

        assert far > near


class TestInfiniteLinePoints:
    def test_a_vertical_line_spans_the_frame(self):
        # theta = 0 is the line x = rho.
        start, end = infinite_line_points(rho=100.0, theta=0.0, span=500)

        assert start[0] == pytest.approx(100, abs=1)
        assert end[0] == pytest.approx(100, abs=1)
        assert abs(start[1] - end[1]) == pytest.approx(1000, abs=2)


class TestLaplacianBlurMetric:
    def test_a_flat_patch_has_no_detail(self, frame):
        flat = np.full_like(frame, 90)

        assert laplacian_blur_metric(flat, (10, 10, 40, 40)) == pytest.approx(0.0)

    def test_an_edge_has_more_detail_than_a_flat_patch(self, frame):
        # (0, 0, 160, 60) covers the painted horizontal line; (0, 60, 40, 100)
        # is bare background.
        assert laplacian_blur_metric(frame, (0, 0, 160, 60)) > laplacian_blur_metric(
            frame, (0, 60, 40, 100)
        )

    def test_a_box_off_the_frame_is_zero_rather_than_an_error(self, frame):
        assert laplacian_blur_metric(frame, (500, 500, 600, 600)) == 0.0

    def test_a_box_reaching_past_the_edge_is_clamped(self, frame):
        assert laplacian_blur_metric(frame, (-20, -20, 200, 200)) > 0.0


class TestZoom:
    def test_zoom_on_object_keeps_the_frame_size(self, frame):
        zoomed = zoom_on_object(frame, (60, 40, 20, 20))

        assert zoomed.shape == frame.shape

    def test_zoom_inset_keeps_the_frame_size_and_changes_the_pixels(self, frame):
        out = zoom_inset(frame, (80, 60))

        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)

    def test_zoom_inset_leaves_the_frame_alone_when_the_centre_is_off_it(self, frame):
        # The crop is clamped into the frame, so an off-frame centre still
        # produces a valid inset rather than an empty crop.
        out = zoom_inset(frame, (10_000, 10_000))

        assert out.shape == frame.shape


class TestUndistorter:
    def test_zero_distortion_leaves_the_image_essentially_unchanged(self, frame):
        height, width = frame.shape[:2]
        K = np.array(
            [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64
        )

        undistorter = Undistorter(K, np.zeros(5, dtype=np.float64))
        out = undistorter(frame)

        assert out.shape == frame.shape
        assert np.abs(out.astype(int) - frame.astype(int)).mean() < 1.0

    def test_the_remap_tables_are_built_once_and_reused(self, frame):
        height, width = frame.shape[:2]
        K = np.array(
            [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64
        )
        undistorter = Undistorter(K, np.zeros(5, dtype=np.float64))

        undistorter(frame)
        first = undistorter.map_x
        undistorter(frame)

        assert undistorter.map_x is first


class TestFigToNumpy:
    def test_rasterizes_a_figure_to_rgba_pixels(self):
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=(2, 1), dpi=50)
        ax.plot([0, 1], [0, 1])

        raster = fig_to_numpy(fig)

        assert raster.ndim == 3
        assert raster.shape[2] == 4
        assert raster.shape[:2] == (50, 100)


class TestDrawDashedLine:
    def test_draws_between_two_points_inside_the_frame(self, frame):
        out = frame.copy()

        draw_dashed_line(out, (10, 10), (150, 10), (0, 0, 255), thickness=2)

        assert not np.array_equal(out, frame)

    def test_a_line_with_both_ends_off_the_frame_draws_nothing(self, frame):
        out = frame.copy()

        draw_dashed_line(out, (-500, -500), (-400, -400), (0, 0, 255))

        assert np.array_equal(out, frame)

    def test_an_endpoint_far_outside_is_clipped_rather_than_walked(self, frame):
        out = frame.copy()

        draw_dashed_line(out, (80, 60), (100_000, 60), (0, 0, 255), thickness=2)

        assert not np.array_equal(out, frame)

    def test_a_zero_length_line_draws_nothing(self, frame):
        out = frame.copy()

        draw_dashed_line(out, (80, 60), (80, 60), (0, 0, 255))

        assert np.array_equal(out, frame)

    def test_the_gaps_leave_the_background_showing(self, frame):
        out = frame.copy()

        draw_dashed_line(
            out, (0, 100), (159, 100), (0, 0, 255), thickness=1, dash_len=4, gap_len=8
        )

        row = out[100]
        painted = (row == (0, 0, 255)).all(axis=1)
        assert painted.any()
        assert not painted.all()


class TestExtractContours:
    def test_finds_the_object_that_appeared_against_a_learnt_background(self):
        background_subtractor = cv2.createBackgroundSubtractorMOG2()
        empty = np.full((120, 160, 3), 30, dtype=np.uint8)
        for _ in range(30):
            extract_contours(empty, background_subtractor)

        moved = empty.copy()
        moved[40:80, 40:80] = 240

        contours = extract_contours(moved, background_subtractor)

        assert contours
        assert max(cv2.contourArea(contour) for contour in contours) > 100

    def test_an_unchanged_frame_has_nothing_moving_in_it(self):
        background_subtractor = cv2.createBackgroundSubtractorMOG2()
        empty = np.full((120, 160, 3), 30, dtype=np.uint8)
        for _ in range(30):
            extract_contours(empty, background_subtractor)

        assert extract_contours(empty, background_subtractor) == ()
