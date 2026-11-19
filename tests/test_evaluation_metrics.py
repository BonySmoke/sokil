"""
What the evaluation counts.

The matching rule carries the most weight: it decides what counts as finding a
landing, and a loose one flatters any detector. One detection per labelled
landing, closest first — otherwise a detector that fires five times around every
landing would score perfectly.
"""

import pytest

from sokil.evaluation.metrics import (
    by_proximity,
    detection_counts,
    match_landings,
    position_errors,
    proximity_ranks,
    strip_of,
    verdict_counts,
)


class TestMatching:
    def test_an_exact_frame_matches(self):
        matches, missed, false_positives = match_landings([100], [100])

        assert [m.labelled_frame for m in matches] == [100]
        assert (missed, false_positives) == ([], [])

    def test_within_the_window_matches(self):
        matches, _, _ = match_landings([102], [100], window=2)

        assert matches[0].frame_error == 2

    def test_just_outside_the_window_does_not(self):
        matches, missed, false_positives = match_landings([103], [100], window=2)

        assert matches == []
        assert (missed, false_positives) == ([100], [103])

    def test_a_labelled_landing_is_claimed_once(self):
        # Three detections round one landing: one match, two false positives.
        matches, missed, false_positives = match_landings(
            [99, 100, 101], [100], window=2
        )

        assert len(matches) == 1
        assert false_positives == [99, 101]
        assert missed == []

    def test_the_closest_detection_wins(self):
        matches, _, false_positives = match_landings([98, 101], [100], window=2)

        assert matches[0].detected_frame == 101
        assert false_positives == [98]

    def test_two_landings_each_take_their_own(self):
        matches, missed, false_positives = match_landings([100, 200], [100, 200])

        assert len(matches) == 2
        assert (missed, false_positives) == ([], [])

    def test_neighbouring_landings_are_not_both_given_one_detection(self):
        matches, missed, _ = match_landings([101], [100, 102], window=2)

        assert len(matches) == 1
        assert len(missed) == 1

    def test_nothing_detected_is_all_missed(self):
        matches, missed, false_positives = match_landings([], [10, 20])

        assert matches == []
        assert missed == [10, 20]
        assert false_positives == []

    def test_nothing_labelled_is_all_false_positives(self):
        matches, missed, false_positives = match_landings([10, 20], [])

        assert matches == []
        assert missed == []
        assert false_positives == [10, 20]

    def test_matches_come_out_in_labelled_order(self):
        matches, _, _ = match_landings([200, 100], [100, 200])

        assert [m.labelled_frame for m in matches] == [100, 200]


class TestDetectionCounts:
    def test_everything_found_and_nothing_invented(self):
        counts = detection_counts([10, 20], [10, 20])

        assert (counts.precision, counts.recall, counts.f1) == (1.0, 1.0, 1.0)

    def test_half_the_detections_are_wrong(self):
        counts = detection_counts([10, 20, 500, 600], [10, 20])

        assert counts.precision == 0.5
        assert counts.recall == 1.0

    def test_half_the_landings_are_missed(self):
        counts = detection_counts([10], [10, 20])

        assert counts.precision == 1.0
        assert counts.recall == 0.5

    def test_nothing_detected_has_no_precision_to_speak_of(self):
        counts = detection_counts([], [10])

        assert counts.precision is None
        assert counts.recall == 0.0
        assert counts.f1 is None

    def test_nothing_labelled_has_no_recall_to_speak_of(self):
        counts = detection_counts([10], [])

        assert counts.recall is None

    def test_the_counts_add_up(self):
        counts = detection_counts([10, 20, 500], [10, 20, 30])

        assert counts.labelled == 3
        assert counts.detected == 3
        assert counts.matched + counts.missed == counts.labelled
        assert counts.matched + counts.false_positives == counts.detected


class TestPositionErrors:
    def test_a_perfect_position_has_no_error(self):
        result = position_errors([((1.0, 2.0), (1.0, 2.0))])

        assert result["median_cm"] == 0.0

    def test_the_error_is_a_distance_in_centimetres(self):
        result = position_errors([((1.0, 2.0), (1.03, 2.04))])

        assert result["median_cm"] == pytest.approx(5.0, abs=0.1)

    def test_nothing_found_reports_nothing_rather_than_zero(self):
        result = position_errors([])

        assert result["count"] == 0
        assert result["median_cm"] is None


class TestStrips:
    def test_the_court_is_cut_into_equal_strips_along_its_length(self):
        assert [strip_of(y, 13.4, 4) for y in (0.1, 3.4, 6.8, 13.3)] == [0, 1, 2, 3]

    def test_a_landing_past_a_baseline_belongs_to_the_end_strip(self):
        # An out call lands just beyond the line, as far from the camera as
        # the in calls beside it.
        assert strip_of(-0.2, 13.4, 4) == 0
        assert strip_of(13.6, 13.4, 4) == 3


class TestProximityRanks:
    def test_the_strip_lowest_in_the_frame_is_the_nearest(self):
        assert proximity_ranks([400.0, 300.0, 200.0, 100.0]) == [0, 1, 2, 3]

    def test_a_camera_at_the_other_end_reverses_the_order(self):
        assert proximity_ranks([100.0, 200.0, 300.0, 400.0]) == [3, 2, 1, 0]


class TestByProximity:
    def test_recall_and_error_are_pooled_per_strip(self):
        landings = [
            {"proximity": 0, "matched": True, "error_cm": 2.0},
            {"proximity": 0, "matched": True, "error_cm": 4.0},
            {"proximity": 1, "matched": True, "error_cm": 20.0},
            {"proximity": 1, "matched": False, "error_cm": None},
        ]

        near, far = by_proximity(landings, bins=2)

        assert (near["labelled"], near["recall"], near["median_cm"]) == (2, 1.0, 3.0)
        assert (far["labelled"], far["recall"], far["median_cm"]) == (2, 0.5, 20.0)

    def test_a_found_landing_with_no_position_counts_for_recall_only(self):
        (strip,) = by_proximity(
            [{"proximity": 0, "matched": True, "error_cm": None}], bins=1
        )

        assert (strip["recall"], strip["positions"], strip["median_cm"]) == (
            1.0,
            0,
            None,
        )

    def test_unranked_landings_are_left_out(self):
        (strip,) = by_proximity(
            [{"proximity": None, "matched": True, "error_cm": 1.0}], bins=1
        )

        assert strip["labelled"] == 0
        assert strip["recall"] is None


class TestVerdictCounts:
    def test_every_call_right(self):
        result = verdict_counts([("in", "in"), ("out", "out")])

        assert result["accuracy"] == 1.0

    def test_the_two_mistakes_are_kept_apart(self):
        # Calling in on an out shuttle hands the point to the wrong player; it
        # is not the same error as the reverse.
        result = verdict_counts([("in", "out"), ("out", "in"), ("in", "in")])

        assert result["called_in_was_out"] == 1
        assert result["called_out_was_in"] == 1
        assert result["accuracy"] == pytest.approx(1 / 3, abs=1e-4)

    def test_an_unknown_truth_is_not_counted(self):
        # A landing the footage cannot resolve cannot be got right or wrong.
        result = verdict_counts([("in", "unknown"), ("in", "in")])

        assert result["count"] == 1

    def test_an_unjudged_call_is_not_counted(self):
        result = verdict_counts([(None, "in"), ("out", "out")])

        assert result["count"] == 1

    def test_nothing_to_judge_reports_nothing(self):
        result = verdict_counts([])

        assert result["count"] == 0
        assert result["accuracy"] is None
