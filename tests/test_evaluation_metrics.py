"""
What the evaluation counts.

The matching rule carries the most weight: it decides what counts as finding a
landing, and a loose one flatters any detector. One detection per labelled
landing, closest first — otherwise a detector that fires five times around every
landing would score perfectly.
"""

import pytest

from sokil.evaluation.metrics import (
    detection_counts,
    match_landings,
    position_errors,
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

    def test_errors_are_grouped_by_distance_along_the_court(self):
        # Near the camera and far from it are different regimes, so an average
        # over both describes neither.
        result = position_errors(
            [((1.0, 1.0), (1.01, 1.0)), ((1.0, 11.0), (1.30, 11.0))]
        )

        assert result["by_distance"]["0-3 m"]["median_cm"] == pytest.approx(
            1.0, abs=0.1
        )
        assert result["by_distance"]["9+ m"]["median_cm"] == pytest.approx(
            30.0, abs=0.1
        )

    def test_nothing_found_reports_nothing_rather_than_zero(self):
        result = position_errors([])

        assert result["count"] == 0
        assert result["median_cm"] is None


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
