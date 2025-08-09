"""
The court-line helpers in sokil.util: Hesse-form conversion, clustering the
detected lines into physical court lines, and matching them to the court model.
"""

from typing import ClassVar

import numpy as np
import pytest
from helpers import hesse_line

from sokil.util import (
    _family_assignments,
    _hesse_average,
    classify_line_clusters,
    consensus_lines,
    correspondences_from_matches,
    filter_outer_lines,
    get_court_rectangle,
    get_intersection_centers,
    get_outermost_lines_cluster,
    identify_line_candidates,
    intersection,
    intersection_of,
    line_reference_coord,
    merge_stripe_edges,
    segment_by_angle_spectralclustering,
    segment_to_hesse,
    segmented_intersections,
    snap_to_intersection,
    split_line_families,
)

IMG_SHAPE = (720, 1280)

# Hesse normal form names a line by its normal's angle, so a VERTICAL line has
# a horizontal normal (theta = 0) and a HORIZONTAL line a vertical one.
VERTICAL_THETA = 0.0
HORIZONTAL_THETA = np.pi / 2


def hough(rho: float, theta: float) -> np.ndarray:
    """One line in the (N, 1, 2) shape cv2.HoughLines returns."""
    return np.array([[rho, theta]], dtype=np.float64)


class TestSegmentToHesse:
    def test_a_horizontal_segment_gets_a_vertical_normal(self):
        rho, theta = segment_to_hesse(0, 40, 100, 40)

        assert theta == pytest.approx(HORIZONTAL_THETA)
        assert rho == pytest.approx(40.0)

    def test_a_vertical_segment_gets_a_horizontal_normal(self):
        rho, theta = segment_to_hesse(70, 0, 70, 200)

        assert theta % np.pi == pytest.approx(0.0, abs=1e-9)
        assert abs(rho) == pytest.approx(70.0)

    def test_theta_stays_within_half_a_turn(self):
        for x2, y2 in ((10, 10), (-10, 10), (10, -10), (-10, -10)):
            _, theta = segment_to_hesse(0, 0, x2, y2)
            assert 0 <= theta < np.pi


class TestSplitLineFamilies:
    def test_splits_by_orientation(self):
        vertical_line = hesse_line(300, VERTICAL_THETA)
        horizontal_line = hesse_line(200, HORIZONTAL_THETA)

        vertical, horizontal = split_line_families([vertical_line, horizontal_line])

        assert vertical == [vertical_line]
        assert horizontal == [horizontal_line]

    def test_a_line_leaning_forty_five_degrees_counts_as_horizontal(self):
        vertical, horizontal = split_line_families([hesse_line(10, np.pi / 4)])

        assert vertical == []
        assert len(horizontal) == 1


class TestLineReferenceCoord:
    def test_a_vertical_lines_reference_is_its_x(self):
        assert line_reference_coord(
            420.0, VERTICAL_THETA, IMG_SHAPE, vertical=True
        ) == pytest.approx(420.0)

    def test_a_horizontal_lines_reference_is_its_y(self):
        assert line_reference_coord(
            250.0, HORIZONTAL_THETA, IMG_SHAPE, vertical=False
        ) == pytest.approx(250.0)

    def test_the_reference_orders_parallel_lines(self):
        left = line_reference_coord(100.0, VERTICAL_THETA, IMG_SHAPE, vertical=True)
        right = line_reference_coord(400.0, VERTICAL_THETA, IMG_SHAPE, vertical=True)

        assert left < right


class TestIntersection:
    def test_a_vertical_and_a_horizontal_line_cross_at_their_coords(self):
        assert intersection([[300.0, VERTICAL_THETA]], [[200.0, HORIZONTAL_THETA]]) == [
            [300, 200]
        ]

    def test_parallel_lines_do_not_cross(self):
        assert (
            intersection([[300.0, VERTICAL_THETA]], [[500.0, VERTICAL_THETA]]) is None
        )

    def test_intersection_of_unwraps_the_point(self):
        assert intersection_of(
            hesse_line(300, VERTICAL_THETA), hesse_line(200, HORIZONTAL_THETA)
        ) == [300, 200]

    def test_intersection_of_parallel_line_dicts_is_none(self):
        assert (
            intersection_of(
                hesse_line(300, VERTICAL_THETA), hesse_line(400, VERTICAL_THETA)
            )
            is None
        )


class TestSegmentedIntersections:
    def test_crosses_every_line_of_one_group_with_every_line_of_the_other(self):
        verticals = [hough(100, VERTICAL_THETA), hough(300, VERTICAL_THETA)]
        horizontals = [hough(50, HORIZONTAL_THETA), hough(250, HORIZONTAL_THETA)]

        points = segmented_intersections([verticals, horizontals])

        assert sorted(tuple(p[0]) for p in points) == [
            (100, 50),
            (100, 250),
            (300, 50),
            (300, 250),
        ]

    def test_lines_within_one_group_are_never_crossed(self):
        verticals = [hough(100, VERTICAL_THETA), hough(300, VERTICAL_THETA)]

        assert segmented_intersections([verticals]) == []


class TestFilterOuterLines:
    def test_near_duplicates_collapse_onto_the_strongest(self):
        # HoughLines returns strongest first, so the 100.0 line represents the
        # group and the two lines beside it are absorbed.
        lines = np.array(
            [[[100.0, 0.0]], [[103.0, 0.0]], [[97.0, 0.0]], [[400.0, 0.0]]]
        )

        filtered = filter_outer_lines(lines)

        assert filtered.reshape(-1, 2)[:, 0].tolist() == [100.0, 400.0]

    def test_lines_at_different_angles_are_kept_apart(self):
        lines = np.array([[[100.0, 0.0]], [[100.0, HORIZONTAL_THETA]]])

        assert len(filter_outer_lines(lines)) == 2

    def test_no_lines_at_all_passes_through(self):
        assert filter_outer_lines(None) is None


class TestGetIntersectionCenters:
    def test_a_tight_cloud_collapses_to_its_mean(self):
        centers = get_intersection_centers([[[100, 100]], [[104, 100]], [[102, 102]]])

        assert len(centers) == 1
        assert centers[0] == pytest.approx([102.0, 100.667], abs=0.01)

    def test_distant_clouds_stay_separate(self):
        centers = get_intersection_centers([[[10, 10]], [[500, 500]]])

        assert len(centers) == 2

    def test_no_intersections_gives_nothing(self):
        assert get_intersection_centers([]) is None


class TestClassifyLineClusters:
    def test_returns_the_vertical_family_first(self):
        verticals = [hough(100, VERTICAL_THETA)]
        horizontals = [hough(100, HORIZONTAL_THETA)]

        assert classify_line_clusters(horizontals, verticals) == (
            verticals,
            horizontals,
        )
        assert classify_line_clusters(verticals, horizontals) == (
            verticals,
            horizontals,
        )


class TestSegmentByAngle:
    def test_a_single_line_is_returned_as_one_group(self):
        lines = [hough(10, 0.0)]

        assert segment_by_angle_spectralclustering(lines) == [lines]

    def test_a_small_set_falls_back_to_the_deterministic_split(self):
        verticals = [hough(100, 0.0), hough(300, 0.05)]
        horizontals = [hough(50, HORIZONTAL_THETA)]

        horizontal, vertical = segment_by_angle_spectralclustering(
            verticals + horizontals
        )

        assert horizontal == horizontals
        assert vertical == verticals

    def test_a_large_set_is_still_split_into_two_families(self):
        lines = [hough(50 * i, 0.01 * i) for i in range(1, 6)]
        lines += [hough(50 * i, HORIZONTAL_THETA + 0.01 * i) for i in range(1, 6)]

        clusters = segment_by_angle_spectralclustering(lines)

        assert len(clusters) == 2
        assert sum(len(cluster) for cluster in clusters) == len(lines)


class TestOutermostLines:
    def test_picks_the_extremes_by_image_position(self):
        lines = [hough(400, 0.0), hough(100, 0.0), hough(250, 0.0)]

        low, high = get_outermost_lines_cluster(lines, IMG_SHAPE)

        assert low[0][0] == 100
        assert high[0][0] == 400

    def test_a_single_line_has_no_pair(self):
        assert get_outermost_lines_cluster([hough(100, 0.0)], IMG_SHAPE) == (None, None)


class TestGetCourtRectangle:
    def test_four_lines_give_the_four_corners(self):
        horizontals = [hough(100, HORIZONTAL_THETA), hough(600, HORIZONTAL_THETA)]
        verticals = [hough(200, VERTICAL_THETA), hough(900, VERTICAL_THETA)]

        corners = get_court_rectangle(IMG_SHAPE, horizontals, verticals)

        assert corners.reshape(-1, 2).tolist() == [
            [200, 100],
            [900, 100],
            [900, 600],
            [200, 600],
        ]

    def test_too_few_lines_means_no_court_this_frame(self):
        horizontals = [hough(100, HORIZONTAL_THETA)]
        verticals = [hough(200, VERTICAL_THETA), hough(900, VERTICAL_THETA)]

        assert get_court_rectangle(IMG_SHAPE, horizontals, verticals) is None


class TestSnapToIntersection:
    # source entries are (label, x, y, confidence); targets are the detected
    # line intersections the keypoints are pulled onto.
    TARGETS: ClassVar[list[list[float]]] = [[100.0, 100.0], [800.0, 700.0]]

    def test_a_prediction_within_the_radius_moves_onto_the_intersection(self):
        snapped = snap_to_intersection(
            [(0, 102.0, 98.0, 0.9)], self.TARGETS, max_radius=30
        )

        assert (float(snapped[0][1]), float(snapped[0][2])) == (100.0, 100.0)

    def test_a_prediction_outside_the_radius_is_left_alone(self):
        snapped = snap_to_intersection(
            [(0, 400.0, 400.0, 0.9)], self.TARGETS, max_radius=30
        )

        assert (float(snapped[0][1]), float(snapped[0][2])) == (400.0, 400.0)

    def test_one_intersection_can_only_claim_one_prediction(self):
        snapped = snap_to_intersection(
            [(0, 100.0, 100.0, 0.9), (1, 101.0, 101.0, 0.9)],
            self.TARGETS,
            max_radius=30,
        )

        assert (float(snapped[0][1]), float(snapped[0][2])) == (100.0, 100.0)
        assert (float(snapped[1][1]), float(snapped[1][2])) == (101.0, 101.0)

    def test_the_label_and_confidence_survive_the_snap(self):
        snapped = snap_to_intersection(
            [(7, 102.0, 98.0, 0.42)], self.TARGETS, max_radius=30
        )

        assert snapped[0][0] == 7
        assert float(snapped[0][3]) == pytest.approx(0.42)

    def test_a_single_target_intersection(self):
        # k=1 makes KDTree.query return scalars rather than arrays, which the
        # match loop has to cope with
        snapped = snap_to_intersection(
            [(0, 102.0, 98.0, 0.9)], [[100.0, 100.0]], max_radius=30
        )

        assert (float(snapped[0][1]), float(snapped[0][2])) == (100.0, 100.0)

    def test_a_single_target_out_of_radius_leaves_the_prediction_alone(self):
        snapped = snap_to_intersection(
            [(0, 500.0, 500.0, 0.9)], [[100.0, 100.0]], max_radius=30
        )

        assert (float(snapped[0][1]), float(snapped[0][2])) == (500.0, 500.0)


class TestHesseAverage:
    def test_two_parallel_lines_average_to_the_line_between_them(self):
        rho, theta = _hesse_average(
            hesse_line(100, VERTICAL_THETA), hesse_line(110, VERTICAL_THETA)
        )

        assert rho == pytest.approx(105.0)
        assert theta == pytest.approx(VERTICAL_THETA)

    def test_a_pair_straddling_the_theta_wrap_stays_on_the_same_side(self):
        # The two painted edges of a near-vertical stripe can come back as
        # theta ~= 0 and theta ~= pi with opposite-signed rho. Their centreline
        # belongs between them, not mirrored through the origin.
        rho, theta = _hesse_average(
            hesse_line(300.0, 0.01), hesse_line(-310.0, np.pi - 0.01)
        )

        assert 0 <= theta < np.pi
        assert rho == pytest.approx(305.0, abs=1.0)


class TestMergeStripeEdges:
    def test_the_two_edges_of_one_stripe_become_its_centreline(self):
        lines = [hesse_line(100, VERTICAL_THETA), hesse_line(110, VERTICAL_THETA)]

        merged = merge_stripe_edges(
            lines, IMG_SHAPE, vertical=True, stripe_merge_eps=25
        )

        assert len(merged) == 1
        assert merged[0]["ref"] == pytest.approx(105.0)

    def test_distinct_court_lines_are_left_alone(self):
        lines = [hesse_line(100, VERTICAL_THETA), hesse_line(400, VERTICAL_THETA)]

        merged = merge_stripe_edges(
            lines, IMG_SHAPE, vertical=True, stripe_merge_eps=25
        )

        assert [line["ref"] for line in merged] == [100.0, 400.0]

    def test_a_merged_pair_is_closed_to_a_third_neighbour(self):
        # Without the "already paired" guard the third line would chain into
        # the pair and drag the centreline off the paint.
        lines = [
            hesse_line(100, VERTICAL_THETA),
            hesse_line(110, VERTICAL_THETA),
            hesse_line(118, VERTICAL_THETA),
        ]

        merged = merge_stripe_edges(
            lines, IMG_SHAPE, vertical=True, stripe_merge_eps=25
        )

        assert [round(line["ref"]) for line in merged] == [105, 118]


class TestConsensusLines:
    def test_keeps_the_line_that_persists_across_frames(self):
        per_frame = [
            [hesse_line(200, VERTICAL_THETA)],
            [hesse_line(203, VERTICAL_THETA)],
            [hesse_line(201, VERTICAL_THETA)],
        ]

        consensus = consensus_lines(per_frame, min_frame_fraction=0.4)

        assert len(consensus) == 1
        assert consensus[0]["ref"] == pytest.approx(201.0)

    def test_drops_a_line_seen_on_a_single_frame(self):
        per_frame = [
            [hesse_line(200, VERTICAL_THETA), hesse_line(900, VERTICAL_THETA)],
            [hesse_line(201, VERTICAL_THETA)],
            [hesse_line(202, VERTICAL_THETA)],
            [hesse_line(203, VERTICAL_THETA)],
        ]

        consensus = consensus_lines(per_frame, min_frame_fraction=0.4)

        assert [round(line["ref"]) for line in consensus] == [202]

    def test_no_lines_at_all_gives_no_consensus(self):
        assert consensus_lines([[], []]) == []

    def test_the_result_is_sorted_by_position(self):
        per_frame = [
            [hesse_line(500, VERTICAL_THETA), hesse_line(100, VERTICAL_THETA)],
            [hesse_line(501, VERTICAL_THETA), hesse_line(101, VERTICAL_THETA)],
        ]

        refs = [line["ref"] for line in consensus_lines(per_frame)]

        assert refs == sorted(refs)


class TestFamilyAssignments:
    def test_larger_assignments_come_first(self):
        assignments = _family_assignments(["a", "b", "c"], [1, 2, 3])

        assert len(assignments[0]) == 3
        assert len(assignments[-1]) == 2

    def test_every_assignment_preserves_order_on_both_sides(self):
        detected = ["a", "b", "c"]
        expected = [1, 2, 3, 4]

        for assignment in _family_assignments(detected, expected):
            lines = [line for line, _ in assignment]
            coords = [coord for _, coord in assignment]
            assert lines == sorted(lines, key=detected.index)
            assert coords == sorted(coords)

    def test_a_single_line_cannot_span_a_homography_grid(self):
        assert _family_assignments(["a"], [1]) == []


class TestIdentifyLineCandidates:
    def test_pairs_every_vertical_assignment_with_every_horizontal_one(self):
        verticals = [hesse_line(r, VERTICAL_THETA) for r in (100, 400)]
        horizontals = [hesse_line(r, HORIZONTAL_THETA) for r in (50, 300)]

        candidates = identify_line_candidates(
            verticals, horizontals, expected_x=[0, 610], expected_y=[0, 1340]
        )

        assert len(candidates) == 1
        assert len(candidates[0]["vertical"]) == 2
        assert len(candidates[0]["horizontal"]) == 2

    def test_the_richest_candidate_is_tried_first(self):
        verticals = [hesse_line(r, VERTICAL_THETA) for r in (100, 400, 700)]
        horizontals = [hesse_line(r, HORIZONTAL_THETA) for r in (50, 300)]

        candidates = identify_line_candidates(
            verticals, horizontals, expected_x=[0, 46, 610], expected_y=[0, 1340]
        )

        first = candidates[0]
        assert len(first["vertical"]) + len(first["horizontal"]) == 5

    def test_the_candidate_list_is_bounded(self):
        verticals = [hesse_line(100 * i, VERTICAL_THETA) for i in range(6)]
        horizontals = [hesse_line(100 * i, HORIZONTAL_THETA) for i in range(6)]

        candidates = identify_line_candidates(
            verticals,
            horizontals,
            expected_x=list(range(6)),
            expected_y=list(range(6)),
            max_candidates=10,
        )

        assert len(candidates) == 10


class TestCorrespondencesFromMatches:
    def test_every_grid_crossing_becomes_one_correspondence(self):
        matches = {
            "vertical": [
                (hesse_line(100, VERTICAL_THETA), 0.0),
                (hesse_line(400, VERTICAL_THETA), 610.0),
            ],
            "horizontal": [
                (hesse_line(50, HORIZONTAL_THETA), 0.0),
                (hesse_line(300, HORIZONTAL_THETA), 1340.0),
            ],
        }

        source, target = correspondences_from_matches(matches)

        assert source.tolist() == [[100, 50], [100, 300], [400, 50], [400, 300]]
        assert target.tolist() == [[0, 0], [0, 1340], [610, 0], [610, 1340]]

    def test_parallel_matches_yield_no_correspondences(self):
        matches = {
            "vertical": [(hesse_line(100, VERTICAL_THETA), 0.0)],
            "horizontal": [(hesse_line(400, VERTICAL_THETA), 0.0)],
        }

        source, _ = correspondences_from_matches(matches)

        assert source.size == 0
