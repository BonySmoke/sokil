"""
The court as the pipeline uses it: the homography, the in/out verdict, and
saving a solved court so a later run can reuse it.
"""

import json

import numpy as np
import pytest

from sokil.court import (
    Court,
    CourtCameraMode,
    CourtConfig,
    CourtModel,
    CourtModelType,
    ViewTransformer,
    load_homography,
    mask_completeness,
    save_homography,
)

# The tests below use the identity homography, so an image pixel is a court
# centimetre; these are the court's own dimensions, for readability.
COURT_WIDTH = CourtModel().width
COURT_LENGTH = CourtModel().length


class FakeSegmenter:
    """A Segmenter that returns whatever polygons the test hands it."""

    def __init__(self, polygons=None):
        self._polygons = [] if polygons is None else polygons
        self.calls = 0

    def polygons(self, image, confidence=None):
        self.calls += 1
        return self._polygons


class TestViewTransformer:
    SOURCE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    TARGET = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)

    def test_solves_a_homography_from_four_correspondences(self):
        transformer = ViewTransformer(self.SOURCE, self.TARGET)

        transformed = transformer.transform_points(np.array([[50, 50]], np.float32))

        assert transformed[0] == pytest.approx([100.0, 100.0], abs=1e-3)

    def test_the_inverse_undoes_the_transform(self):
        transformer = ViewTransformer(self.SOURCE, self.TARGET)
        points = np.array([[10, 90], [55, 5]], dtype=np.float32)

        roundtrip = transformer.inverse_transform_points(
            transformer.transform_points(points)
        )

        assert roundtrip == pytest.approx(points, abs=1e-3)

    def test_mismatched_point_sets_are_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            ViewTransformer(self.SOURCE, self.TARGET[:3])

    def test_three_dimensional_points_are_rejected(self):
        points = np.zeros((4, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="2D coordinates"):
            ViewTransformer(points, points)

    def test_an_empty_point_set_passes_straight_through(self):
        transformer = ViewTransformer(self.SOURCE, self.TARGET)
        empty = np.empty((0, 2), dtype=np.float32)

        assert transformer.transform_points(empty).size == 0
        assert transformer.inverse_transform_points(empty).size == 0

    def test_from_matrix_skips_the_correspondences(self):
        matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)

        transformer = ViewTransformer.from_matrix(matrix)

        assert transformer.transform_points(np.array([[10, 20]], np.float32))[
            0
        ] == pytest.approx([20.0, 40.0])

    def test_a_singular_matrix_is_rejected(self):
        with pytest.raises(ValueError, match="not invertible"):
            ViewTransformer.from_matrix(np.zeros((3, 3)))


class TestCourtFromHomography:
    def test_needs_neither_a_frame_nor_a_model(self, identity_court):
        assert identity_court.image is None
        assert identity_court.segmentation_model is None
        assert identity_court.get_view_transformer() is not None

    def test_carries_the_court_config_through(self):
        config = CourtConfig(court_model_type=CourtModelType.SINGLES)

        court = Court.from_homography(np.eye(3), court_config=config)

        assert court.court_model.court_model_type is CourtModelType.SINGLES

    def test_projects_the_model_corners_back_onto_the_image(self, identity_court):
        corners = identity_court.project_court_corners()

        assert corners == pytest.approx(
            np.array(identity_court.court_model.doubles_boundaries(), np.float32),
            abs=1e-3,
        )

    def test_projects_all_thirty_vertices_in_model_order(self, identity_court):
        points = identity_court.get_court_projection_points()

        assert points.shape == (30, 2)
        assert points[0].tolist() == [0, 0]


class TestInOutVerdict:
    def test_a_shuttle_well_inside_the_court_is_in(self, identity_court):
        assert identity_court.shuttle_intersects_court((300, 600)) is True

    def test_a_shuttle_well_outside_the_court_is_out(self, identity_court):
        assert identity_court.shuttle_intersects_court((-50, 600)) is False

    def test_a_shuttle_straddling_the_line_counts_as_in(self, identity_court):
        # the shuttle has a real radius: touching the line is in
        radius = identity_court.court_model.shuttle_diameter / 2

        assert identity_court.shuttle_intersects_court((-radius / 2, 600)) is True

    def test_the_singles_alley_is_out_for_a_singles_court(self):
        config = CourtConfig(court_model_type=CourtModelType.SINGLES)
        court = Court.from_homography(np.eye(3), court_config=config)
        inside_the_alley = court.court_model.side_corridor_width / 2

        assert court.shuttle_intersects_court((inside_the_alley, 600)) is False

    def test_the_same_point_is_in_for_a_doubles_court(self, identity_court):
        inside_the_alley = identity_court.court_model.side_corridor_width / 2

        assert identity_court.shuttle_intersects_court((inside_the_alley, 600)) is True

    def test_an_unwatched_side_never_calls_out(self):
        # A camera on the left corridor cannot see the right sideline, so a
        # shuttle far beyond it stays IN rather than being called on a line
        # this camera has no view of.
        config = CourtConfig(court_camera_mode=CourtCameraMode.LEFT_CORRIDOR)
        court = Court.from_homography(np.eye(3), court_config=config)

        assert court.shuttle_intersects_court((COURT_WIDTH + 500, 600)) is True
        assert court.shuttle_intersects_court((-50, 600)) is False

    def test_there_is_no_verdict_without_a_homography(self, frame):
        # nothing segments, so no lines are found and nothing can be solved
        court = Court(image=frame, segmentation_model=FakeSegmenter([]))

        assert court.shuttle_intersects_court((100, 100)) is None


class TestObservableArea:
    def test_a_shuttle_on_the_court_is_observable(self, identity_court):
        assert identity_court.shuttle_inside_max_observable_area((300, 600)) is True

    def test_a_shuttle_just_off_the_court_is_still_within_the_padding(
        self, identity_court
    ):
        assert (
            identity_court.shuttle_inside_max_observable_area((-30, 600), padding=50)
            is True
        )

    def test_a_shuttle_far_away_is_not(self, identity_court):
        assert (
            identity_court.shuttle_inside_max_observable_area((-500, 600), padding=50)
            is False
        )

    def test_the_gate_ignores_the_camera_mode(self):
        # This is a plausibility gate on the whole physical court, not the
        # verdict, so a corridor camera uses the same bound.
        config = CourtConfig(court_camera_mode=CourtCameraMode.LEFT_CORRIDOR)
        court = Court.from_homography(np.eye(3), court_config=config)

        assert court.shuttle_inside_max_observable_area((COURT_WIDTH - 10, 600)) is True

    def test_there_is_no_gate_without_a_homography(self, frame):
        court = Court(image=frame, segmentation_model=FakeSegmenter([]))

        assert court.shuttle_inside_max_observable_area((100, 100)) is None


class TestSegmentation:
    SQUARE = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)

    def test_the_contour_comes_from_the_segmenter(self, frame):
        court = Court(image=frame, segmentation_model=FakeSegmenter([self.SQUARE]))

        contour = court.get_contour()

        assert contour.shape == (4, 1, 2)

    def test_the_contour_is_only_segmented_once(self, frame):
        segmenter = FakeSegmenter([self.SQUARE])
        court = Court(image=frame, segmentation_model=segmenter)

        court.get_contour()
        court.get_contour()

        assert segmenter.calls == 1

    def test_no_mask_means_no_contour(self, frame):
        court = Court(image=frame, segmentation_model=FakeSegmenter([]))

        assert court.get_contour() is None

    def test_inside_tests_a_point_against_the_mask(self, frame):
        court = Court(image=frame, segmentation_model=FakeSegmenter([self.SQUARE]))

        assert court.inside((50, 50)) is True
        assert court.inside((5, 5)) is False

    def test_inside_has_no_answer_without_a_mask(self, frame):
        court = Court(image=frame, segmentation_model=FakeSegmenter([]))

        assert court.inside((50, 50)) is None

    def test_segmenting_without_a_frame_is_an_error(self):
        court = Court(image=None, segmentation_model=FakeSegmenter())

        with pytest.raises(ValueError, match="image property is not set"):
            court.segment()


class TestMaskCompleteness:
    def test_a_frame_with_no_mask_scores_zero(self, frame):
        court = Court(image=frame, segmentation_model=FakeSegmenter([]))

        assert mask_completeness(court) == 0.0

    def test_a_full_convex_mask_beats_a_ragged_one(self, frame):
        square = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
        # the same outline with a large bite taken out of it, as a player
        # standing in front of the court would leave
        bitten = np.array(
            [[10, 10], [90, 10], [90, 90], [50, 50], [10, 90]], dtype=np.float32
        )

        full = mask_completeness(
            Court(image=frame, segmentation_model=FakeSegmenter([square]))
        )
        ragged = mask_completeness(
            Court(image=frame, segmentation_model=FakeSegmenter([bitten]))
        )

        assert full > ragged

    def test_a_degenerate_mask_scores_zero(self, frame):
        line = np.array([[10, 10], [90, 10]], dtype=np.float32)
        court = Court(image=frame, segmentation_model=FakeSegmenter([line]))

        assert mask_completeness(court) == 0.0


class TestHomographyRoundTrip:
    def test_a_saved_court_loads_back_unchanged(self, tmp_path):
        matrix = np.array(
            [[1.5, 0.1, -20.0], [0.05, 1.2, 8.0], [1e-4, 2e-4, 1.0]], dtype=np.float64
        )
        court = Court.from_homography(matrix)
        path = tmp_path / "court.json"

        save_homography(court, path)

        assert load_homography(path) == pytest.approx(matrix, rel=1e-9)

    def test_the_file_is_readable_json_naming_what_it_holds(self, tmp_path):
        path = tmp_path / "nested" / "court.json"

        save_homography(Court.from_homography(np.eye(3)), path)

        assert json.loads(path.read_text())["image_to_court"] == [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]

    def test_a_bare_matrix_is_accepted_too(self, tmp_path):
        path = tmp_path / "bare.json"
        path.write_text("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]")

        assert load_homography(path) == pytest.approx(np.eye(3))

    def test_a_court_with_no_homography_cannot_be_saved(self, tmp_path, frame):
        court = Court(image=frame, segmentation_model=FakeSegmenter([]))

        with pytest.raises(ValueError, match="no homography"):
            save_homography(court, tmp_path / "court.json")


class TestSolveHomographyFromLines:
    """
    The geometric solve, end to end, on a court drawn to known dimensions.

    The court model is projected onto an image with a known scale and offset;
    the solver is then handed the lines that projection produces and has to
    recover a homography that maps the image back onto the model.
    """

    SCALE = 0.4
    OFFSET_X, OFFSET_Y = 30, 20

    def to_image(self, x_cm: float, y_cm: float) -> tuple[float, float]:
        return (x_cm * self.SCALE + self.OFFSET_X, y_cm * self.SCALE + self.OFFSET_Y)

    def build(self, model: CourtModel):
        width = int(model.width * self.SCALE) + 2 * self.OFFSET_X
        height = int(model.length * self.SCALE) + 2 * self.OFFSET_Y
        image = np.full((height, width, 3), 40, dtype=np.uint8)

        boundary = np.array(
            [
                self.to_image(x, y)
                for x, y in (
                    (0, 0),
                    (model.width, 0),
                    (model.width, model.length),
                    (0, model.length),
                )
            ],
            dtype=np.float32,
        )
        court = Court(image=image, segmentation_model=FakeSegmenter([boundary]))

        # theta = 0 is a vertical line at x = rho; theta = pi/2 a horizontal
        # one at y = rho
        verticals = [
            {"rho": self.to_image(coord, 0)[0], "theta": 0.0, "length": float(height)}
            for coord in model.stripe_center_coords("x")
        ]
        horizontals = [
            {
                "rho": self.to_image(0, coord)[1],
                "theta": float(np.pi / 2),
                "length": float(width),
            }
            for coord in model.stripe_center_coords("y")
        ]
        return court, verticals, horizontals

    def test_recovers_the_projection_it_was_built_from(self):
        model = CourtModel()
        court, verticals, horizontals = self.build(model)

        assert court.solve_homography_from_lines(verticals, horizontals) is True

        transformer = court.get_view_transformer()
        near_corner = transformer.transform_points(
            np.array([self.to_image(0, 0)], dtype=np.float32)
        )[0]
        far_corner = transformer.transform_points(
            np.array([self.to_image(model.width, model.length)], dtype=np.float32)
        )[0]

        assert near_corner == pytest.approx([0.0, 0.0], abs=2.0)
        assert far_corner == pytest.approx([model.width, model.length], abs=2.0)

    def test_the_accepted_solve_records_its_segmentation_overlap(self):
        court, verticals, horizontals = self.build(CourtModel())

        court.solve_homography_from_lines(verticals, horizontals)

        assert court.homography_iou > 0.9

    def test_a_recovered_court_calls_a_landing_the_same_way(self):
        model = CourtModel()
        court, verticals, horizontals = self.build(model)
        court.solve_homography_from_lines(verticals, horizontals)

        assert court.shuttle_intersects_court(self.to_image(300, 600)) is True
        assert court.shuttle_intersects_court(self.to_image(-40, 600)) is False

    def test_too_few_lines_cannot_span_the_grid(self):
        model = CourtModel()
        court, verticals, horizontals = self.build(model)

        assert (
            court.solve_homography_from_lines(verticals[:1], horizontals[:1]) is False
        )

    def test_a_court_that_was_never_solved_has_no_overlap_recorded(self):
        model = CourtModel()
        court, _, _ = self.build(model)

        assert court.homography_iou is None


class FakePoseEstimator:
    """A PoseEstimator returning whatever keypoints the test hands it."""

    def __init__(self, points=None, confidences=None):
        self.points = points
        self.confidences = confidences
        self.calls = 0

    def keypoints(self, image, confidence=None):
        self.calls += 1
        return self.points, self.confidences


class TestKeypoints:
    POINTS = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    CONFIDENCES = np.array([0.9, 0.1], dtype=np.float32)

    def court(self, frame, estimator):
        return Court(
            image=frame,
            segmentation_model=FakeSegmenter([]),
            keypoint_model=estimator,
        )

    def test_reads_the_points_and_their_confidences(self, frame):
        estimator = FakePoseEstimator(self.POINTS, self.CONFIDENCES)

        points, confidences = self.court(frame, estimator).get_keypoints()

        assert points is self.POINTS
        assert confidences is self.CONFIDENCES

    def test_the_model_is_only_asked_once(self, frame):
        estimator = FakePoseEstimator(self.POINTS, self.CONFIDENCES)
        court = self.court(frame, estimator)

        court.get_keypoints()
        court.get_keypoints()

        assert estimator.calls == 1

    def test_a_frame_with_no_keypoints_reports_none(self, frame):
        estimator = FakePoseEstimator(None, None)

        assert self.court(frame, estimator).get_keypoints() == (None, None)

    def test_without_a_frame_there_is_nothing_to_estimate_from(self):
        court = Court(
            image=None,
            segmentation_model=FakeSegmenter([]),
            keypoint_model=FakePoseEstimator(),
        )

        with pytest.raises(ValueError, match="image property is not set"):
            court.get_keypoints()


class TestKeypointFallback:
    """
    The fallback path: when the geometric solve finds nothing, the learned
    keypoints are snapped to detected line intersections and solved from.

    The frame is a court painted to known dimensions, and the "model" returns
    the vertices exactly where that projection puts them.
    """

    SCALE = 0.4
    OFFSET_X, OFFSET_Y = 30, 20

    def build(self, keypoint_confidence=0.9, visible=None):
        model = CourtModel()
        width = int(model.width * self.SCALE) + 2 * self.OFFSET_X
        height = int(model.length * self.SCALE) + 2 * self.OFFSET_Y
        image = np.full((height, width, 3), 40, dtype=np.uint8)

        for coord in model.unique_line_coords("x"):
            x = int(coord * self.SCALE + self.OFFSET_X)
            image[:, max(0, x - 2) : x + 3] = 235
        for coord in model.unique_line_coords("y"):
            y = int(coord * self.SCALE + self.OFFSET_Y)
            image[max(0, y - 2) : y + 3, :] = 235

        boundary = np.array(
            [
                (x * self.SCALE + self.OFFSET_X, y * self.SCALE + self.OFFSET_Y)
                for x, y in (
                    (0, 0),
                    (model.width, 0),
                    (model.width, model.length),
                    (0, model.length),
                )
            ],
            dtype=np.float32,
        )

        points = np.array(
            [
                (x * self.SCALE + self.OFFSET_X, y * self.SCALE + self.OFFSET_Y)
                for x, y in model.vertices
            ],
            dtype=np.float32,
        )
        confidences = np.full(len(points), keypoint_confidence, dtype=np.float32)
        if visible is not None:
            confidences = np.where(visible, keypoint_confidence, 0.0).astype(np.float32)

        court = Court(
            image=image,
            segmentation_model=FakeSegmenter([boundary]),
            keypoint_model=FakePoseEstimator(points, confidences),
        )
        return court, model

    def test_solves_a_homography_from_the_predicted_keypoints(self):
        court, _ = self.build()

        transformer = court._keypoint_view_transformer()

        assert transformer is not None
        near_corner = transformer.transform_points(
            np.array([[self.OFFSET_X, self.OFFSET_Y]], dtype=np.float32)
        )[0]
        assert near_corner == pytest.approx([0.0, 0.0], abs=15.0)

    def test_low_confidence_keypoints_are_dropped(self):
        # only three vertices are confident, which is one short of a homography
        visible = np.zeros(30, dtype=bool)
        visible[:3] = True
        court, _ = self.build(visible=visible)

        assert court._keypoint_view_transformer() is None

    def test_a_frame_with_no_predictions_cannot_fall_back(self, frame):
        court = Court(
            image=frame,
            segmentation_model=FakeSegmenter([]),
            keypoint_model=FakePoseEstimator(None, None),
        )

        assert court._keypoint_view_transformer() is None

    def test_snapping_can_be_switched_off(self):
        court, _ = self.build()
        court.court_config = CourtConfig(snap_keypoints=False)

        assert court._keypoint_view_transformer() is not None

    def test_the_view_transformer_prefers_the_geometric_solve(self):
        court, _ = self.build()

        transformer = court.get_view_transformer()

        assert transformer is not None
        # solved once, then cached
        assert court.get_view_transformer() is transformer

    def test_a_court_without_a_keypoint_model_has_no_fallback(self, frame):
        court = Court(image=frame, segmentation_model=FakeSegmenter([]))

        assert court.get_view_transformer() is None
