"""
Which end of the court the solver decides faces the camera.

A court is almost symmetric about its mid-court line, so the assignment that
swaps the two baselines puts a court of the same shape in the same place and
scores the same segmentation IoU. It leaves in/out correct, because the
boundary is symmetric too, and mirrors every reported position along the
court's length — which is why the mask cannot be allowed the last word.
"""

import numpy as np
import pytest

from sokil.court import MIN_END_SEPARATION_PX, Court, CourtModel, ViewTransformer

# A camera above the court looking down it, as court->image. Court y = 0 is the
# baseline under the camera, so it lands low in the frame (image y 350) and
# y = length lands high (image y 91): that difference is the only thing telling
# the two ends apart.
PERSPECTIVE = np.array([[1.6, 0.35, 240.0], [0.0, -0.35, 560.0], [0.0, -0.00045, 1.6]])


def image_to_court(matrix: np.ndarray) -> ViewTransformer:
    """A transformer from a court->image matrix, which is how these are built."""
    return ViewTransformer.from_matrix(np.linalg.inv(matrix))


def mirrored(matrix: np.ndarray) -> np.ndarray:
    """The same court, placed end for end."""
    length = CourtModel().length
    mirror = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, length], [0.0, 0.0, 1.0]])
    return matrix @ mirror


def court_with(matrix: np.ndarray) -> Court:
    return Court.from_homography(np.linalg.inv(matrix))


class TestFacesCamera:
    def test_a_court_with_its_near_baseline_at_y_zero_faces_the_camera(self):
        court = court_with(PERSPECTIVE)

        assert court._origin_nearest_to_the_camera(image_to_court(PERSPECTIVE))

    def test_the_same_court_placed_end_for_end_does_not(self):
        court = court_with(PERSPECTIVE)

        assert not court._origin_nearest_to_the_camera(
            image_to_court(mirrored(PERSPECTIVE))
        )

    def test_the_two_are_the_only_options_and_disagree(self):
        # Guards the test itself: if both read the same way, it proves nothing.
        court = court_with(PERSPECTIVE)

        upright = court._origin_nearest_to_the_camera(image_to_court(PERSPECTIVE))
        flipped = court._origin_nearest_to_the_camera(
            image_to_court(mirrored(PERSPECTIVE))
        )

        assert upright != flipped

    def test_the_near_baseline_really_is_lower_in_the_frame(self):
        # The assumption the whole check rests on, stated so it is checked
        # rather than believed.
        model = CourtModel()
        ends = np.array(
            [[model.width / 2, 0.0], [model.width / 2, model.length]],
            dtype=np.float32,
        )

        projected = image_to_court(PERSPECTIVE).inverse_transform_points(ends)

        assert projected[0][1] > projected[1][1]


class TestUndecidableCourts:
    """
    A camera level with the court, or looking along its length, flattens the
    difference between the ends. Turning a court round on that is worse than
    leaving it where the mask put it.
    """

    def flat_matrix(self, separation: float) -> np.ndarray:
        """A court->image matrix placing the two baselines `separation` apart."""
        model = CourtModel()
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, separation / model.length, 100.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def test_ends_too_close_together_are_left_alone(self):
        matrix = self.flat_matrix(MIN_END_SEPARATION_PX / 2)
        court = court_with(matrix)

        assert court._origin_nearest_to_the_camera(image_to_court(matrix))
        assert court._origin_nearest_to_the_camera(image_to_court(mirrored(matrix)))

    def test_ends_far_enough_apart_are_judged(self):
        matrix = self.flat_matrix(MIN_END_SEPARATION_PX * 20)
        court = court_with(matrix)

        assert court._origin_nearest_to_the_camera(
            image_to_court(matrix)
        ) != court._origin_nearest_to_the_camera(image_to_court(mirrored(matrix)))

    def test_a_degenerate_transformer_is_left_alone(self):
        court = court_with(PERSPECTIVE)
        singular = ViewTransformer.from_matrix(np.eye(3))
        singular.m_inv = np.full((3, 3), np.nan)

        assert court._origin_nearest_to_the_camera(singular)


class TestOrientationBeatsIou:
    """
    Orientation is decided before the mask gets a say, because the mask scores
    both ends identically and cannot break the tie.
    """

    def test_a_worse_fitting_court_the_right_way_round_wins(self, monkeypatch):
        from sokil import court as court_module

        court = court_with(PERSPECTIVE)
        upright = np.linalg.inv(PERSPECTIVE)
        flipped = np.linalg.inv(mirrored(PERSPECTIVE))

        built = [
            ViewTransformer.from_matrix(flipped),
            ViewTransformer.from_matrix(upright),
        ]
        scores = {id(built[0]): 0.99, id(built[1]): 0.60}

        monkeypatch.setattr(
            court_module,
            "identify_line_candidates",
            lambda *args, **kwargs: [object(), object()],
        )
        monkeypatch.setattr(
            court_module,
            "correspondences_from_matches",
            lambda match: (np.zeros((4, 2), np.float32), np.zeros((4, 2), np.float32)),
        )
        monkeypatch.setattr(
            court_module, "ViewTransformer", lambda **kwargs: built.pop(0)
        )
        monkeypatch.setattr(
            Court,
            "_transformer_score",
            lambda self, transformer, source, target: (0.0, scores[id(transformer)]),
        )

        chosen = court._solve_homography_from_lines([], [])

        assert court._origin_nearest_to_the_camera(chosen)
        assert court.homography_iou == pytest.approx(0.60)
