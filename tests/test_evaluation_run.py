"""
Composing the pipeline for one configuration and scoring what comes out.

The configurations differ only in which stage is handed ground truth, so what
needs holding down is that each one substitutes what it claims to and nothing
else — and that a clip whose court could not be solved says so rather than
reporting figures nobody should read.
"""

import cv2
import numpy as np
import pandas as pd
import pytest

from sokil.court import CourtCameraMode, CourtModelType
from sokil.dataset.schema import columns
from sokil.evaluation.run import CONFIGURATIONS, ground_truth_court, run_clip

HOMOGRAPHY = np.array([[1.6, 0.35, 240.0], [0.0, 0.55, 120.0], [0.0, 0.00045, 1.0]])
VERTICES = [1, 5, 26, 30, 11, 15, 3, 28]
FRAMES, SIZE = 30, (160, 120)


def to_image(x_cm: float, y_cm: float) -> tuple[float, float]:
    projected = HOMOGRAPHY @ np.array([x_cm, y_cm, 1.0])
    return tuple(projected[:2] / projected[2])


@pytest.fixture
def clip(tmp_path):
    """A clip id, a row for it, and a tiny video on disk."""
    path = tmp_path / "clip_a.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, SIZE)
    for index in range(FRAMES):
        frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        cv2.circle(frame, (20 + index * 4, 60), 6, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    row = pd.Series(
        {
            "fps": 30.0,
            "width": SIZE[0],
            "height": SIZE[1],
            "camera_mode": CourtCameraMode.FULL.value,
            "court_type": CourtModelType.DOUBLES.value,
        }
    )
    return "clip_a", row, path


@pytest.fixture
def court_points():
    from sokil.court import CourtModel

    model = CourtModel().vertices
    return pd.DataFrame(
        [
            {
                "clip_id": "clip_a",
                "frame": 0,
                "vertex": vertex,
                "x_px": to_image(*model[vertex - 1])[0],
                "y_px": to_image(*model[vertex - 1])[1],
                "visible": True,
                "source": "manual",
            }
            for vertex in VERTICES
        ]
    )


class ScriptedDetector:
    """Answers from a dict keyed by frame, as a ground-truth source does."""

    def __init__(self, boxes: dict):
        self.boxes = boxes

    def best_box(self, image, confidence=None, frame=None):
        box = self.boxes.get(frame)
        return None if box is None else (box, 1.0)


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=columns("events"))


class TestConfigurations:
    def test_each_configuration_says_what_it_substitutes(self):
        assert set(CONFIGURATIONS) == {"manual", "shuttle-detector", "automatic"}
        assert all(CONFIGURATIONS.values())


class TestRunClip:
    def test_the_configuration_is_carried_into_the_result(self, clip, court_points):
        clip_id, row, path = clip

        result = run_clip(
            clip_id,
            path,
            row,
            empty_events(),
            court_points,
            configuration="manual",
            detector=ScriptedDetector({0: (10, 50, 20, 60)}),
            court=ground_truth_court(clip_id, row, court_points),
        )

        assert result.configuration == "manual"

    def test_it_tracks_the_frames_the_detector_answered_for(self, clip, court_points):
        clip_id, row, path = clip
        boxes = {index: (10 + index, 50, 20 + index, 60) for index in range(5)}

        result = run_clip(
            clip_id,
            path,
            row,
            empty_events(),
            court_points,
            configuration="manual",
            detector=ScriptedDetector(boxes),
            court=ground_truth_court(clip_id, row, court_points),
        )

        assert result.frames_tracked == 5

    def test_a_clip_with_no_landings_invents_none(self, clip, court_points):
        clip_id, row, path = clip

        result = run_clip(
            clip_id,
            path,
            row,
            empty_events(),
            court_points,
            configuration="manual",
            detector=ScriptedDetector({}),
            court=ground_truth_court(clip_id, row, court_points),
        )

        assert result.detection["labelled"] == 0
        assert result.detection["matched"] == 0

    def test_the_annotation_s_own_fit_is_reported(self, clip, court_points):
        # Points taken from a real homography fit it exactly, so the residual
        # is the floor: anything larger would mean the fit, not the court.
        clip_id, row, path = clip

        result = run_clip(
            clip_id,
            path,
            row,
            empty_events(),
            court_points,
            configuration="manual",
            detector=ScriptedDetector({}),
            court=ground_truth_court(clip_id, row, court_points),
        )

        assert result.court_error["vertices"] == len(VERTICES)
        assert result.court_error["annotation_px"]["median"] < 0.1

    def test_the_annotated_court_still_reports_its_own_precision(
        self, clip, court_points
    ):
        # Not zero in general: it measures how precisely the vertices were
        # clicked, which is the floor under every landing position. These
        # points come from an exact homography, so here the floor is nil.
        clip_id, row, path = clip

        result = run_clip(
            clip_id,
            path,
            row,
            empty_events(),
            court_points,
            configuration="manual",
            detector=ScriptedDetector({}),
            court=ground_truth_court(clip_id, row, court_points),
        )

        assert result.court_error["vertex_cm"]["max"] < 1.0

    def test_a_failed_solve_reports_no_landing_positions(self, clip, court_points):
        # A distance measured against a court the run never had would read as
        # a result rather than an absence.
        clip_id, row, path = clip
        landings = pd.DataFrame(
            [
                {
                    **{column: None for column in columns("events")},
                    "clip_id": clip_id,
                    "frame": 2,
                    "call": "in",
                    "court_type": "doubles",
                    "annotator": "test",
                }
            ],
            columns=columns("events"),
        )

        class FailingSolver:
            def solve(self, source):
                return None

        result = run_clip(
            clip_id,
            path,
            row,
            landings,
            court_points,
            configuration="automatic",
            detector=ScriptedDetector(
                {index: (10 + index * 3, 50, 20 + index * 3, 60) for index in range(8)}
            ),
            court_solver=FailingSolver(),
        )

        assert result.court_solved is False
        assert result.position_error["count"] == 0

    def test_neither_a_court_nor_a_solver_is_a_mistake(self, clip, court_points):
        clip_id, row, path = clip

        with pytest.raises(ValueError, match="court or a solver"):
            run_clip(
                clip_id,
                path,
                row,
                empty_events(),
                court_points,
                configuration="automatic",
                detector=ScriptedDetector({}),
            )

    def test_a_court_that_could_not_be_solved_is_recorded_as_such(
        self, clip, court_points
    ):
        # Without this the figures beneath it read as bad rather than absent.
        clip_id, row, path = clip

        class FailingSolver:
            def solve(self, source):
                return None

        result = run_clip(
            clip_id,
            path,
            row,
            empty_events(),
            court_points,
            configuration="automatic",
            detector=ScriptedDetector({}),
            court_solver=FailingSolver(),
        )

        assert result.court_solved is False

    def test_a_given_court_counts_as_solved(self, clip, court_points):
        clip_id, row, path = clip

        result = run_clip(
            clip_id,
            path,
            row,
            empty_events(),
            court_points,
            configuration="manual",
            detector=ScriptedDetector({}),
            court=ground_truth_court(clip_id, row, court_points),
        )

        assert result.court_solved is True


class TestGroundTruthCourt:
    def test_it_places_the_court_the_keypoints_describe(self, clip, court_points):
        from sokil.court import CourtModel

        clip_id, row, _ = clip
        court = ground_truth_court(clip_id, row, court_points)
        model = CourtModel()

        placed = court.get_view_transformer().transform_points(
            np.array([to_image(305.0, 670.0)])
        )[0]

        assert placed[0] == pytest.approx(305.0, abs=1.0)
        assert placed[1] == pytest.approx(670.0, abs=1.0)
        assert court.court_model.width == model.width

    def test_it_takes_the_camera_mode_and_court_type_from_the_clip(
        self, clip, court_points
    ):
        clip_id, row, _ = clip
        row = row.copy()
        row["court_type"] = CourtModelType.SINGLES.value

        court = ground_truth_court(clip_id, row, court_points)

        assert court.court_model.court_model_type == CourtModelType.SINGLES

    def test_a_clip_with_no_keypoints_says_so(self, clip, court_points):
        _, row, _ = clip

        with pytest.raises(ValueError, match="No court keypoints"):
            ground_truth_court("clip_ghost", row, court_points)


class TestCourtOrientation:
    """
    A badminton court is almost symmetric end to end, so a solver matching
    lines by mask area cannot tell which baseline is which. In/out survives
    that; a reported coordinate does not. Reporting the court's own length as
    an "error" would bury the diagnosis in a number nobody can act on.
    """

    def flipped_court(self, clip, court_points):
        """A court of the right shape, placed the wrong way round."""
        from sokil.court import Court, CourtModel

        clip_id, row, _ = clip
        upright = ground_truth_court(clip_id, row, court_points)
        matrix = np.array(upright.get_view_transformer().m, dtype=float)

        # Mirror the court axis: y becomes length - y.
        mirror = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, CourtModel().length], [0.0, 0.0, 1.0]]
        )
        return Court.from_homography(mirror @ matrix)

    def test_an_upright_court_is_not_called_flipped(self, clip, court_points):
        clip_id, row, path = clip

        result = run_clip(
            clip_id,
            path,
            row,
            empty_events(),
            court_points,
            configuration="manual",
            detector=ScriptedDetector({}),
            court=ground_truth_court(clip_id, row, court_points),
        )

        assert result.court_error["flipped_end_for_end"] is False

    def test_a_flipped_court_is_named_as_flipped(self, clip, court_points):
        clip_id, row, path = clip

        result = run_clip(
            clip_id,
            path,
            row,
            empty_events(),
            court_points,
            configuration="automatic",
            detector=ScriptedDetector({}),
            court=self.flipped_court(clip, court_points),
        )

        assert result.court_error["flipped_end_for_end"] is True

    def test_a_flipped_court_s_distance_is_measured_after_unflipping(
        self, clip, court_points
    ):
        # Otherwise it reads as the court's own length, which says nothing
        # about how well the court was placed.
        clip_id, row, path = clip

        result = run_clip(
            clip_id,
            path,
            row,
            empty_events(),
            court_points,
            configuration="automatic",
            detector=ScriptedDetector({}),
            court=self.flipped_court(clip, court_points),
        )

        assert result.court_error["vertex_cm"]["median"] < 1.0
