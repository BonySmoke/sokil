"""
Court keypoints: naming them, and catching the ones placed wrongly.

The reprojection check earns its keep here. A vertex clicked on the wrong
intersection looks entirely plausible on screen and in a CSV, and it is only
visible against the other vertices, which the court model already knows the
distances between.
"""

import numpy as np
import pandas as pd
import pytest
from test_dataset_store import build, clip_row

from sokil.court import CourtModel
from sokil.dataset import Dataset, UnknownVertex, columns, resolve_vertex
from sokil.dataset.court import MAX_RESIDUAL_PX, reprojection_residuals

# A plausible camera: the court seen in perspective, roughly filling 1080p.
HOMOGRAPHY = np.array(
    [
        [1.6, 0.35, 240.0],
        [0.0, 0.55, 120.0],
        [0.0, 0.00045, 1.0],
    ]
)


def project(vertices: list[int]) -> np.ndarray:
    model = CourtModel().vertices
    points = np.array([model[v - 1] for v in vertices], dtype=float)
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    projected = homogeneous @ HOMOGRAPHY.T
    return projected[:, :2] / projected[:, 2][:, None]


def court_rows(
    clip_id: str = "clip_a",
    frame: int = 0,
    vertices: list[int] | None = None,
    nudge: dict[int, tuple[float, float]] | None = None,
) -> list[dict]:
    """Annotated keypoints for one keyframe, optionally with a vertex moved."""
    vertices = vertices or [1, 5, 26, 30, 11, 15, 3, 28]
    points = project(vertices)
    nudge = nudge or {}

    rows = []
    for vertex, (x, y) in zip(vertices, points):
        dx, dy = nudge.get(vertex, (0.0, 0.0))
        rows.append(
            {
                "clip_id": clip_id,
                "frame": frame,
                "vertex": vertex,
                "x_px": x + dx,
                "y_px": y + dy,
                "visible": True,
                "source": "solver_adjusted",
            }
        )
    return rows


def with_court(tmp_path, clips, rows) -> Dataset:
    dataset = build(tmp_path, clips, [])
    dataset.write("court_points", pd.DataFrame(rows, columns=columns("court_points")))
    return dataset


class TestVertexNames:
    def test_accepts_a_vertex_number(self):
        assert resolve_vertex("17") == 17

    def test_accepts_the_named_corners_the_skeleton_uses(self):
        # The four outer doubles corners, as the CVAT skeleton names them.
        assert resolve_vertex("bottom left") == 1
        assert resolve_vertex("bottom right") == 5
        assert resolve_vertex("top right") == 26
        assert resolve_vertex("top left") == 30

    def test_is_case_and_space_insensitive(self):
        assert resolve_vertex("  Top Left  ") == 30

    def test_rejects_a_number_off_the_court(self):
        with pytest.raises(UnknownVertex):
            resolve_vertex("31")

    def test_rejects_a_name_nobody_defined(self):
        with pytest.raises(UnknownVertex, match="net post"):
            resolve_vertex("net post")

    def test_the_named_corners_are_the_outer_corners_of_the_court(self):
        model = CourtModel()
        corners = [
            model.vertices[resolve_vertex(n) - 1]
            for n in ("bottom left", "bottom right", "top right", "top left")
        ]

        assert corners == [
            (0, 0),
            (model.width, 0),
            (model.width, model.length),
            (0, model.length),
        ]


class TestReprojection:
    def test_points_from_a_real_homography_have_no_residual(self):
        vertices = [1, 5, 26, 30, 11, 15, 3, 28]

        residuals = reprojection_residuals(vertices, project(vertices))

        assert residuals.max() < 0.01

    def test_a_misplaced_vertex_stands_out(self):
        vertices = [1, 5, 26, 30, 11, 15, 3, 28]
        points = project(vertices)
        points[4] += (40.0, 25.0)

        residuals = reprojection_residuals(vertices, points)

        assert residuals.max() > MAX_RESIDUAL_PX

    def test_too_few_points_fit_nothing(self):
        assert reprojection_residuals([1, 5, 26], project([1, 5, 26])) is None

    def test_collinear_points_fit_nothing(self):
        # Vertices 1 to 5 are the five intersections along one baseline.
        vertices = [1, 2, 3, 4, 5]

        assert reprojection_residuals(vertices, project(vertices)) is None


class TestCourtPointValidation:
    def test_a_well_annotated_court_has_no_problems(self, tmp_path):
        dataset = with_court(tmp_path, [clip_row("clip_a")], court_rows())

        assert dataset.validate() == []

    def test_catches_a_vertex_on_the_wrong_intersection(self, tmp_path):
        dataset = with_court(
            tmp_path, [clip_row("clip_a")], court_rows(nudge={11: (45.0, 30.0)})
        )

        assert any("px from where" in problem for problem in dataset.validate())

    def test_catches_too_few_visible_vertices(self, tmp_path):
        dataset = with_court(
            tmp_path, [clip_row("clip_a")], court_rows(vertices=[1, 5, 26])
        )

        assert any("3 visible vertices" in problem for problem in dataset.validate())

    def test_catches_a_visible_vertex_with_no_position(self, tmp_path):
        rows = court_rows()
        rows[0]["x_px"] = None
        rows[0]["y_px"] = None

        dataset = with_court(tmp_path, [clip_row("clip_a")], rows)

        assert any("visible without a position" in p for p in dataset.validate())

    def test_catches_a_hidden_vertex_that_still_has_one(self, tmp_path):
        rows = court_rows()
        rows[0]["visible"] = False

        dataset = with_court(tmp_path, [clip_row("clip_a")], rows)

        assert any("marked not visible" in problem for problem in dataset.validate())

    def test_allows_a_vertex_out_of_frame(self, tmp_path):
        # A corridor camera mode sees a fraction of the court, so most vertices
        # being absent is the normal case rather than a fault.
        rows = court_rows(vertices=[1, 5, 26, 30, 11, 15, 3, 28])
        rows.append(
            {
                "clip_id": "clip_a",
                "frame": 0,
                "vertex": 18,
                "x_px": None,
                "y_px": None,
                "visible": False,
                "source": "manual",
            }
        )

        dataset = with_court(tmp_path, [clip_row("clip_a")], rows)

        assert dataset.validate() == []

    def test_catches_a_keyframe_past_the_end_of_its_clip(self, tmp_path):
        dataset = with_court(
            tmp_path,
            [clip_row("clip_a", frame_count=100)],
            court_rows(frame=4000),
        )

        assert any("past the 100 frames" in problem for problem in dataset.validate())

    def test_catches_a_clip_with_no_court_at_all(self, tmp_path):
        dataset = with_court(
            tmp_path, [clip_row("clip_a"), clip_row("clip_b")], court_rows("clip_a")
        )

        assert any("have no court keypoints" in p for p in dataset.validate())

    def test_court_keypoints_are_optional(self, tmp_path):
        dataset = build(tmp_path, [clip_row("clip_a")], [])

        assert dataset.validate() == []


class TestKeyframes:
    def test_a_static_camera_needs_one_keyframe(self, tmp_path):
        dataset = with_court(
            tmp_path, [clip_row("clip_a", camera_static=True)], court_rows(frame=0)
        )

        assert dataset.validate() == []

    def test_catches_a_static_clip_with_several_keyframes(self, tmp_path):
        rows = court_rows(frame=0) + court_rows(frame=300)

        dataset = with_court(tmp_path, [clip_row("clip_a", camera_static=True)], rows)

        assert any("marked camera_static" in problem for problem in dataset.validate())

    def test_catches_a_moving_clip_described_by_one_keyframe(self, tmp_path):
        dataset = with_court(
            tmp_path, [clip_row("clip_a", camera_static=False)], court_rows(frame=0)
        )

        assert any("cannot describe it" in problem for problem in dataset.validate())

    def test_a_moving_camera_is_described_by_several(self, tmp_path):
        rows = court_rows(frame=0) + court_rows(frame=300) + court_rows(frame=900)

        dataset = with_court(tmp_path, [clip_row("clip_a", camera_static=False)], rows)

        assert dataset.validate() == []
