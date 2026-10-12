"""
Reading court keypoints out of a CVAT skeleton track.

Keyframe collapsing is the part that needs holding down. Clicking a vertex
without dragging it leaves a keyframe behind, so a clip whose camera never moved
comes back claiming several courts — and the dataset would then contradict its
own camera_static flag on footage that is perfectly fine.
"""

from types import SimpleNamespace

import pytest

from sokil.dataset.cvat_import import (
    ADJUSTED_TOLERANCE_PX,
    court_points_from_track,
    default_clip_id,
    merge_court_points,
)

# The four named corners plus one numbered vertex, which is enough to exercise
# naming, interpolation and collapsing without writing thirty of everything.
SUBLABELS = {
    2: "bottom left",
    6: "bottom right",
    27: "top right",
    31: "top left",
    13: "12",
}
VERTICES = {2: 1, 6: 5, 27: 26, 31: 30, 13: 12}


def shape(frame: int, x: float, y: float, outside: bool = False):
    return SimpleNamespace(frame=frame, points=[x, y], outside=outside, occluded=False)


def track(elements: dict[int, list]):
    return SimpleNamespace(
        elements=[
            SimpleNamespace(label_id=label_id, shapes=shapes)
            for label_id, shapes in elements.items()
        ]
    )


def one_court(frame: int = 0, offset: float = 0.0, outside: set[int] | None = None):
    """A skeleton where every element sits at a distinct, valid position."""
    outside = outside or set()
    return {
        label_id: [
            shape(
                frame,
                100.0 * index + offset,
                200.0 * index + offset,
                outside=label_id in outside,
            )
        ]
        for index, label_id in enumerate(SUBLABELS, start=1)
    }


def rows_for(elements, **kwargs):
    return court_points_from_track(track(elements), SUBLABELS, "clip_a", **kwargs)


class TestVertexMapping:
    def test_maps_sublabels_to_court_vertices(self):
        rows = rows_for(one_court())

        assert sorted(row["vertex"] for row in rows) == sorted(VERTICES.values())

    def test_skips_an_element_whose_sublabel_is_unknown(self):
        elements = one_court()
        elements[999] = [shape(0, 5.0, 5.0)]

        rows = rows_for(elements)

        assert len(rows) == len(SUBLABELS)


class TestVisibility:
    def test_an_outside_vertex_is_not_visible_and_has_no_position(self):
        rows = rows_for(one_court(outside={6}))

        hidden = next(row for row in rows if row["vertex"] == VERTICES[6])
        assert hidden["visible"] is False
        assert hidden["x_px"] is None
        assert hidden["y_px"] is None

    def test_a_visible_vertex_keeps_its_position(self):
        rows = rows_for(one_court())

        visible = next(row for row in rows if row["vertex"] == VERTICES[2])
        assert visible["visible"] is True
        assert visible["x_px"] == 100.0


class TestKeyframeCollapsing:
    def test_a_repeated_court_collapses_to_one_keyframe(self):
        # What a click that does not drag leaves behind: a second keyframe at
        # exactly the same place.
        elements = {
            label_id: [shapes[0], shape(500, shapes[0].points[0], shapes[0].points[1])]
            for label_id, shapes in one_court().items()
        }

        rows = rows_for(elements)

        assert {row["frame"] for row in rows} == {0}

    def test_a_court_that_moved_keeps_both_keyframes(self):
        elements = {
            label_id: [
                shapes[0],
                shape(500, shapes[0].points[0] + 30, shapes[0].points[1] + 30),
            ]
            for label_id, shapes in one_court().items()
        }

        rows = rows_for(elements)

        assert {row["frame"] for row in rows} == {0, 500}

    def test_a_nudge_within_tolerance_is_not_a_new_court(self):
        elements = {
            label_id: [
                shapes[0],
                shape(500, shapes[0].points[0] + 1.0, shapes[0].points[1]),
            ]
            for label_id, shapes in one_court().items()
        }

        rows = rows_for(elements)

        assert {row["frame"] for row in rows} == {0}

    def test_a_vertex_leaving_view_is_a_new_court(self):
        # Nothing moved, but what the camera can see did, and a homography
        # fitted to the smaller set is a different measurement.
        elements = one_court()
        for label_id, shapes in elements.items():
            shapes.append(
                shape(
                    500, shapes[0].points[0], shapes[0].points[1], outside=label_id == 6
                )
            )

        rows = rows_for(elements)

        assert {row["frame"] for row in rows} == {0, 500}


class TestInterpolation:
    def test_an_element_without_its_own_keyframe_is_interpolated(self):
        # One vertex is re-placed at frame 100; the others never move, so at
        # frame 100 they must still report where CVAT would draw them.
        elements = one_court()
        elements[2].append(shape(100, 300.0, 400.0))

        rows = rows_for(elements)

        at_100 = {row["vertex"]: row for row in rows if row["frame"] == 100}
        assert at_100[VERTICES[2]]["x_px"] == 300.0
        assert at_100[VERTICES[6]]["x_px"] == 200.0

    def test_a_position_between_keyframes_moves_linearly(self):
        elements = one_court()
        elements[2] = [shape(0, 0.0, 0.0), shape(100, 100.0, 200.0)]
        # A second element keyframed at 50 forces a keyframe there for all.
        elements[6] = [shape(0, 500.0, 500.0), shape(50, 540.0, 500.0)]

        rows = rows_for(elements)

        at_50 = {row["vertex"]: row for row in rows if row["frame"] == 50}
        assert at_50[VERTICES[2]]["x_px"] == 50.0
        assert at_50[VERTICES[2]]["y_px"] == 100.0


class TestSource:
    def test_points_are_manual_without_a_pre_annotation(self):
        rows = rows_for(one_court())

        assert {row["source"] for row in rows} == {"manual"}

    def test_an_untouched_pre_annotation_reads_as_the_solver(self):
        rows = rows_for(
            one_court(),
            preannotation={
                v: (100.0 * i, 200.0 * i)
                for i, v in enumerate(VERTICES.values(), start=1)
            },
        )

        assert {row["source"] for row in rows} == {"solver"}

    def test_a_moved_point_reads_as_adjusted(self):
        preannotation = {
            v: (100.0 * i + ADJUSTED_TOLERANCE_PX + 5, 200.0 * i)
            for i, v in enumerate(VERTICES.values(), start=1)
        }

        rows = rows_for(one_court(), preannotation=preannotation)

        assert {row["source"] for row in rows} == {"solver_adjusted"}


class TestClipId:
    def test_suggests_a_clip_id_from_the_video_name(self):
        assert (
            default_clip_id("capture_fast_20260419_100233-cut-merged-177688.mp4")
            == "capture_fast_20260419_100233-cut-merged-177688"
        )

    def test_lowercases_and_replaces_what_a_clip_id_may_not_hold(self):
        assert default_clip_id("Court 5 Left.MP4") == "court-5-left"


@pytest.mark.parametrize("frame", [0, 250, 1000])
def test_rows_carry_the_frame_they_describe(frame):
    rows = rows_for(one_court(frame=frame))

    assert {row["frame"] for row in rows} == {frame}


class TestMergingCourtPoints:
    """
    A run that imports one task must not drop the clips it did not cover.
    Replacing the whole table loses them silently, with the only copy left in
    CVAT — and a table that looks complete is the worst way to lose data.
    """

    def rows_for(self, clip_id: str, frame: int = 0) -> list[dict]:
        return [
            {
                "clip_id": clip_id,
                "frame": frame,
                "vertex": vertex,
                "x_px": 10.0 * vertex,
                "y_px": 20.0 * vertex,
                "visible": True,
                "source": "manual",
            }
            for vertex in (1, 5, 26, 30)
        ]

    def test_an_empty_table_takes_the_import_whole(self):
        merged = merge_court_points(None, self.rows_for("clip_a"))

        assert set(merged["clip_id"]) == {"clip_a"}

    def test_another_clip_s_points_are_kept(self):
        existing = merge_court_points(None, self.rows_for("clip_a"))

        merged = merge_court_points(existing, self.rows_for("clip_b"))

        assert set(merged["clip_id"]) == {"clip_a", "clip_b"}

    def test_the_imported_clip_is_replaced_not_added_to(self):
        existing = merge_court_points(None, self.rows_for("clip_a", frame=0))

        merged = merge_court_points(existing, self.rows_for("clip_a", frame=500))

        assert set(merged["frame"]) == {500}

    def test_replacing_one_clip_leaves_the_other_untouched(self):
        existing = merge_court_points(None, self.rows_for("clip_a", frame=0))
        existing = merge_court_points(existing, self.rows_for("clip_b", frame=0))

        merged = merge_court_points(existing, self.rows_for("clip_a", frame=900))

        assert set(merged[merged["clip_id"] == "clip_b"]["frame"]) == {0}
        assert set(merged[merged["clip_id"] == "clip_a"]["frame"]) == {900}

    def test_every_declared_column_survives_the_merge(self):
        from sokil.dataset.schema import columns

        merged = merge_court_points(None, self.rows_for("clip_a"))

        assert list(merged.columns) == columns("court_points")
