"""
Reading the ground-truth shuttle track out of CVAT.

This table is what lets the hit detection be measured without the models, so
the boxes have to be exactly the ones the annotator drew, on exactly the frames
they drew them. A box resurrected one frame past the end of a flight is not a
rounding error: it is a shuttle that was not there.
"""

from types import SimpleNamespace

from sokil.dataset.track_import import _interpolate, frame_numbers, merge_tracks


def shape(frame: int, x1, y1, x2, y2, outside=False):
    return SimpleNamespace(
        frame=frame, points=[x1, y1, x2, y2], outside=outside, occluded=False
    )


class TestInterpolate:
    def test_a_keyframe_gives_its_own_box(self):
        shapes = [shape(0, 10, 20, 30, 40), shape(10, 50, 60, 70, 80)]

        assert _interpolate(shapes, 0) == (10, 20, 30, 40)

    def test_between_keyframes_the_box_moves_linearly(self):
        shapes = [shape(0, 0, 0, 10, 10), shape(10, 100, 200, 110, 210)]

        assert _interpolate(shapes, 5) == (50, 100, 60, 110)

    def test_before_the_track_starts_there_is_nothing(self):
        shapes = [shape(10, 0, 0, 10, 10), shape(20, 0, 0, 10, 10)]

        assert _interpolate(shapes, 5) is None

    def test_after_the_track_ends_there_is_nothing(self):
        shapes = [shape(0, 0, 0, 10, 10), shape(10, 0, 0, 10, 10)]

        assert _interpolate(shapes, 15) is None

    def test_a_track_ending_outside_does_not_resurrect_the_shuttle(self):
        # CVAT marks the last shape of a flight outside, and that shape still
        # carries the box it had. Interpolating up to it would put the shuttle
        # back for one frame — on the very frame the next flight begins, which
        # is how this was found.
        shapes = [shape(0, 0, 0, 10, 10), shape(10, 90, 90, 100, 100, outside=True)]

        assert _interpolate(shapes, 10) is None

    def test_the_frame_before_an_outside_keyframe_still_has_a_box(self):
        shapes = [shape(0, 0, 0, 10, 10), shape(10, 90, 90, 100, 100, outside=True)]

        assert _interpolate(shapes, 9) is not None

    def test_an_interval_that_starts_outside_is_empty(self):
        # The shuttle left at frame 5 and came back at 10; nothing in between.
        shapes = [
            shape(0, 0, 0, 10, 10),
            shape(5, 50, 50, 60, 60, outside=True),
            shape(10, 80, 80, 90, 90),
        ]

        assert _interpolate(shapes, 7) is None
        assert _interpolate(shapes, 10) == (80, 80, 90, 90)

    def test_two_keyframes_on_one_frame_do_not_divide_by_zero(self):
        shapes = [shape(5, 0, 0, 10, 10), shape(5, 0, 0, 10, 10)]

        assert _interpolate(shapes, 5) == (0, 0, 10, 10)


class TestFrameNumbers:
    def task_with(self, names: list[str]):
        return SimpleNamespace(
            id=1, get_frames_info=lambda: [SimpleNamespace(name=n) for n in names]
        )

    def test_reads_the_number_off_each_frame_name(self):
        task = self.task_with(["clip_a_000000.jpg", "clip_a_000001.jpg"])

        assert frame_numbers(task) == {0: 0, 1: 1}

    def test_the_filename_decides_not_cvat_s_index(self):
        # Frames extracted with a stride: CVAT indexes 0,1,2 but the video's
        # frames are 0,30,60, and the video's numbering is the shared one.
        task = self.task_with(
            ["clip_a_000000.jpg", "clip_a_000030.jpg", "clip_a_000060.jpg"]
        )

        assert frame_numbers(task) == {0: 0, 1: 30, 2: 60}

    def test_reads_through_a_directory(self):
        task = self.task_with(["clip_a/clip_a/frame_000123.jpg"])

        assert frame_numbers(task) == {0: 123}

    def test_a_frame_with_no_number_is_left_out(self):
        task = self.task_with(["clip_a_000000.jpg", "screenshot.jpg"])

        assert frame_numbers(task) == {0: 0}


class TestMergeTracks:
    def rows_for(self, clip_id: str, frames: range) -> list[dict]:
        return [
            {"clip_id": clip_id, "frame": f, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}
            for f in frames
        ]

    def test_an_empty_table_takes_the_import_whole(self):
        merged = merge_tracks(None, self.rows_for("clip_a", range(3)))

        assert len(merged) == 3

    def test_another_clip_s_track_is_kept(self):
        existing = merge_tracks(None, self.rows_for("clip_a", range(3)))

        merged = merge_tracks(existing, self.rows_for("clip_b", range(2)))

        assert set(merged["clip_id"]) == {"clip_a", "clip_b"}

    def test_re_importing_a_clip_replaces_its_track(self):
        existing = merge_tracks(None, self.rows_for("clip_a", range(10)))

        merged = merge_tracks(existing, self.rows_for("clip_a", range(3)))

        assert len(merged) == 3

    def test_every_declared_column_survives(self):
        from sokil.dataset.schema import columns

        merged = merge_tracks(None, self.rows_for("clip_a", range(2)))

        assert list(merged.columns) == columns("tracks")
