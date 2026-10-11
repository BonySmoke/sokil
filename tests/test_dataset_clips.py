"""
Describing clips from the recordings themselves.
"""

import cv2
import numpy as np
import pandas as pd
import pytest

from sokil.dataset.clips import (
    blank_columns,
    clip_rows,
    is_empty,
    observed_camera_mode,
    observed_camera_static,
    observed_court_type,
    probe,
    session_id,
    source_uri,
)
from sokil.dataset.schema import columns

COUNT, SIZE, FPS = 24, (64, 48), 30


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "court_5_left_20260419_100233-cut.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for index in range(COUNT):
        writer.write(np.full((SIZE[1], SIZE[0], 3), index, np.uint8))
    writer.release()
    return path


def events_for(clip_id: str, court_types: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"clip_id": clip_id, "court_type": kind} for kind in court_types]
    )


def court_points_for(clip_id: str, frames: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"clip_id": clip_id, "frame": f, "vertex": v, "visible": True}
            for f in frames
            for v in (1, 2, 9, 10, 11, 12, 19, 20, 21, 22, 29, 30)
        ]
    )


class TestProbe:
    def test_reads_what_the_clip_says_about_itself(self, video):
        assert probe(video) == {
            "frame_count": COUNT,
            "fps": float(FPS),
            "width": SIZE[0],
            "height": SIZE[1],
        }

    def test_an_unreadable_clip_says_so(self, tmp_path):
        (tmp_path / "broken.mp4").write_bytes(b"not a video")

        with pytest.raises(ValueError, match="Could not open video"):
            probe(tmp_path / "broken.mp4")


class TestSessionId:
    def test_reads_the_recording_date_out_of_the_name(self):
        assert session_id("court_5_left_20260419_100233-cut") == "s-2026-04-19"

    def test_a_name_with_no_date_gets_none_rather_than_a_guess(self):
        assert session_id("some-clip") is None


class TestObservedCourtType:
    def test_takes_the_type_the_landings_were_called_against(self):
        events = events_for("clip_a", ["singles", "singles"])

        assert observed_court_type(events, "clip_a") == "singles"

    def test_landings_that_disagree_settle_nothing(self):
        # A majority vote here would quietly pick one and hide the mistake.
        events = events_for("clip_a", ["singles", "doubles", "doubles"])

        assert observed_court_type(events, "clip_a") is None

    def test_a_clip_with_no_landings_settles_nothing(self):
        assert observed_court_type(events_for("clip_b", ["singles"]), "clip_a") is None


class TestObservedCameraStatic:
    def test_one_keyframe_means_the_camera_held_still(self):
        assert observed_camera_static(court_points_for("clip_a", [0, 0, 0]), "clip_a")

    def test_several_keyframes_mean_it_did_not(self):
        points = court_points_for("clip_a", [0, 0, 300, 300])

        assert observed_camera_static(points, "clip_a") is False

    def test_a_clip_with_no_court_settles_nothing(self):
        points = court_points_for("clip_a", [0])

        assert observed_camera_static(points, "clip_b") is None


class TestClipRows:
    def test_writes_every_declared_column(self, video):
        row = clip_rows({"clip_a": video})[0]

        assert set(row) == set(columns("clips"))

    def test_fills_in_what_the_file_answers(self, video):
        row = clip_rows({"clip_a": video})[0]

        assert row["frame_count"] == COUNT
        assert row["width"] == SIZE[0]

    def test_leaves_the_judgement_calls_empty(self, video):
        row = clip_rows({"clip_a": video})[0]

        assert row["camera_mode"] is None
        assert row["source_uri"] is None

    def test_takes_the_court_type_from_the_landings(self, video):
        rows = clip_rows({"clip_a": video}, events=events_for("clip_a", ["singles"]))

        assert rows[0]["court_type"] == "singles"

    def test_takes_camera_static_from_the_court_keyframes(self, video):
        rows = clip_rows(
            {"clip_a": video}, court_points=court_points_for("clip_a", [0, 500])
        )

        assert rows[0]["camera_static"] is False

    def test_never_overwrites_an_answer_given_by_hand(self, video):
        existing = pd.DataFrame(
            [
                {
                    **{c: None for c in columns("clips")},
                    "clip_id": "clip_a",
                    "camera_mode": "left_corridor",
                    "court_type": "doubles",
                }
            ]
        )

        rows = clip_rows(
            {"clip_a": video},
            events=events_for("clip_a", ["singles"]),
            existing=existing,
        )

        assert rows[0]["camera_mode"] == "left_corridor"
        assert rows[0]["court_type"] == "doubles"

    def test_re_reads_the_file_even_for_a_known_clip(self, video):
        # The digest and frame count describe the file, so a replaced recording
        # has to show up rather than being taken on trust from last time.
        existing = pd.DataFrame(
            [
                {
                    **{c: None for c in columns("clips")},
                    "clip_id": "clip_a",
                    "frame_count": 1,
                }
            ]
        )

        rows = clip_rows({"clip_a": video}, existing=existing)

        assert rows[0]["frame_count"] == COUNT

    def test_rows_come_out_in_clip_order(self, video):
        rows = clip_rows({"clip_b": video, "clip_a": video})

        assert [row["clip_id"] for row in rows] == ["clip_a", "clip_b"]


class TestBlankColumns:
    def test_reports_what_is_still_empty(self, video):
        rows = clip_rows({"clip_a": video})

        assert "camera_mode" in blank_columns(rows)["clip_a"]

    def test_a_complete_row_is_not_reported(self):
        rows = [{column: "filled" for column in columns("clips")}]

        assert blank_columns(rows) == {}


class TestEmptyCells:
    """
    A cell read back from CSV is empty as NaN or NA, never None. Treating only
    None as empty makes every blank look filled, so nothing gets filled in and
    nothing gets reported — which is how a column stays blank for ever.
    """

    def test_none_is_empty(self):
        assert is_empty(None)

    def test_a_missing_number_is_empty(self):
        assert is_empty(float("nan"))

    def test_a_missing_string_is_empty(self):
        assert is_empty(pd.NA)

    def test_a_value_is_not_empty(self):
        assert not is_empty("left_corridor")

    def test_false_is_a_value_not_a_blank(self):
        assert not is_empty(False)

    def test_zero_is_a_value(self):
        assert not is_empty(0)

    def test_fills_a_column_that_came_back_empty_from_csv(self, video, tmp_path):
        existing = pd.DataFrame(
            [{**{c: None for c in columns("clips")}, "clip_id": "clip_a"}]
        )
        existing.to_csv(tmp_path / "clips.csv", index=False)
        reread = pd.read_csv(tmp_path / "clips.csv")

        rows = clip_rows(
            {"clip_a": video},
            court_points=court_points_for("clip_a", [0]),
            existing=reread,
        )

        assert rows[0]["camera_static"] is True

    def test_reports_a_column_that_came_back_empty_from_csv(self, video, tmp_path):
        rows = clip_rows({"clip_a": video})

        assert "source_uri" in blank_columns(rows)["clip_a"]


class TestObservedCameraMode:
    def test_a_corridor_s_own_vertices_name_it(self):
        points = pd.DataFrame(
            [
                {"clip_id": "clip_a", "vertex": v, "visible": True}
                for v in (1, 2, 9, 10, 11, 12, 19, 20, 21, 22, 29, 30)
            ]
        )

        assert observed_camera_mode(points, "clip_a") == "left_corridor"

    def test_extra_vertices_do_not_confuse_it(self):
        # A corridor camera sees the centre line too without watching it.
        points = pd.DataFrame(
            [
                {"clip_id": "clip_a", "vertex": v, "visible": True}
                for v in (1, 2, 9, 10, 11, 12, 19, 20, 21, 22, 29, 30, 18, 23, 28)
            ]
        )

        assert observed_camera_mode(points, "clip_a") == "left_corridor"

    def test_the_whole_court_is_full(self):
        points = pd.DataFrame(
            [{"clip_id": "clip_a", "vertex": v, "visible": True} for v in range(1, 31)]
        )

        assert observed_camera_mode(points, "clip_a") == "full"

    def test_a_partial_view_of_no_corridor_settles_nothing(self):
        points = pd.DataFrame(
            [{"clip_id": "clip_a", "vertex": v, "visible": True} for v in (1, 5, 17)]
        )

        assert observed_camera_mode(points, "clip_a") is None

    def test_vertices_out_of_view_do_not_count(self):
        points = pd.DataFrame(
            [
                {"clip_id": "clip_a", "vertex": v, "visible": v in (1, 2, 9, 10)}
                for v in range(1, 31)
            ]
        )

        assert observed_camera_mode(points, "clip_a") is None

    def test_a_clip_with_no_court_settles_nothing(self):
        points = pd.DataFrame([{"clip_id": "clip_b", "vertex": 1, "visible": True}])

        assert observed_camera_mode(points, "clip_a") is None


class TestSourceUri:
    """
    Where a clip says its recording is. It has to resolve wherever the release
    is unpacked, not just on the machine that wrote it.
    """

    def test_a_recording_inside_the_dataset_gets_a_relative_path(self, tmp_path):
        root = tmp_path / "dataset"
        videos = root / "videos"
        videos.mkdir(parents=True)
        video = videos / "clip_a.mp4"
        video.write_bytes(b"")

        assert source_uri(video, root) == "videos/clip_a.mp4"

    def test_a_recording_outside_the_dataset_gets_nothing(self, tmp_path):
        # "../somewhere" resolves on this machine and nowhere else, which is
        # worse than an honest blank.
        root = tmp_path / "dataset"
        root.mkdir()
        outside = tmp_path / "elsewhere" / "clip_a.mp4"
        outside.parent.mkdir()
        outside.write_bytes(b"")

        assert source_uri(outside, root) is None

    def test_without_a_dataset_root_there_is_nothing_to_be_relative_to(self, tmp_path):
        assert source_uri(tmp_path / "clip_a.mp4", None) is None

    def test_it_is_filled_in_when_the_recording_is_carried(self, video, tmp_path):
        root = tmp_path
        rows = clip_rows({"clip_a": video}, dataset_root=root)

        assert rows[0]["source_uri"] == video.name

    def test_a_uri_given_by_hand_is_kept(self, video, tmp_path):
        existing = pd.DataFrame(
            [
                {
                    **{c: None for c in columns("clips")},
                    "clip_id": "clip_a",
                    "source_uri": "https://example.org/clip_a.mp4",
                }
            ]
        )

        rows = clip_rows({"clip_a": video}, existing=existing, dataset_root=tmp_path)

        assert rows[0]["source_uri"] == "https://example.org/clip_a.mp4"
