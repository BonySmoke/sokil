"""
The annotation dataset's tables: what they accept, and what they catch.

The cross-table checks carry the weight here. A bad enum value is caught by
anything; a landing pointing at a clip that no longer exists, or a clip sitting
in two splits at once, survives a per-file review and then quietly inflates
every figure measured afterwards.
"""

import pandas as pd
import pytest

from sokil.dataset import Dataset, DatasetError, columns


def clip_row(clip_id: str = "clip_a", **overrides) -> dict:
    row = {
        "clip_id": clip_id,
        "source_uri": None,
        "frame_count": 1000,
        "fps": 120.0,
        "width": 1920,
        "height": 1080,
        "session_id": "session-01",
        "camera_mode": "left_corridor",
        "court_type": "doubles",
        "camera_static": True,
        "notes": None,
    }
    return {**row, **overrides}


def event_row(clip_id: str = "clip_a", frame: int = 73, **overrides) -> dict:
    row = {
        "clip_id": clip_id,
        "frame": frame,
        "call": "out",
        "court_type": "doubles",
        "line": "sideline",
        "x_px": 812.5,
        "y_px": 604.25,
        "x_m": 3.1416,
        "y_m": 5.25,
        "annotator": "user",
        "notes": None,
    }
    return {**row, **overrides}


def build(tmp_path, clips: list[dict], events: list[dict], splits=None) -> Dataset:
    dataset = Dataset(tmp_path)
    dataset.write("clips", pd.DataFrame(clips, columns=columns("clips")))
    dataset.write("events", pd.DataFrame(events, columns=columns("events")))
    if splits is not None:
        dataset.write_splits(splits)
    return dataset


@pytest.fixture
def dataset(tmp_path) -> Dataset:
    return build(
        tmp_path,
        [clip_row("clip_a"), clip_row("clip_b")],
        [event_row("clip_a", 73), event_row("clip_b", 412)],
        {"train": ["clip_a"], "valid": ["clip_b"], "test": []},
    )


class TestRoundTrip:
    def test_a_valid_dataset_has_no_problems(self, dataset):
        assert dataset.validate() == []

    def test_reads_back_what_it_wrote(self, dataset):
        assert list(dataset.read("clips")["clip_id"]) == ["clip_a", "clip_b"]
        assert list(dataset.read("events")["frame"]) == [73, 412]

    def test_writes_rows_sorted_and_columns_in_order(self, tmp_path):
        dataset = build(
            tmp_path,
            [clip_row("clip_b"), clip_row("clip_a")],
            [event_row("clip_b", 9), event_row("clip_a", 5)],
        )

        lines = dataset.path("events").read_text().splitlines()
        assert lines[0] == ",".join(columns("events"))
        assert [line.split(",")[0] for line in lines[1:]] == ["clip_a", "clip_b"]

    def test_rounds_on_write_so_floats_do_not_churn(self, tmp_path):
        dataset = build(
            tmp_path,
            [clip_row()],
            [event_row(x_px=812.5555555, x_m=3.14159265)],
        )

        row = dataset.path("events").read_text().splitlines()[1]
        assert "812.56" in row
        assert "3.1416" in row

    def test_an_unset_cell_stays_empty(self, tmp_path):
        dataset = build(tmp_path, [clip_row(notes=None)], [])

        assert dataset.path("clips").read_text().splitlines()[1].endswith(",")

    def test_a_missing_table_says_so(self, tmp_path):
        with pytest.raises(DatasetError, match="No clips table"):
            Dataset(tmp_path).read("clips")


class TestSchemaChecks:
    def test_rejects_an_unknown_call(self, tmp_path):
        dataset = build(tmp_path, [clip_row()], [event_row(call="maybe")])

        assert any("call" in problem for problem in dataset.validate())

    def test_rejects_a_clip_id_that_would_not_survive_a_path(self, tmp_path):
        dataset = build(tmp_path, [clip_row("Clip A")], [])

        assert any("clip_id" in problem for problem in dataset.validate())

    def test_a_source_that_is_not_there_is_reported(self, tmp_path):
        # An empty cell says the video is not here; a dangling path says it is.
        dataset = build(tmp_path, [clip_row(source_uri="videos/gone.mp4")], [])

        assert any("not a file under" in problem for problem in dataset.validate())

    def test_a_source_that_is_there_is_accepted(self, tmp_path):
        (tmp_path / "videos").mkdir()
        (tmp_path / "videos" / "clip_a.mp4").write_bytes(b"")

        dataset = build(tmp_path, [clip_row(source_uri="videos/clip_a.mp4")], [])

        assert dataset.validate() == []

    def test_a_url_is_taken_on_trust(self, tmp_path):
        # Checking it would mean reaching over the network to validate a table.
        dataset = build(
            tmp_path, [clip_row(source_uri="https://example.org/clip_a.mp4")], []
        )

        assert dataset.validate() == []

        assert any("source_uri" in problem for problem in dataset.validate())

    def test_accepts_a_url_as_a_source(self, tmp_path):
        dataset = build(
            tmp_path, [clip_row(source_uri="https://example.org/a.mp4")], []
        )

        assert dataset.validate() == []

    def test_accepts_a_clip_whose_video_is_not_published(self, tmp_path):
        dataset = build(tmp_path, [clip_row(source_uri=None)], [])

        assert dataset.validate() == []


class TestCrossTableChecks:
    def test_catches_a_landing_on_a_clip_that_does_not_exist(self, tmp_path):
        dataset = build(tmp_path, [clip_row("clip_a")], [event_row("clip_ghost", 5)])

        assert any("not in clips.csv" in problem for problem in dataset.validate())

    def test_catches_a_landing_past_the_end_of_its_clip(self, tmp_path):
        dataset = build(
            tmp_path, [clip_row(frame_count=100)], [event_row("clip_a", 4000)]
        )

        assert any("past the 100 frames" in problem for problem in dataset.validate())

    def test_catches_two_landings_on_the_same_frame(self, tmp_path):
        dataset = build(
            tmp_path, [clip_row()], [event_row("clip_a", 73), event_row("clip_a", 73)]
        )

        assert any("clip_id" in problem for problem in dataset.validate())

    def test_catches_a_verdict_with_no_boundary_to_judge_it_against(self, tmp_path):
        dataset = build(tmp_path, [clip_row()], [event_row(court_type=None)])

        assert any("without a court_type" in problem for problem in dataset.validate())

    def test_allows_an_unknown_call_to_have_no_boundary(self, tmp_path):
        dataset = build(
            tmp_path,
            [clip_row()],
            [event_row(call="unknown", court_type=None)],
        )

        assert dataset.validate() == []

    def test_catches_court_metres_with_no_pixel_behind_them(self, tmp_path):
        dataset = build(tmp_path, [clip_row()], [event_row(x_px=None, y_px=None)])

        assert any("no clicked pixel" in problem for problem in dataset.validate())

    def test_catches_a_camera_mode_the_pipeline_cannot_read(self, tmp_path):
        dataset = build(tmp_path, [clip_row(camera_mode="sideways")], [])

        assert any("camera_mode" in problem for problem in dataset.validate())


class TestSplitChecks:
    def test_catches_a_clip_in_two_splits(self, tmp_path):
        dataset = build(
            tmp_path,
            [clip_row("clip_a"), clip_row("clip_b")],
            [],
            {"train": ["clip_a", "clip_b"], "valid": ["clip_b"], "test": []},
        )

        assert any("belongs to exactly one" in p for p in dataset.validate())

    def test_catches_a_split_naming_a_clip_that_does_not_exist(self, tmp_path):
        dataset = build(
            tmp_path, [clip_row("clip_a")], [], {"train": ["clip_a", "clip_gone"]}
        )

        assert any("not in clips.csv" in problem for problem in dataset.validate())

    def test_catches_a_clip_left_out_of_every_split(self, tmp_path):
        dataset = build(
            tmp_path,
            [clip_row("clip_a"), clip_row("clip_b")],
            [],
            {"train": ["clip_a"]},
        )

        assert any("are in no split" in problem for problem in dataset.validate())

    def test_says_nothing_when_no_splits_are_drawn_yet(self, tmp_path):
        dataset = build(tmp_path, [clip_row("clip_a")], [])

        assert dataset.validate() == []

    def test_raising_reports_every_problem_at_once(self, tmp_path):
        dataset = build(
            tmp_path,
            [clip_row("clip_a")],
            [event_row("clip_ghost", 5), event_row("clip_a", 99999)],
        )

        with pytest.raises(DatasetError) as caught:
            dataset.validate_or_raise()

        assert "clip_ghost" in str(caught.value)
        assert "past the 1000 frames" in str(caught.value)


def court_rows(
    clip_id: str = "clip_a",
    vertices=(1, 2, 9, 10, 11, 12, 19, 20, 21, 22, 29, 30),
) -> list[dict]:
    """
    Keypoints taken from a real projection, so the court fits itself.

    The left corridor's own vertices by default, so camera_mode is derivable
    from them as it is on a real clip.
    """
    import numpy as np

    from sokil.court import CourtModel

    homography = np.array(
        [[1.6, 0.35, 240.0], [0.0, -0.35, 560.0], [0.0, -0.00045, 1.6]]
    )
    model = CourtModel().vertices

    rows = []
    for vertex in vertices:
        x_cm, y_cm = model[vertex - 1]
        placed = homography @ np.array([x_cm, y_cm, 1.0])
        rows.append(
            {
                "clip_id": clip_id,
                "frame": 0,
                "vertex": vertex,
                "x_px": float(placed[0] / placed[2]),
                "y_px": float(placed[1] / placed[2]),
                "visible": True,
                "source": "manual",
            }
        )
    return rows


def track_row(clip_id: str = "clip_a", frame: int = 73, **overrides) -> dict:
    row = {
        "clip_id": clip_id,
        "frame": frame,
        "x1": 100.0,
        "y1": 200.0,
        "x2": 120.0,
        "y2": 224.0,
    }
    return {**row, **overrides}


def with_tracks(tmp_path, clips, events, rows) -> Dataset:
    dataset = build(tmp_path, clips, events)
    dataset.write("tracks", pd.DataFrame(rows, columns=columns("tracks")))
    return dataset


class TestTrackChecks:
    """
    Whether the ground-truth track can stand in for the detector. The last
    check is the one that matters: a landing with no box on its own frame
    cannot be measured with the models taken out.
    """

    def test_a_consistent_track_has_no_problems(self, tmp_path):
        dataset = with_tracks(
            tmp_path, [clip_row()], [event_row("clip_a", 73)], [track_row(frame=73)]
        )

        assert dataset.validate() == []

    def test_tracks_are_optional(self, tmp_path):
        dataset = build(tmp_path, [clip_row()], [event_row("clip_a", 73)])

        assert dataset.validate() == []

    def test_catches_a_track_for_a_clip_that_does_not_exist(self, tmp_path):
        dataset = with_tracks(
            tmp_path, [clip_row("clip_a")], [], [track_row("clip_ghost")]
        )

        assert any("not in clips.csv" in p for p in dataset.validate())

    def test_catches_a_box_past_the_end_of_the_clip(self, tmp_path):
        dataset = with_tracks(
            tmp_path, [clip_row(frame_count=100)], [], [track_row(frame=4000)]
        )

        assert any("past the 100 frames" in p for p in dataset.validate())

    def test_catches_a_box_with_no_area(self, tmp_path):
        dataset = with_tracks(
            tmp_path, [clip_row()], [], [track_row(x1=120.0, x2=100.0)]
        )

        assert any("no area" in p for p in dataset.validate())

    def test_catches_a_box_reaching_past_the_frame(self, tmp_path):
        dataset = with_tracks(
            tmp_path, [clip_row(width=640, height=480)], [], [track_row(x2=900.0)]
        )

        assert any("past its 640x480 frame" in p for p in dataset.validate())

    def test_catches_a_landing_with_no_box_on_its_own_frame(self, tmp_path):
        dataset = with_tracks(
            tmp_path,
            [clip_row()],
            [event_row("clip_a", 73)],
            [track_row(frame=70), track_row(frame=71)],
        )

        assert any("no ground-truth box on" in p for p in dataset.validate())

    def test_a_clip_with_no_track_at_all_is_not_faulted_for_its_landings(
        self, tmp_path
    ):
        # Tracks imported for one clip say nothing about a clip not imported yet.
        dataset = with_tracks(
            tmp_path,
            [clip_row("clip_a"), clip_row("clip_b")],
            [event_row("clip_b", 10)],
            [track_row("clip_a", frame=5)],
        )

        assert not any("no ground-truth box" in p for p in dataset.validate())


class TestStalePositionChecks:
    """
    Landings whose court exists but whose position was never worked out.

    A derived value stored once goes stale silently — the click is kept, the
    metres are lost, and nothing about the row says so. Four real landings sat
    like this and the dataset validated clean, which is what this check is for.
    """

    def with_court_and_events(self, tmp_path, events):
        dataset = build(tmp_path, [clip_row("clip_a")], events)
        dataset.write(
            "court_points",
            pd.DataFrame(
                [
                    {
                        "clip_id": "clip_a",
                        "frame": 0,
                        "vertex": vertex,
                        "x_px": 10.0 * vertex,
                        "y_px": 20.0 * vertex,
                        "visible": True,
                        "source": "manual",
                    }
                    for vertex in (1, 5, 26, 30)
                ],
                columns=columns("court_points"),
            ),
        )
        return dataset

    def test_a_click_with_no_position_is_reported(self, tmp_path):
        dataset = self.with_court_and_events(
            tmp_path, [event_row("clip_a", 73, x_m=None, y_m=None)]
        )

        assert any("no position" in problem for problem in dataset.validate())

    def test_the_fix_is_named(self, tmp_path):
        dataset = self.with_court_and_events(
            tmp_path, [event_row("clip_a", 73, x_m=None, y_m=None)]
        )

        assert any("dataset refresh" in problem for problem in dataset.validate())

    def test_a_landing_with_a_position_is_not_reported(self, tmp_path):
        dataset = self.with_court_and_events(tmp_path, [event_row("clip_a", 73)])

        assert not any("no position" in problem for problem in dataset.validate())

    def test_a_landing_with_no_click_is_not_reported(self, tmp_path):
        # Nothing to derive from, so nothing stale: that is an unplaced
        # landing, not an out-of-date one.
        dataset = self.with_court_and_events(
            tmp_path,
            [event_row("clip_a", 73, x_px=None, y_px=None, x_m=None, y_m=None)],
        )

        assert not any("no position" in problem for problem in dataset.validate())

    def test_a_clip_with_no_court_is_not_reported(self, tmp_path):
        # The court being there is what makes the absence a fault rather than
        # a stage not reached.
        dataset = build(
            tmp_path,
            [clip_row("clip_a")],
            [event_row("clip_a", 73, x_m=None, y_m=None)],
        )

        assert not any("no position" in problem for problem in dataset.validate())


class TestStaleDerivationChecks:
    """
    Derived values that no longer agree with their source.

    Most of what the tables hold is worked out rather than typed, and each is
    written down once — so each goes stale the moment its source changes,
    silently, because nothing about a filled-in cell says when it was filled.
    Catching it here is what lets the order of work stop mattering.
    """

    def with_court(self, tmp_path, clips, events):
        dataset = build(tmp_path, clips, events)
        dataset.write(
            "court_points",
            pd.DataFrame(court_rows(), columns=columns("court_points")),
        )
        return dataset

    def test_a_court_type_that_contradicts_its_landings_is_reported(self, tmp_path):
        dataset = self.with_court(
            tmp_path,
            [clip_row("clip_a", court_type="singles")],
            [event_row("clip_a", 73, court_type="doubles")],
        )

        problems = dataset.validate()

        assert any("were called against 'doubles'" in problem for problem in problems)

    def test_a_camera_mode_that_contradicts_its_keypoints_is_reported(self, tmp_path):
        dataset = self.with_court(
            tmp_path,
            [clip_row("clip_a", camera_mode="right_corridor")],
            [event_row("clip_a", 73)],
        )

        problems = dataset.validate()

        assert any("court keypoints say" in problem for problem in problems)

    def test_the_fix_is_named_once(self, tmp_path):
        dataset = self.with_court(
            tmp_path,
            [clip_row("clip_a", court_type="singles", camera_mode="right_corridor")],
            [event_row("clip_a", 73, court_type="doubles")],
        )

        problems = dataset.validate()

        assert sum("dataset refresh" in problem for problem in problems) == 1

    def test_values_that_agree_are_not_reported(self, tmp_path):
        dataset = self.with_court(
            tmp_path,
            [clip_row("clip_a", court_type="doubles", camera_mode="left_corridor")],
            [event_row("clip_a", 73, court_type="doubles")],
        )

        assert dataset.validate() == []

    def test_landings_that_disagree_settle_nothing_and_report_nothing(self, tmp_path):
        # The clip's own answer stands when the landings cannot agree on one.
        dataset = self.with_court(
            tmp_path,
            [clip_row("clip_a", court_type="doubles", camera_mode="left_corridor")],
            [
                event_row("clip_a", 73, court_type="doubles"),
                event_row("clip_a", 90, court_type="singles"),
            ],
        )

        assert not any("called against" in problem for problem in dataset.validate())
