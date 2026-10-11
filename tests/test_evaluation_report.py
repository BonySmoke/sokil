"""
The record an evaluation run leaves behind.

A metric without the conditions that produced it cannot be compared with a
later one, so what matters here is that the record carries the thresholds, the
commit and the machine — and that a dirty checkout says so, since the sha alone
would then be a lie.
"""

import json

import pytest

from sokil.evaluation import report
from sokil.evaluation.run import ClipResult, summarise
from sokil.hits import HitDetectionConfig


def clip_result(clip_id="clip_a", labelled=4, matched=3, missed=1, false_positives=2):
    return ClipResult(
        clip_id=clip_id,
        configuration="manual",
        detection={
            "labelled": labelled,
            "detected": matched + false_positives,
            "matched": matched,
            "missed": missed,
            "false_positives": false_positives,
            "precision": 0.6,
            "recall": 0.75,
            "f1": 0.667,
            "missed_frames": [100],
            "false_positive_frames": [500, 600],
            "frame_error": [0, 1, -1],
        },
        position_error={
            "count": matched,
            "median_cm": 12.0,
            "p90_cm": 20.0,
            "by_distance": {},
        },
        verdict_correctness={
            "count": matched,
            "accuracy": 1.0,
            "called_in_was_out": 0,
            "called_out_was_in": 0,
        },
        court_error={"vertices": 12, "median_px": 1.5, "max_px": 4.0},
        frames_tracked=1000,
    )


class TestSummarise:
    def test_counts_are_pooled_not_averaged(self):
        # A clip with 40 landings and one with 4 are not equal evidence, and
        # averaging their rates would pretend they were.
        totals = summarise(
            [
                clip_result(
                    "big", labelled=40, matched=40, missed=0, false_positives=0
                ),
                clip_result(
                    "small", labelled=4, matched=0, missed=4, false_positives=0
                ),
            ]
        )

        assert totals["detection"]["labelled"] == 44
        assert totals["detection"]["recall"] == round(40 / 44, 4)

    def test_the_counts_add_up(self):
        totals = summarise([clip_result(), clip_result("clip_b")])

        detection = totals["detection"]
        assert detection["matched"] + detection["missed"] == detection["labelled"]
        assert (
            detection["matched"] + detection["false_positives"] == detection["detected"]
        )

    def test_clips_are_counted(self):
        assert summarise([clip_result(), clip_result("clip_b")])["clips"] == 2


@pytest.fixture
def written_record(tmp_path):
    return report.record(
        [clip_result()], HitDetectionConfig(), tmp_path, ["clip_a"], {}
    )


class TestRecord:
    def test_carries_the_thresholds_under_test(self):
        # The figure means nothing without the configuration that produced it.
        config = HitDetectionConfig(angle_change_threshold=95.0)

        written = report.record([clip_result()], config, "dataset", ["clip_a"], {})

        assert written["hit_config"]["angle_change_threshold"] == 95.0

    def test_carries_what_could_change_a_figure(self, written_record):
        # The versions that decode the frames and do the arithmetic, and the
        # architecture, which decides which SIMD paths OpenCV takes.
        environment = written_record["environment"]

        assert environment["opencv"]
        assert environment["numpy"]
        assert environment["python"]
        assert environment["architecture"]

    def test_leaves_out_what_could_not(self, written_record):
        # Whether a line call is right cannot depend on which laptop asked, and
        # a hostname is usually somebody's name. Those belong in a benchmark.
        environment = written_record["environment"]

        assert "host" not in environment
        assert "platform" not in environment
        assert "processor" not in environment

    def test_carries_the_commit_and_whether_it_was_clean(self, written_record):
        assert "sha" in written_record["git"]
        assert "dirty" in written_record["git"]

    def test_carries_which_clips_were_measured(self, tmp_path):
        written = report.record(
            [clip_result()], HitDetectionConfig(), tmp_path, ["clip_a", "clip_b"], {}
        )

        assert written["dataset"]["clips"] == ["clip_a", "clip_b"]

    def test_is_written_as_readable_json(self, tmp_path):
        written = report.record(
            [clip_result()], HitDetectionConfig(), tmp_path, ["clip_a"], {}
        )

        path = report.write(written, tmp_path / "evaluations")

        reread = json.loads(path.read_text())
        assert reread["clips"][0]["clip_id"] == "clip_a"

    def test_a_run_never_overwrites_an_earlier_one(self, tmp_path):
        # Runs are compared by reading two of them, so one must not replace
        # another. The name carries the time, so a later run sorts after.
        first = report.record(
            [clip_result()], HitDetectionConfig(), tmp_path, ["clip_a"], {}
        )
        second = dict(first, at="2030-01-01T00:00:00+00:00")

        a = report.write(first, tmp_path / "evaluations")
        b = report.write(second, tmp_path / "evaluations")

        assert a != b
        assert b.name > a.name


class TestDatasetState:
    """
    The tables a figure rests on. Re-deriving a landing position moves every
    distance in the report, so a record that cannot show the dataset changed
    cannot explain the difference.
    """

    def write_tables(self, directory, events: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "events.csv").write_text(events)
        (directory / "clips.csv").write_text("clip_id\nclip_a\n")

    def test_digests_each_table_it_finds(self, tmp_path):
        self.write_tables(tmp_path, "clip_id,frame\nclip_a,10\n")

        state = report.dataset_state(tmp_path, ["clip_a"])

        assert len(state["tables"]["events"]["sha256"]) == 64
        assert state["tables"]["events"]["rows"] == 1

    def test_a_changed_table_changes_its_digest(self, tmp_path):
        self.write_tables(tmp_path, "clip_id,frame\nclip_a,10\n")
        before = report.dataset_state(tmp_path, ["clip_a"])

        self.write_tables(tmp_path, "clip_id,frame\nclip_a,11\n")
        after = report.dataset_state(tmp_path, ["clip_a"])

        assert (
            before["tables"]["events"]["sha256"] != after["tables"]["events"]["sha256"]
        )

    def test_the_same_table_digests_the_same(self, tmp_path):
        self.write_tables(tmp_path, "clip_id,frame\nclip_a,10\n")

        first = report.dataset_state(tmp_path, ["clip_a"])
        second = report.dataset_state(tmp_path, ["clip_a"])

        assert first == second

    def test_a_table_that_does_not_exist_is_simply_absent(self, tmp_path):
        self.write_tables(tmp_path, "clip_id,frame\nclip_a,10\n")

        state = report.dataset_state(tmp_path, ["clip_a"])

        assert "tracks" not in state["tables"]
        assert "events" in state["tables"]

    def test_carries_which_clips_were_measured(self, tmp_path):
        self.write_tables(tmp_path, "clip_id,frame\nclip_a,10\n")

        state = report.dataset_state(tmp_path, ["clip_a", "clip_b"])

        assert state["clips"] == ["clip_a", "clip_b"]


class TestPooledCourtError:
    """
    One figure for how well the courts were placed, over every clip.

    A flipped court's error is measured after unflipping, so folding it in
    would report a court centimetres out when it is the length of the court
    out. A clip with no court contributes an absence, not a zero.
    """

    def with_court(self, clip_id, median=None, flipped=False, solved=True):
        result = clip_result(clip_id)
        result.court_solved = solved
        result.court_error = (
            {}
            if median is None
            else {
                "vertices": 12,
                "annotation_px": {"median": 1.5, "max": 4.0},
                "vertex_cm": {"median": median, "max": median * 2},
                "flipped_end_for_end": flipped,
            }
        )
        return result

    def test_the_median_of_the_per_clip_medians(self):
        totals = summarise(
            [
                self.with_court("a", median=4.0),
                self.with_court("b", median=6.0),
                self.with_court("c", median=11.0),
            ]
        )

        assert totals["court_error"]["median_cm"] == 6.0

    def test_a_flipped_court_is_counted_not_averaged_in(self):
        totals = summarise(
            [
                self.with_court("a", median=4.0),
                self.with_court("b", median=3.0, flipped=True),
            ]
        )

        assert totals["court_error"]["median_cm"] == 4.0
        assert totals["court_error"]["flipped"] == 1

    def test_a_clip_with_no_court_is_counted_as_unsolved(self):
        totals = summarise(
            [self.with_court("a", median=4.0), self.with_court("b", solved=False)]
        )

        assert totals["court_error"]["unsolved"] == 1
        assert totals["court_error"]["clips"] == 1

    def test_no_court_anywhere_reports_nothing_rather_than_zero(self):
        totals = summarise([self.with_court("a", solved=False)])

        assert totals["court_error"]["median_cm"] is None
