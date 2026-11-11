"""
Run stats: stages timed end to end, models' own measurements, and the report.
"""

import json

import pytest

from sokil import timing
from sokil.timing import RunStats


@pytest.fixture
def clock(monkeypatch):
    """A perf_counter that only moves when a test says so."""
    now = [100.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(timing.sokil, "STARTED", 90.0)
    return now


class FakeModel:
    def __init__(self, loads=1, load_seconds=1.2, inferences=10, inference_seconds=0.5):
        self.model_path = "models/court-hailo8_hailo_model"
        self.loads = loads
        self.load_seconds = load_seconds
        self.inferences = inferences
        self.inference_seconds = inference_seconds


class TestStages:
    def test_a_stage_is_timed_and_counts_its_frames(self, clock):
        stats = RunStats()

        with stats.stage("tracking") as stage:
            clock[0] += 2.0
            stage.frames = 94

        assert stats.stages[0].seconds == 2.0
        assert stats.stages[0].fps == 47.0

    def test_a_stage_that_fails_is_still_timed(self, clock):
        stats = RunStats()

        with pytest.raises(RuntimeError), stats.stage("render"):
            clock[0] += 1.0
            raise RuntimeError("disk full")

        assert stats.stages[0].seconds == 1.0

    def test_a_stage_run_again_adds_to_its_row(self, clock):
        stats = RunStats()
        for _ in range(3):
            with stats.stage("hit detection"):
                clock[0] += 0.5

        assert len(stats.stages) == 1
        assert stats.stages[0].seconds == 1.5

    def test_a_stage_without_frames_has_no_fps(self, clock):
        stats = RunStats()
        with stats.stage("in/out judging"):
            clock[0] += 0.1

        assert stats.stages[0].fps is None


class TestModels:
    def test_a_model_s_fps_comes_from_its_inference_time_alone(self):
        stats = RunStats()

        stats.add_model("segmenter", FakeModel(inferences=10, inference_seconds=0.5))

        assert stats.models[0].seconds_per_frame == 0.05
        assert stats.models[0].fps == 20.0

    def test_an_absent_or_unused_model_is_left_out(self):
        stats = RunStats()

        stats.add_model("keypoints", None)
        stats.add_model("detector", FakeModel(loads=0, inferences=0))

        assert stats.models == []


class TestReport:
    def test_startup_and_total_count_from_the_package_import(self, clock):
        stats = RunStats()

        stats.start()
        clock[0] += 5.0

        assert stats.startup_seconds == 10.0
        assert stats.total_seconds == 15.0

    def test_the_json_holds_every_stage_and_model(self, clock, tmp_path):
        stats = RunStats()
        stats.start()
        with stats.stage("tracking") as stage:
            clock[0] += 2.0
            stage.frames = 94
        stats.add_model("detector", FakeModel())

        stats.write(tmp_path / "stats.json")
        written = json.loads((tmp_path / "stats.json").read_text())

        assert written["startup_seconds"] == 10.0
        assert written["stages"] == [
            {"name": "tracking", "seconds": 2.0, "frames": 94, "fps": 47.0}
        ]
        assert written["models"][0]["fps"] == 20.0
        assert written["models"][0]["load_seconds"] == 1.2

    def test_the_table_has_a_row_per_stage_and_two_per_model(self, clock):
        stats = RunStats()
        stats.start()
        with stats.stage("tracking"):
            pass
        stats.add_model("detector", FakeModel())

        table = stats.table()

        assert table.row_count == 1 + 1 + 2
