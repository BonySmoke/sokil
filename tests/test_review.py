"""
Review: the order the pipeline runs its stages in and what it hands between
them. Every stage that needs a checkpoint or a decode pass is replaced by a
fake, so what is tested here is the wiring, not the vision.
"""

import numpy as np
import pytest

from sokil import review as review_module
from sokil.court import Court, CourtSolver, CourtSolverConfig
from sokil.frame import Hit
from sokil.hits import HitDetectionConfig, HitResult
from sokil.render import RenderConfig
from sokil.review import Review
from sokil.tracker import TrackerConfig
from sokil.video import DecodedFrame

FPS = 30.0
WIDTH, HEIGHT = 1920, 1080


class FakeSource:
    """Stands in for a VideoSource without a file behind it."""

    def __init__(self, frames=()):
        self.fps = FPS
        self.width = WIDTH
        self.height = HEIGHT
        self.frame_count = len(frames)
        self._frames = list(frames)
        self.sampled = []

    def __iter__(self):
        return iter(self._frames)

    def sample(self, stride):
        self.sampled.append(stride)
        return iter(self._frames[::stride])

    def describe(self):
        return "fake source"


class FakeTracker:
    def __init__(self, tracks=()):
        self.tracks = list(tracks)
        self.sources = []

    def track(self, source):
        self.sources.append(source)
        return self.tracks


class FakeSolver:
    def __init__(self, court=None):
        self.court = court
        self.court_detector_kwargs = {}
        self.consensus_verticals = []
        self.consensus_horizontals = []
        self.calls = 0

    def solve(self, source):
        self.calls += 1
        return self.court


class FakeRenderer:
    def __init__(self, config=None):
        self.config = config or RenderConfig()
        self.rendered = []
        self.assigned = []

    def assign_track_points(self, tracks, trajectories):
        self.assigned.append((tracks, trajectories))

    def render(self, source, tracks, court, output_path, explainer=None):
        self.rendered.append((source, tracks, court, output_path, explainer))


@pytest.fixture
def review(monkeypatch, tmp_path, make_track):
    """
    A Review with every model-backed stage replaced.

    The constructor builds a ShuttleTracker (which loads a checkpoint) and a
    CourtSolver, so both classes are stubbed for the duration.
    """
    monkeypatch.setattr(review_module, "ShuttleTracker", lambda *a, **k: FakeTracker())
    monkeypatch.setattr(review_module, "CourtSolver", lambda *a, **k: FakeSolver())

    review = Review(
        video_path=tmp_path / "clip.mp4",
        court_detector_kwargs={},
        shuttle_detector_kwargs={},
    )
    review._source = FakeSource(frames=[object()] * 4)
    review.renderer = FakeRenderer()
    return review


def bounce_tracks(make_track):
    """A descending flight that turns sharply: one detectable landing."""
    positions = [(100 + 3 * i, 100 + 14 * i) for i in range(8)]
    contact_x, contact_y = positions[-1]
    positions += [(contact_x + 3 * (i + 1), contact_y - 14 * (i + 1)) for i in range(8)]
    return [
        make_track(
            number=number,
            xyxy=(x - 5, y - 5, x + 5, y + 5),
            measured_cork=(x, y),
        )
        for number, (x, y) in enumerate(positions)
    ]


class TestConstruction:
    def test_the_video_path_becomes_a_path(self, review, tmp_path):
        assert review.video_path == tmp_path / "clip.mp4"

    def test_the_results_start_empty(self, review):
        assert review.tracks == []
        assert review.court is None
        assert review.trajectories == []
        assert review.hit_result == HitResult()

    def test_the_source_properties_come_from_the_video(self, review):
        assert review.fps == FPS
        assert review.frame_width == WIDTH
        assert review.frame_height == HEIGHT

    def test_the_configs_are_passed_through(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            review_module, "ShuttleTracker", lambda *a, **k: FakeTracker()
        )
        monkeypatch.setattr(review_module, "CourtSolver", lambda *a, **k: FakeSolver())
        hit_config = HitDetectionConfig(window=9)
        render_config = RenderConfig(explain=True)

        review = Review(
            video_path=tmp_path / "clip.mp4",
            court_detector_kwargs={},
            shuttle_detector_kwargs={},
            hit_config=hit_config,
            render_config=render_config,
        )

        assert review.hit_config is hit_config
        assert review.hit_detector.config is hit_config
        assert review.renderer.config is render_config


class TestUndistortion:
    def test_off_when_no_calibration_is_given(self, review):
        assert review._get_undistorter() is None

    def test_off_when_only_one_of_the_two_files_is_given(self, review, tmp_path):
        review.camera_intrinsics_path = tmp_path / "K.npy"

        assert review._get_undistorter() is None

    def test_on_when_both_files_are_given(self, review, tmp_path):
        K = np.array(
            [[WIDTH, 0, WIDTH / 2], [0, WIDTH, HEIGHT / 2], [0, 0, 1]], dtype=np.float64
        )
        np.save(tmp_path / "K.npy", K)
        np.save(tmp_path / "dist.npy", np.zeros(5))
        review.camera_intrinsics_path = tmp_path / "K.npy"
        review.camera_dist_path = tmp_path / "dist.npy"

        assert review._get_undistorter() is not None


class TestPreprocess:
    def test_detects_the_shuttle_then_solves_the_court(self, review, make_track):
        tracks = bounce_tracks(make_track)
        review.tracker = FakeTracker(tracks)
        solved = Court.from_homography(np.eye(3))
        review.court_solver = FakeSolver(solved)

        review.preprocess()

        assert review.tracks == tracks
        assert review.court is solved

    def test_a_supplied_homography_is_used_instead_of_solving(self, review):
        review.court_homography = np.eye(3)
        review.court_solver = FakeSolver(Court.from_homography(np.eye(3)))

        review.preprocess()

        assert review.court_solver.calls == 0
        assert review.court.get_view_transformer() is not None

    def test_an_unsolvable_court_leaves_the_review_without_one(self, review):
        review.court_solver = FakeSolver(None)

        review.preprocess()

        assert review.court is None


class TestStages:
    def test_hit_detection_fills_the_result_and_the_trajectories(
        self, review, make_track
    ):
        review.tracks = bounce_tracks(make_track)

        hit_indices = review.set_frame_hit_candidates()

        assert len(hit_indices) == 1
        assert review.trajectories == review.hit_result.trajectories
        assert review.hit_result.candidate_count == 1

    def test_the_detector_is_told_this_videos_frame_width(self, review, make_track):
        # the teleport threshold is a share of the frame width, so the detector
        # cannot resolve it without being told what this video is
        review.tracks = bounce_tracks(make_track)
        seen = {}

        def spy(tracks, court, fps, config=None, frame_width=None):
            seen.update(fps=fps, frame_width=frame_width)
            return HitResult()

        review.hit_detector.detect = spy

        review.set_frame_hit_candidates()

        assert seen == {"fps": FPS, "frame_width": WIDTH}

    def test_a_per_call_config_overrides_the_reviews_own(self, review, make_track):
        review.tracks = bounce_tracks(make_track)
        # ground_hits_only rejects nothing here, but a huge angle threshold does
        strict = HitDetectionConfig(
            angle_change_threshold=179.0, speed_drop_threshold=1.0
        )

        assert review.set_frame_hit_candidates(strict) == []

    def test_judging_needs_a_court(self, review, make_track):
        review.tracks = bounce_tracks(make_track)
        review.set_frame_hit_candidates()

        assert review.set_hit_positions() == 0

    def test_with_a_court_every_hit_gets_a_verdict(self, review, make_track):
        review.tracks = bounce_tracks(make_track)
        review.court = Court.from_homography(np.eye(3))
        review.set_frame_hit_candidates()

        judged = review.set_hit_positions()

        assert judged == 1
        assert all(track.hit.is_in is not None for track in review.tracks if track.hit)

    def test_the_track_points_are_handed_to_the_renderer(self, review, make_track):
        review.tracks = bounce_tracks(make_track)
        review.set_frame_hit_candidates()

        review.set_track_points()

        tracks, trajectories = review.renderer.assigned[0]
        assert tracks is review.tracks
        assert trajectories is review.trajectories


class TestRendering:
    def test_the_renderer_is_given_the_source_tracks_and_court(self, review):
        review.court = Court.from_homography(np.eye(3))

        review.video_to_file("out.mp4")

        source, tracks, court, output_path, _ = review.renderer.rendered[0]
        assert source is review.source
        assert tracks is review.tracks
        assert court is review.court
        assert output_path == "out.mp4"

    def test_no_explainer_unless_it_was_asked_for(self, review):
        assert review._explainer() is None

    def test_the_explainer_carries_the_solvers_consensus_lines(self, review):
        review.renderer.config = RenderConfig(explain=True)
        review.court_solver.consensus_verticals = ["v"]
        review.court_solver.consensus_horizontals = ["h"]

        explainer = review._explainer()

        assert explainer.consensus_verticals == ["v"]
        assert explainer.consensus_horizontals == ["h"]


class TestRun:
    def test_runs_every_stage_in_order_and_returns_the_hits(self, review, make_track):
        review.tracker = FakeTracker(bounce_tracks(make_track))
        review.court_solver = FakeSolver(Court.from_homography(np.eye(3)))

        result = review.run()

        assert len(result.hit_indices) == 1
        assert review.renderer.assigned  # track points were assigned
        assert review.renderer.rendered == []  # nothing was written

    def test_an_output_path_writes_the_review_video(self, review, make_track):
        review.tracker = FakeTracker(bounce_tracks(make_track))
        review.court_solver = FakeSolver(Court.from_homography(np.eye(3)))

        review.run(output_path="out.mp4")

        assert review.renderer.rendered[0][3] == "out.mp4"

    def test_the_hits_are_judged_by_the_end_of_a_run(self, review, make_track):
        review.tracker = FakeTracker(bounce_tracks(make_track))
        review.court_solver = FakeSolver(Court.from_homography(np.eye(3)))

        result = review.run()

        hit_track = review.tracks[result.hit_indices[0]]
        assert isinstance(hit_track.hit, Hit)
        assert hit_track.hit.is_in is not None


class TestExports:
    def test_the_stats_export_names_the_video_by_default(self, review, make_track):
        review.tracks = bounce_tracks(make_track)

        rows = review.export_shuttle_stats()

        assert {row["video"] for row in rows} == {"clip.mp4"}

    def test_an_explicit_video_name_wins(self, review, make_track):
        review.tracks = bounce_tracks(make_track)

        rows = review.export_shuttle_stats(video="labelled.mp4")

        assert rows[0]["video"] == "labelled.mp4"

    def test_the_analytics_plot_covers_the_tracks(self, review, make_track):
        from matplotlib import pyplot as plt

        review.tracks = bounce_tracks(make_track)

        fig, axs = review.plot_analytics()

        assert len(axs) == 5
        plt.close(fig)


def decoded_frames(count: int) -> list[DecodedFrame]:
    """A run of decoded frames, as VideoSource.sample would yield them."""
    image = np.zeros((4, 4, 3), np.uint8)
    return [
        DecodedFrame(number=n, image=image, model_input=image, timestamp=n / FPS)
        for n in range(count)
    ]


class FakeFrameCourt:
    """A per-frame Court that segments nothing, so no homography is solved."""

    solves = False

    def __init__(self, image=None, **kwargs):
        self.image = image
        self.homography_iou = 0.5

    def detect_court_lines(self):
        return [], []

    def get_contour(self):
        return None

    def solve_homography_from_lines(self, verticals, horizontals):
        return type(self).solves


class TestCourtSolver:
    """The solver itself, over frames that no model ever looks at."""

    def test_samples_about_one_frame_a_second(self, monkeypatch):
        source = FakeSource(frames=decoded_frames(90))
        monkeypatch.setattr("sokil.court.Court", FakeFrameCourt)

        # min_samples=1 switches the floor off — it has its own tests below
        CourtSolver({}, CourtSolverConfig(sample_seconds=1.0, min_samples=1)).solve(
            source
        )

        assert source.sampled == [round(FPS)]

    def test_a_shorter_sample_interval_takes_more_frames(self, monkeypatch):
        source = FakeSource(frames=decoded_frames(90))
        monkeypatch.setattr("sokil.court.Court", FakeFrameCourt)

        solver = CourtSolver({}, CourtSolverConfig(sample_seconds=0.5, min_samples=1))
        solver.solve(source)

        assert solver.sample_stride == round(FPS * 0.5)
        assert len(solver.sampled_frame_numbers) == len(source._frames) // 15

    def test_it_records_the_lines_it_saw_on_each_sampled_frame(self, monkeypatch):
        source = FakeSource(frames=decoded_frames(60))
        monkeypatch.setattr("sokil.court.Court", FakeFrameCourt)
        solver = CourtSolver({})

        solver.solve(source)

        assert len(solver.lines_per_frame) == len(solver.sampled_frame_numbers)

    def test_a_court_that_cannot_be_solved_from_the_lines_is_no_court(
        self, monkeypatch
    ):
        source = FakeSource(frames=decoded_frames(60))
        monkeypatch.setattr("sokil.court.Court", FakeFrameCourt)

        assert CourtSolver({}).solve(source) is None

    def test_the_frame_with_the_most_complete_mask_is_the_one_solved_on(
        self, monkeypatch
    ):
        # four sampled frames; the third has the least occluded court mask, so
        # its segmentation is what scores the candidate homographies
        source = FakeSource(frames=decoded_frames(120))
        monkeypatch.setattr("sokil.court.Court", FakeFrameCourt)
        monkeypatch.setattr(FakeFrameCourt, "solves", True)
        completeness = iter([1.0, 2.0, 9.0, 3.0])
        monkeypatch.setattr(
            "sokil.court.mask_completeness", lambda court: next(completeness)
        )
        solver = CourtSolver({}, CourtSolverConfig(min_samples=1))

        solved = solver.solve(source)

        assert solved is not None
        assert solver.solved_on_index == 2

    def test_a_clip_with_no_frames_has_no_court(self):
        assert CourtSolver({}).solve(FakeSource(frames=[])) is None

    def test_the_config_defaults_are_the_documented_ones(self):
        config = CourtSolverConfig()

        assert config.sample_seconds == 1.0
        assert config.min_frame_fraction == 0.4

    def test_the_solver_starts_with_nothing_recorded(self):
        solver = CourtSolver({})

        assert solver.sample_stride == 0
        assert solver.sampled_frame_numbers == []
        assert solver.solved_on_index is None


class TestSamplingFloor:
    """
    How many frames the solver looks at. The recommended camera records at
    120 fps and rallies are short, so a one-second spacing can ask for two
    frames — and consensus over two frames is not consensus.
    """

    def solve(self, frames, monkeypatch, **config):
        monkeypatch.setattr("sokil.court.Court", FakeFrameCourt)
        source = FakeSource(frames=decoded_frames(frames))
        CourtSolver({}, CourtSolverConfig(**config)).solve(source)
        return source

    def test_the_default_asks_for_at_least_a_dozen_frames(self):
        assert CourtSolverConfig().min_samples >= 12

    def test_a_short_clip_is_sampled_more_tightly_than_the_spacing_asks(
        self, monkeypatch
    ):
        # 60 frames at 30 fps is 2 s, which at one second apart is 2 frames
        source = self.solve(60, monkeypatch, min_samples=12)

        assert source.sampled == [60 // 12]

    def test_a_long_clip_keeps_its_time_based_spacing(self, monkeypatch):
        # 1800 frames at 30 fps already gives 60 samples, well over the floor
        source = self.solve(1800, monkeypatch, min_samples=12)

        assert source.sampled == [round(FPS)]

    def test_the_floor_can_be_switched_off(self, monkeypatch):
        source = self.solve(60, monkeypatch, min_samples=1)

        assert source.sampled == [round(FPS)]

    def test_a_clip_shorter_than_the_floor_samples_every_frame(self, monkeypatch):
        source = self.solve(5, monkeypatch, min_samples=12)

        assert source.sampled == [1]

    def test_an_unknown_frame_count_leaves_the_spacing_alone(self, monkeypatch):
        # containers can report a wrong or missing count, so it is only a hint
        monkeypatch.setattr("sokil.court.Court", FakeFrameCourt)
        source = FakeSource(frames=decoded_frames(60))
        source.frame_count = 0

        CourtSolver({}, CourtSolverConfig(min_samples=12)).solve(source)

        assert source.sampled == [round(FPS)]

    def test_the_floor_never_widens_the_spacing(self, monkeypatch):
        # a caller asking for dense sampling is not overridden by the floor
        source = self.solve(600, monkeypatch, sample_seconds=0.1, min_samples=12)

        assert source.sampled == [round(FPS * 0.1)]


class TestTrackerConfig:
    def test_the_defaults_describe_a_cpu_run(self):
        config = TrackerConfig()

        assert config.device == "cpu"
        assert 0 < config.confidence < 1
        assert config.frame_history > 0
