"""
The code for the instant review system.
"""

import logging
from pathlib import Path

import numpy as np

from .court import Court, CourtSolver, CourtSolverConfig
from .explanation import Explainer
from .frame import ShuttleTrack
from .hits import HitDetectionConfig, HitDetector, HitJudge, HitResult, detect_hits
from .render import RenderConfig, ReviewRenderer
from .report import Analytics, StatsExporter
from .tracker import ShuttleTracker, TrackerConfig
from .util import Undistorter
from .video import VideoSource

logger = logging.getLogger(__name__)

__all__ = [
    "HitDetectionConfig",
    "HitResult",
    "Review",
    "detect_hits",
]


class Review:
    """
    Runs the review pipeline over one video and holds its results.

    The step methods can be called one at a time (preprocess ->
    set_frame_hit_candidates -> set_hit_positions -> set_track_points ->
    video_to_file), which is what a grid search or a notebook wants, or all at
    once through run().
    """

    def __init__(
        self,
        video_path: str,
        court_detector_kwargs: dict,
        shuttle_detector_kwargs: dict,
        grayscale: bool = True,
        hit_config: HitDetectionConfig | None = None,
        tracker_config: TrackerConfig | None = None,
        court_solver_config: CourtSolverConfig | None = None,
        render_config: RenderConfig | None = None,
        camera_intrinsics_path: Path | None = None,
        camera_dist_path: Path | None = None,
        court_homography: "np.ndarray | None" = None,
    ):
        self.video_path = Path(video_path)
        self.grayscale = grayscale
        self.camera_intrinsics_path = camera_intrinsics_path
        self.camera_dist_path = camera_dist_path
        # when given, the court is taken as-is and never solved from this video
        self.court_homography = court_homography

        self.hit_config = hit_config or HitDetectionConfig()

        self.tracker = ShuttleTracker(shuttle_detector_kwargs, tracker_config)
        self.court_solver = CourtSolver(court_detector_kwargs, court_solver_config)
        self.hit_detector = HitDetector(self.hit_config)
        self.hit_judge = HitJudge()
        self.renderer = ReviewRenderer(render_config)
        self.stats_exporter = StatsExporter()
        self.analytics = Analytics()

        # results, filled in by the steps below
        self.tracks: list[ShuttleTrack] = []
        # the session court: the camera is static, so one reliably solved
        # homography is shared by every frame
        self.court: Court | None = None
        # a list of smooth trajectories, as indices into self.tracks
        self.trajectories: list[list[int]] = []
        self.hit_result = HitResult()

        self._source: VideoSource | None = None

    @property
    def source(self) -> VideoSource:
        """The video, opened lazily and re-read by every pass that needs it."""
        if self._source is None:
            self._source = VideoSource(
                self.video_path,
                undistorter=self._get_undistorter(),
                grayscale=self.grayscale,
            )
        return self._source

    @property
    def fps(self):
        return self.source.fps

    @property
    def frame_width(self):
        return self.source.width

    @property
    def frame_height(self):
        return self.source.height

    def _get_undistorter(self) -> Undistorter | None:
        """Build the undistorter from the calibration files, if both are set."""
        if not (self.camera_intrinsics_path and self.camera_dist_path):
            logger.info("Undistortion: off")
            return None

        K = np.load(str(self.camera_intrinsics_path))
        dist = np.load(str(self.camera_dist_path))
        logger.info("Undistortion: on")
        return Undistorter(K, dist)

    def preprocess(self):
        """Detect the shuttle on every frame, then establish the session court."""
        self.tracks = self.tracker.track(self.source)

        if self.court_homography is not None:
            logger.info("Using the supplied court homography (not solving)")
            self.court = Court.from_homography(
                self.court_homography,
                court_config=self.court_solver.court_detector_kwargs.get(
                    "court_config"
                ),
            )
            return

        self.court = self.court_solver.solve(self.source)

    def set_frame_hit_candidates(self, config: HitDetectionConfig | None = None):
        """
        Detect hits and rebuild the drawable trajectories.

        :param config: overrides self.hit_config for this call (lets a grid
            search sweep params without rebuilding the Review).
        :returns: indices of the accepted hit tracks.
        """
        self.hit_result = self.hit_detector.detect(
            self.tracks, self.court, self.fps, config, frame_width=self.frame_width
        )
        self.trajectories = self.hit_result.trajectories
        return self.hit_result.hit_indices

    def set_hit_positions(self):
        """For every hit, estimate whether it was IN or OUT of the court."""
        return self.hit_judge.judge(self.tracks, self.court)

    def set_track_points(self):
        """Based on the trajectories, set the drawable track points."""
        self.renderer.assign_track_points(self.tracks, self.trajectories)

    def video_to_file(self, output_path: str):
        """Store the review video to the file."""
        self.renderer.render(
            self.source, self.tracks, self.court, output_path, self._explainer()
        )

    def _explainer(self) -> "Explainer | None":
        if not self.renderer.config.explain:
            return None

        return Explainer(
            court_detector_kwargs=self.court_solver.court_detector_kwargs,
            config=self.renderer.config,
            consensus_verticals=self.court_solver.consensus_verticals,
            consensus_horizontals=self.court_solver.consensus_horizontals,
        )

    def export_shuttle_stats(
        self,
        output_path: str | None = None,
        video: str | None = None,
        append: bool = False,
    ):
        """Export per-frame shuttle stats (see StatsExporter)."""
        return self.stats_exporter.export(
            self.tracks,
            video=video or self.video_path.name,
            output_path=output_path,
            append=append,
        )

    def plot_analytics(self):
        return self.analytics.plot(self.tracks)

    def run(self, output_path: str | None = None) -> HitResult:
        """
        The whole pipeline: detect, judge and (optionally) render.

        :param output_path: when given, the review video is written there.
        """
        self.preprocess()
        self.set_frame_hit_candidates()
        self.set_hit_positions()
        self.set_track_points()

        if output_path:
            self.video_to_file(output_path)

        return self.hit_result
