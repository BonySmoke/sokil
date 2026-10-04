"""
Shuttle tracking: the per-frame detection pass over a video.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from rich.progress import Progress

from .frame import ShuttleTrack
from .models import ShuttleDetector
from .shuttle import DirectionTracker, estimate_cork_position
from .util import laplacian_blur_metric, signed_angle_change
from .video import VideoSource

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """Knobs for the shuttle detection pass (see ShuttleTracker)."""

    device: str = "cpu"  # torch device the detector runs on
    confidence: float = 0.5  # minimum detection confidence
    kalman_process_noise: float = 0.03  # smoothing of the speed estimate
    ema_alpha: float = 0.8  # direction EMA: 1.0 = no smoothing, 0.5 = ~2 frames
    # the number of frames to keep in history.
    # useful for background subtraction, cork detection, etc.
    frame_history: int = 8


class ShuttleTracker:
    """
    Detects the shuttle and measures its motion.
    """

    def __init__(
        self,
        detector_kwargs: dict | None = None,
        config: TrackerConfig | None = None,
        detector: ShuttleDetector | None = None,
    ):
        """
        :param detector_kwargs: the weights to detect with, when no detector is
            given.
        :param detector: anything answering best_box(image, frame=...). The
            trained detector by default; a ground-truth source (see
            sokil.dataset.boxes.ShuttleBoxSource) puts annotated boxes through
            the same pass, so what the tracker derives can be measured without
            a model in it.
        """
        self.config = config or TrackerConfig()

        if detector is None:
            if detector_kwargs is None:
                raise ValueError("Give either detector_kwargs or a detector")
            detector = ShuttleDetector(
                detector_kwargs["model"],
                confidence=self.config.confidence,
                device=self.config.device,
                imgsz=detector_kwargs.get("imgsz", 640),
            )

        self.detector = detector

    def _new_kalman(self) -> cv2.KalmanFilter:
        # 4 dynamic params (x, y, dx, dy), 2 measured (x, y)
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kalman.processNoiseCov = (
            np.eye(4, dtype=np.float32) * self.config.kalman_process_noise
        )
        return kalman

    def track(self, source: VideoSource) -> list[ShuttleTrack]:
        """
        Run the detector over the whole video.

        :param source: the video to decode; its grayscale flag decides whether
            the detector sees the colour frame or the grayscale conversion.
        :returns: one ShuttleTrack per frame the shuttle was detected on,
            in decode order.
        """
        logger.info("Preprocessing %s", source.describe())

        kalman = self._new_kalman()
        direction_tracker = DirectionTracker(alpha=self.config.ema_alpha)

        tracks: list[ShuttleTrack] = []
        frame_count = 0
        execution_times = []

        progress = Progress()
        progress_task = progress.add_task(
            "Detecting shuttle", total=source.frame_count or None
        )
        progress.start()

        recent_frames = deque(maxlen=self.config.frame_history)

        try:
            for decoded in source:
                start_time = time.perf_counter()
                progress.advance(progress_task)
                frame_count += 1

                gray = cv2.cvtColor(decoded.image, cv2.COLOR_BGR2GRAY)
                previous_frames = list(recent_frames)
                recent_frames.append(gray)

                # The frame number travels with the image: a detector ignores
                # it, while a source that already knows where the shuttle is
                # needs it to look the frame up rather than predict on it.
                detection = self.detector.best_box(
                    decoded.model_input, frame=decoded.number
                )

                # the shuttle has not been found
                if detection is None:
                    continue

                (
                    (
                        top_left_x,
                        top_left_y,
                        bottom_right_x,
                        bottom_right_y,
                    ),
                    _,
                ) = detection

                center_x = (top_left_x + bottom_right_x) / 2
                center_y = (top_left_y + bottom_right_y) / 2

                angle_degree_change = 0.0
                previous_centers = [track.bbox_center for track in tracks[-2:]]
                if len(previous_centers) == 2:
                    angle_degree_change = signed_angle_change(
                        p1=previous_centers[0],
                        p2=previous_centers[1],
                        p3=(center_x, center_y),
                    )

                measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
                kalman.predict()
                kalman.correct(measurement)

                velocity_x, velocity_y = kalman.statePost[2, 0], kalman.statePost[3, 0]
                kalman_speed = float(np.sqrt(velocity_x**2 + velocity_y**2))

                ema_vx, ema_vy = direction_tracker.update(center_x, center_y)

                xyxy = (
                    int(top_left_x),
                    int(top_left_y),
                    int(bottom_right_x),
                    int(bottom_right_y),
                )

                measured_cork = estimate_cork_position(
                    gray,
                    previous_frames,
                    xyxy,
                    (ema_vx, ema_vy),
                )

                tracks.append(
                    ShuttleTrack(
                        number=decoded.number,
                        xyxy=xyxy,
                        speed=kalman_speed,
                        angle_change=angle_degree_change,
                        velocity=(ema_vx, ema_vy),
                        timestamp=decoded.timestamp,
                        blur=laplacian_blur_metric(decoded.image, xyxy),
                        measured_cork=measured_cork,
                    )
                )

                execution_times.append(time.perf_counter() - start_time)
        finally:
            progress.stop()

        logger.info(
            "Shuttle detection done: %d/%d frames have a shuttle "
            "(per-frame time avg %.3fs, min %.3fs, max %.3fs)",
            len(tracks),
            frame_count,
            float(np.mean(execution_times)) if execution_times else 0.0,
            min(execution_times, default=0.0),
            max(execution_times, default=0.0),
        )

        return tracks
