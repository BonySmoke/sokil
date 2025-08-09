"""
Video decoding for the review pipeline.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from .util import Undistorter

logger = logging.getLogger(__name__)


@dataclass
class DecodedFrame:
    """
    One frame of the video.

    :param number: frame number in decode order — the id every stage keys on.
    :param image: the frame as it should be shown (undistorted, colour).
    :param model_input: the frame as the detectors should see it.
    :param timestamp: seconds into the video.
    """

    number: int
    image: cv2.typing.MatLike
    model_input: cv2.typing.MatLike
    timestamp: float


class VideoSource:
    """
    A re-readable video: each iteration opens the file and decodes from the top.

    The same instance can be iterated by several stages in turn — it holds no
    frames, only the path and the per-frame transforms (undistortion, grayscale)
    that every stage must see identically.
    """

    def __init__(
        self,
        video_path: str | Path,
        undistorter: Undistorter | None = None,
        grayscale: bool = False,
    ):
        self.video_path = Path(video_path)
        self.undistorter = undistorter
        self.grayscale = grayscale

        capture = self._open()
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        # containers can report a wrong or missing count; treat it as a hint
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        return capture

    def _decode(self, image, number: int, timestamp: float) -> DecodedFrame:
        if self.undistorter is not None:
            image = self.undistorter(image)

        model_input = image
        if self.grayscale:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            model_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return DecodedFrame(
            number=number, image=image, model_input=model_input, timestamp=timestamp
        )

    def __iter__(self):
        """Yield every frame, in decode order."""
        yield from self.sample(1)

    def sample(self, stride: int):
        """
        Yield every `stride`-th frame (0, stride, 2*stride, ...).

        Skipped frames are grabbed but not decoded, which is much cheaper than
        decoding them and exact, unlike seeking by frame index (which lands on
        the nearest keyframe for many containers).
        """
        capture = self._open()
        number = 0

        try:
            while True:
                if number % stride:
                    if not capture.grab():
                        break
                    number += 1
                    continue

                success, image = capture.read()
                if not success:
                    break

                timestamp = round(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)
                yield self._decode(image, number, timestamp)
                number += 1
        finally:
            capture.release()

    def describe(self) -> str:
        return (
            f"{self.video_path.name}: {self.width}x{self.height} @ "
            f"{self.fps:.1f} fps, ~{self.frame_count} frames"
        )
