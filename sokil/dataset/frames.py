"""
Random access to a clip's frames, by decode order, without writing any out.

The labeller needs to jump around a clip; the dataset needs frame numbers that
mean the same thing here, in CVAT and at inference. Both are satisfied by
counting decoded frames, which is what this does and what the pipeline does.

Seeking is deliberately not used. CAP_PROP_POS_FRAMES is a request to land on a
frame, and on variable-rate or concatenated captures it lands near one instead,
so a frame number would stop meaning the same thing in two places. Going
backwards past the cache therefore reopens the clip and decodes forward again,
which costs a second or two on a rally-length clip and cannot be wrong.
"""

import logging
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Frames are cached exactly as decoded, so a frame read twice is the same frame
# both times and the same frame the pipeline sees. That costs memory — a 640x480
# frame is about 0.9 MB — so the cache is bounded by a budget rather than a frame
# count, and holds fewer frames of larger footage.
DEFAULT_CACHE_MB = 256


class VideoFrames:
    """
    A clip's frames, indexed by decode order.

    len() is what the container claims, which is close enough for a scrollbar;
    the true count is only known once the clip has been read to the end, and
    frames past the real end read as None.
    """

    def __init__(self, path: Path, cache_mb: float = DEFAULT_CACHE_MB):
        self.path = Path(path)

        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._capture = None
        self._next = 0

        probe = self._open()
        self._claimed = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(probe.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._end = None

        frame_bytes = max(self.width * self.height * 3, 1)
        self.cache_size = max(int(cache_mb * 1024 * 1024 / frame_bytes), 2)
        logger.debug("Caching up to %d frames of %s", self.cache_size, self.path.name)

    def _open(self):
        self.release()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {self.path}")
        self._capture = capture
        self._next = 0
        return capture

    def __len__(self) -> int:
        return self._end if self._end is not None else max(self._claimed, 0)

    def _remember(self, index: int, frame: np.ndarray) -> None:
        self._cache[index] = frame
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def __getitem__(self, index: int) -> np.ndarray | None:
        """The frame at that decode position, or None past the end of the clip."""
        if index < 0:
            raise IndexError(f"Frame {index} is before the start of the clip")

        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached

        if index < self._next:
            # Already decoded past it and it has fallen out of the cache. The
            # only way back that keeps frame numbers honest is from the top.
            logger.debug("Rewinding %s to reach frame %d", self.path.name, index)
            self._open()

        frame = None
        while self._next <= index:
            ok, decoded = self._capture.read()
            if not ok:
                self._end = self._next
                return None
            self._remember(self._next, decoded)
            self._next += 1
            frame = decoded

        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
