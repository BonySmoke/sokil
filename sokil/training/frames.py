"""
Pulling frames out of source videos for annotation.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from ..util import Undistorter

logger = logging.getLogger(__name__)


def load_undistorter(
    camera_intrinsics_path: str | None, camera_dist_path: str | None
) -> Undistorter | None:
    if not camera_intrinsics_path or not camera_dist_path:
        return None
    K = np.load(camera_intrinsics_path)
    dist = np.load(camera_dist_path)
    return Undistorter(K, dist)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int = 30,
    undistorter: Undistorter | None = None,
) -> list[Path]:
    """
    Extract every Nth frame to output_dir with deterministic names
    ({video_stem}_{frame:06d}.jpg) so CVAT exports map back to source frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    paths = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            if undistorter is not None:
                frame = undistorter(frame)
            path = output_dir / f"{video_path.stem}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(path)

        frame_idx += 1

    cap.release()
    logger.info("Extracted %d frames from %s", len(paths), video_path)
    return paths
