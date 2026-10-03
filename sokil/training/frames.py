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


def extract_court_frame(
    video_path: Path,
    output_dir: Path,
    undistorter: Undistorter | None = None,
    samples: int = 40,
    median: bool = True,
) -> Path:
    """
    Write the single frame a static clip's court is annotated on.

    By default it is the temporal median of frames sampled across the clip,
    which is an empty court: players move and the median keeps what did not, so
    no line intersection is hidden behind a leg. A court is annotated by
    clicking those intersections, so removing the people is most of the work.

    With median=False the first frame is used instead, which is quicker and
    enough when the clip happens to start empty.

    The name follows the same <clip>_<frame>.jpg convention as every other
    extracted frame, at frame 0: a static camera means one court, valid from
    the first frame on.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{video_path.stem}_{0:06d}.jpg"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not median or total <= 1:
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Could not read a frame from {video_path}")
            frames = [frame]
        else:
            # Spread the samples across the clip: players are only averaged
            # away by frames far enough apart to have moved between them.
            wanted = min(samples, total)
            indexes = np.linspace(0, total - 1, wanted).astype(int)
            frames = []
            for index in indexes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)

            if not frames:
                raise ValueError(f"Could not read any frames from {video_path}")
    finally:
        cap.release()

    image = np.median(np.stack(frames), axis=0).astype(np.uint8)
    if undistorter is not None:
        image = undistorter(image)

    cv2.imwrite(str(path), image)
    logger.info(
        "Wrote the %s of %d frame(s) from %s -> %s",
        "median" if len(frames) > 1 else "first frame",
        len(frames),
        video_path.name,
        path,
    )
    return path


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
