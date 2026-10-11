"""
Turning CVAT shuttle annotations into the ground-truth track.
"""

import logging
import re
from itertools import pairwise
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def frame_numbers(task) -> dict[int, int]:
    """
    CVAT's frame index -> the video frame number its filename carries.

    A name with no number in it is left out: it cannot be placed on the video's
    frame axis, so an annotation on it cannot be either.
    """
    numbers = {}
    for index, info in enumerate(task.get_frames_info()):
        match = re.search(r"(\d+)$", Path(info.name).stem)
        if match is None:
            logger.warning(
                "Frame %d of task %d has no number: %s", index, task.id, info.name
            )
            continue
        numbers[index] = int(match.group(1))
    return numbers


def _interpolate(shapes: list, index: int):
    """
    The box at a CVAT frame index, or None where the track is outside.

    CVAT holds a track before its first keyframe and moves linearly between
    them, so this reproduces the boxes the annotator saw rather than only the
    ones they placed.
    """
    ordered = sorted(shapes, key=lambda shape: shape.frame)

    if index < ordered[0].frame or index > ordered[-1].frame:
        return None

    # A keyframe speaks for its own frame. Where a track ends, CVAT marks the
    # last shape outside, and that shape still carries the box it had — so
    # interpolating up to it would resurrect the shuttle for one frame, on the
    # very frame the next flight's track begins.
    exact = {shape.frame: shape for shape in ordered}
    if index in exact:
        shape = exact[index]
        return None if shape.outside else tuple(shape.points[:4])

    for before, after in pairwise(ordered):
        if before.frame <= index <= after.frame:
            if before.outside:
                return None
            span = after.frame - before.frame
            ratio = 0.0 if span == 0 else (index - before.frame) / span
            start = np.array(before.points[:4], dtype=float)
            end = np.array(after.points[:4], dtype=float)
            return tuple(start + (end - start) * ratio)

    raise AssertionError("index inside the track but between no two keyframes")


def track_rows(task, clip_id: str, label_name: str = "Shuttlecock") -> list[dict]:
    """
    tracks rows for one clip, from the rectangle tracks of one CVAT task.

    A frame carrying two boxes is refused rather than resolved: the pipeline
    follows one shuttle, so ground truth with two does not say which one it
    should have followed.
    """
    from ..training.cvat import _enum_value

    numbers = frame_numbers(task)
    annotations = task.get_annotations()

    labels = {label.id: label.name for label in task.get_labels()}

    boxes: dict[int, tuple] = {}
    duplicates = []

    sources = [(track.shapes, track.label_id) for track in annotations.tracks]
    sources += [
        ([shape], shape.label_id)
        for shape in annotations.shapes
        if _enum_value(shape.type) == "rectangle"
    ]

    for shapes, label_id in sources:
        if labels.get(label_id) != label_name:
            continue
        if not shapes:
            continue

        first, last = min(s.frame for s in shapes), max(s.frame for s in shapes)
        for index in range(first, last + 1):
            if index not in numbers:
                continue

            box = _interpolate(shapes, index)
            if box is None:
                continue

            frame = numbers[index]
            if frame in boxes:
                duplicates.append(frame)
                continue
            boxes[frame] = box

    if duplicates:
        raise ValueError(
            f"Clip {clip_id!r} has more than one shuttle box on "
            f"{len(duplicates)} frame(s), e.g. {sorted(duplicates)[:5]}; the "
            f"pipeline follows one shuttle, so ground truth cannot hold two"
        )

    rows = [
        {
            "clip_id": clip_id,
            "frame": frame,
            "x1": round(float(box[0]), 2),
            "y1": round(float(box[1]), 2),
            "x2": round(float(box[2]), 2),
            "y2": round(float(box[3]), 2),
        }
        for frame, box in sorted(boxes.items())
    ]

    logger.info("%s: %d annotated frames", clip_id, len(rows))
    return rows


def import_tracks(
    task_ids: list[int] | None = None,
    project_name: str = "Badminton Shuttle Detection",
    clip_ids: dict[int, str] | None = None,
    known_clips: set[str] | None = None,
    completed_only: bool = True,
    session=None,
) -> list[dict]:
    """
    Read shuttle annotations from CVAT and return tracks rows.

    Only tasks whose jobs are all completed are read by default: a half
    annotated clip has gaps that are nobody's ground truth, and measuring
    against it would count the annotator's unfinished work as the algorithm's
    failure.
    """
    from .cvat_import import open_session, resolve_clip_id

    clip_ids = clip_ids or {}
    rows = []

    with open_session(session) as cvat:
        for task in cvat.tasks(project_name, task_ids):
            if completed_only and not cvat.completed(task):
                logger.info(
                    "Skipping task %d (%s): not all jobs completed",
                    task.id,
                    task.name,
                )
                continue

            clip_id = resolve_clip_id(task, known_clips, clip_ids.get(task.id))
            rows += track_rows(task, clip_id)

    return rows


def merge_tracks(existing, rows: list[dict]):
    """Fold imported tracks in by clip, leaving clips the run did not cover."""
    import pandas as pd

    from .schema import columns

    fresh = pd.DataFrame(rows, columns=columns("tracks"))
    if existing is None or existing.empty:
        return fresh

    kept = existing[~existing["clip_id"].isin(set(fresh["clip_id"]))]
    return pd.concat([kept, fresh], ignore_index=True)
