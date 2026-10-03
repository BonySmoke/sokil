"""
TODO: turn this into a single CVAT client outside of this module.
Turn CVAT court-keypoint annotations into court_points.csv
"""

import logging
from itertools import pairwise

import numpy as np

from .court import resolve_vertex
from .schema import PointSource

logger = logging.getLogger(__name__)

# How far the court must move for a keyframe to be a new court rather than the
# previous one re-confirmed. A click that does not drag leaves a keyframe at
# exactly the old position; a hand nudging a point leaves one a pixel away.
KEYFRAME_TOLERANCE_PX = 2.0

# How far a point must sit from its pre-annotation to count as adjusted, when a
# pre-annotation is available to compare against.
ADJUSTED_TOLERANCE_PX = 1.0


def _interpolate(shapes: list, frame: int) -> tuple[tuple[float, float], bool]:
    """
    Where one skeleton element is at a frame, and whether it is out of view.

    CVAT holds a track before its first keyframe and after its last, and moves
    linearly in between, so this reproduces what the annotator saw rather than
    only reading back the keyframes themselves.
    """
    ordered = sorted(shapes, key=lambda shape: shape.frame)

    if frame <= ordered[0].frame:
        first = ordered[0]
        return tuple(first.points[:2]), bool(first.outside)

    if frame >= ordered[-1].frame:
        last = ordered[-1]
        return tuple(last.points[:2]), bool(last.outside)

    for before, after in pairwise(ordered):
        if before.frame <= frame <= after.frame:
            span = after.frame - before.frame
            ratio = 0.0 if span == 0 else (frame - before.frame) / span
            start = np.array(before.points[:2], dtype=float)
            end = np.array(after.points[:2], dtype=float)
            point = start + (end - start) * ratio
            # An element goes out of view at the keyframe that says so, not
            # gradually, so the interval takes the state it started in.
            return tuple(point), bool(before.outside)

    raise AssertionError("frame outside the track it was taken from")


def _vertex_of(element, sublabel_names: dict[int, str]) -> int | None:
    name = sublabel_names.get(element.label_id)
    if name is None:
        logger.warning("Skipping element with unknown sublabel id %s", element.label_id)
        return None
    return resolve_vertex(name)


def _keyframe_positions(track, sublabel_names: dict[int, str]) -> dict:
    """
    Every element's position at every frame any element has a keyframe on,
    keyed by frame then by court vertex.
    """
    frames = sorted({shape.frame for el in track.elements for shape in el.shapes})
    positions = {}

    for frame in frames:
        at_frame = {}
        for element in track.elements:
            if not element.shapes:
                continue
            vertex = _vertex_of(element, sublabel_names)
            if vertex is not None:
                at_frame[vertex] = _interpolate(element.shapes, frame)
        positions[frame] = at_frame

    return positions


def _shape_positions(shapes: list, sublabel_names: dict[int, str]) -> dict:
    """
    The same mapping, for a task annotated as plain shapes.

    An image task has no interpolation to do: each skeleton stands on its own
    frame, which is how a one-frame court task comes back.
    """
    positions = {}

    for shape in shapes:
        at_frame = positions.setdefault(shape.frame, {})
        for element in getattr(shape, "elements", None) or []:
            vertex = _vertex_of(element, sublabel_names)
            if vertex is not None:
                at_frame[vertex] = (
                    tuple(element.points[:2]),
                    bool(element.outside),
                )

    return positions


def _moved(previous: dict, current: dict, tolerance: float) -> bool:
    """Whether the court at one keyframe differs from the court at the last."""
    for vertex, (point, outside) in current.items():
        if vertex not in previous:
            return True

        previous_point, previous_outside = previous[vertex]
        if outside != previous_outside:
            return True
        if outside:
            continue
        if np.linalg.norm(np.array(point) - np.array(previous_point)) > tolerance:
            return True

    return False


def _point_source(
    vertex: int, point: tuple[float, float], preannotation: dict | None
) -> str:
    """
    Whether a point was placed by hand, taken from the solver, or moved off it.

    Derived by comparing against the pre-annotation that was uploaded, so
    automation bias is measurable without the annotator recording anything.
    """
    if not preannotation or vertex not in preannotation:
        return PointSource.MANUAL.value

    offset = np.linalg.norm(np.array(point) - np.array(preannotation[vertex]))
    if offset > ADJUSTED_TOLERANCE_PX:
        return PointSource.SOLVER_ADJUSTED.value
    return PointSource.SOLVER.value


def court_points_from_track(
    track,
    sublabel_names: dict[int, str],
    clip_id: str,
    preannotation: dict[int, tuple[float, float]] | None = None,
    tolerance: float = KEYFRAME_TOLERANCE_PX,
) -> list[dict]:
    """
    court_points rows for one annotated clip.

    :param track: the skeleton track, whose elements are the court vertices.
    :param sublabel_names: CVAT sublabel id -> its name.
    :param clip_id: the clip these points describe.
    :param preannotation: vertex -> the position the solver proposed, if the
        task was pre-annotated; used only to fill in the source column.
    :param tolerance: how far the court must move to count as a new keyframe.
    """
    return _rows_from_positions(
        _keyframe_positions(track, sublabel_names), clip_id, preannotation, tolerance
    )


def court_points_from_shapes(
    shapes: list,
    sublabel_names: dict[int, str],
    clip_id: str,
    preannotation: dict[int, tuple[float, float]] | None = None,
    tolerance: float = KEYFRAME_TOLERANCE_PX,
) -> list[dict]:
    """court_points rows for a clip annotated as shapes rather than a track."""
    return _rows_from_positions(
        _shape_positions(shapes, sublabel_names), clip_id, preannotation, tolerance
    )


def _rows_from_positions(
    positions: dict,
    clip_id: str,
    preannotation: dict[int, tuple[float, float]] | None,
    tolerance: float,
) -> list[dict]:
    rows = []
    kept = None

    for frame in sorted(positions):
        at_frame = positions[frame]

        if kept is not None and not _moved(kept, at_frame, tolerance):
            logger.debug(
                "%s: keyframe %d repeats the previous court, dropping it",
                clip_id,
                frame,
            )
            continue

        for vertex in sorted(at_frame):
            point, outside = at_frame[vertex]
            rows.append(
                {
                    "clip_id": clip_id,
                    "frame": frame,
                    "vertex": vertex,
                    "x_px": None if outside else round(float(point[0]), 2),
                    "y_px": None if outside else round(float(point[1]), 2),
                    "visible": not outside,
                    "source": _point_source(vertex, point, preannotation),
                }
            )

        kept = at_frame

    logger.info(
        "%s: %d keyframe(s) from %d annotated, %d rows",
        clip_id,
        len({row["frame"] for row in rows}),
        len(positions),
        len(rows),
    )
    return rows


def read_subsets(
    project_name: str,
    clip_ids: dict[int, str] | None = None,
    known_clips: set[str] | None = None,
) -> dict[str, str]:
    """
    Each clip's CVAT subset, from the tasks of one project.

    :returns: clip_id -> the subset string CVAT holds, which may be empty.
    """
    from ..training.cvat import find_project, make_cvat_client

    clip_ids = clip_ids or {}
    subsets = {}

    with make_cvat_client() as client:
        project = find_project(client, project_name)

        for task in client.tasks.list(project_id=project.id):
            clip_id = resolve_clip_id(task, known_clips, clip_ids.get(task.id))
            subset = getattr(task, "subset", "") or ""

            if clip_id in subsets and subsets[clip_id] != subset:
                raise ValueError(
                    f"Clip {clip_id!r} has more than one task in "
                    f"{project_name!r} and they disagree: "
                    f"{subsets[clip_id]!r} and {subset!r}"
                )

            subsets[clip_id] = subset

    return subsets


def import_court_points(
    task_ids: list[int] | None = None,
    project_name: str = "Badminton Court Keypoints",
    clip_ids: dict[int, str] | None = None,
    tolerance: float = KEYFRAME_TOLERANCE_PX,
    known_clips: set[str] | None = None,
) -> list[dict]:
    """
    Read court keypoint tasks from CVAT and return court_points rows.

    Without explicit task_ids, every task in the project is read. A task with no
    skeleton track is skipped with a warning rather than failing the run: a task
    that has not been annotated yet is a normal thing to find.
    """
    from ..training.cvat import find_project, make_cvat_client

    clip_ids = clip_ids or {}
    rows = []

    with make_cvat_client() as client:
        project = find_project(client, project_name)

        labels = project.get_labels()
        skeletons = [label for label in labels if getattr(label, "sublabels", None)]
        if not skeletons:
            raise ValueError(
                f"CVAT project {project_name!r} has no skeleton label to read "
                f"court vertices from"
            )

        sublabel_names = {
            sublabel.id: sublabel.name
            for label in skeletons
            for sublabel in label.sublabels
        }

        if task_ids is None:
            task_ids = sorted(
                task.id for task in client.tasks.list(project_id=project.id)
            )

        for task_id in task_ids:
            task = client.tasks.retrieve(task_id)
            annotations = task.get_annotations()
            tracks = annotations.tracks
            shapes = [
                shape
                for shape in annotations.shapes
                if getattr(shape, "elements", None)
            ]

            if not tracks and not shapes:
                logger.warning(
                    "Task %d (%s) has no skeleton annotation; skipping",
                    task_id,
                    task.name,
                )
                continue

            clip_id = resolve_clip_id(task, known_clips, clip_ids.get(task_id))

            if tracks:
                if len(tracks) > 1:
                    logger.warning(
                        "Task %d (%s) has %d tracks; reading the first",
                        task_id,
                        task.name,
                        len(tracks),
                    )
                rows += court_points_from_track(
                    tracks[0], sublabel_names, clip_id, tolerance=tolerance
                )
            else:
                rows += court_points_from_shapes(
                    shapes, sublabel_names, clip_id, tolerance=tolerance
                )

    return rows


def default_clip_id(name: str, strip_frame: bool = False) -> str:
    """
    The clip id a CVAT task's filename suggests.

    A suggestion only: the recording may have been renamed since it was
    uploaded, and the dataset's clip id is whatever clips.csv says. Validation
    catches the mismatch either way, by reporting a clip nothing knows about.

    :param strip_frame: drop the trailing _<frame> an extracted image carries.
        A video task's name is the recording itself, which may legitimately end
        in digits, so its whole stem is kept.
    """
    import re
    from pathlib import Path

    stem = Path(name).stem.lower()
    if strip_frame:
        stem = re.sub(r"_\d+$", "", stem) or stem

    return "".join(c if c.isalnum() or c in "._-" else "-" for c in stem).strip("-")


def merge_court_points(existing, rows: list[dict]):
    """
    Fold freshly imported court points into the table, by clip.

    A run that imports one task is the last word on that task's clip and says
    nothing about any other, so replacing the whole table would quietly drop
    every clip the run did not cover.
    """
    import pandas as pd

    from .schema import columns

    fresh = pd.DataFrame(rows, columns=columns("court_points"))
    if existing is None or existing.empty:
        return fresh

    kept = existing[~existing["clip_id"].isin(set(fresh["clip_id"]))]
    return pd.concat([kept, fresh], ignore_index=True)


# What create_cvat_task prefixes a task name with, per model.
TASK_NAME_PREFIXES = ("shuttle-", "court_keypoints-", "court-")


def clip_id_candidates(task) -> list[str]:
    """
    Every clip id a CVAT task plausibly refers to, best guess first.

    There is no one reliable source. A video task's frames carry the recording's
    name. An image task's frames carry <clip>_<frame>, except where they were
    uploaded in a folder, where the folder carries it — and where the recording
    was renamed after upload, the frames carry a name nobody uses any more and
    only the task name is current. So every reading is offered and the manifest
    picks (see resolve_clip_id).
    """
    from pathlib import Path

    from ..training.cvat import _enum_value

    frames = task.get_frames_info()
    name = frames[0].name if frames else ""
    candidates = []

    parent = Path(name).parent
    if parent != Path("."):
        candidates.append(default_clip_id(parent.parts[0]))

    if name:
        from_video = _enum_value(task.mode) == "interpolation"
        candidates.append(default_clip_id(name, strip_frame=not from_video))

    task_name = task.name
    for prefix in TASK_NAME_PREFIXES:
        if task_name.startswith(prefix):
            task_name = task_name[len(prefix) :]
            break
    candidates.append(default_clip_id(task_name))

    # Keep the order, drop repeats and anything empty or meaningless.
    seen, ordered = set(), []
    for candidate in candidates:
        if candidate and candidate not in seen and candidate != "frame":
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def resolve_clip_id(
    task, known: set[str] | None = None, override: str | None = None
) -> str:
    """
    Which clip a task annotates, decided by the manifest where there is one.

    clips.csv is the dataset's register of clips, so when a task's frames and
    its name disagree — which happens as soon as a recording is renamed after
    being uploaded — the manifest is what settles it rather than a rule about
    which field to trust.
    """
    if override:
        return override

    candidates = clip_id_candidates(task)
    if not candidates:
        raise ValueError(f"Task {task.id} ({task.name!r}) names no clip")

    if known:
        for candidate in candidates:
            if candidate in known:
                return candidate
        logger.warning(
            "Task %d (%s) refers to %s, none of which is in clips.csv; taking %r",
            task.id,
            task.name,
            candidates,
            candidates[0],
        )

    return candidates[0]
