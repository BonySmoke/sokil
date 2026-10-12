"""
Reading, writing and validating the annotation dataset.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import schema
from .schema import Call
from .splits import SUBSET_SPLITS

logger = logging.getLogger(__name__)

# The split names, taken from the mapping that defines them rather than
# written out again: a second list is a second thing to keep in step.
SPLITS = tuple(dict.fromkeys(SUBSET_SPLITS.values()))


class DatasetError(RuntimeError):
    """The dataset on disk is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Dataset:
    """
    An annotation dataset rooted at a directory.

    One <root>/<table>.csv per table in schema.SCHEMAS, and
    <root>/splits/<split>.txt holding one clip_id per line. The tables are not
    listed here on purpose: a list written out by hand goes stale the first
    time one is added, which it already had.
    """

    root: Path

    def path(self, table: str) -> Path:
        return self.root / f"{table}.csv"

    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    def read(self, table: str) -> pd.DataFrame:
        """Read one table, with its declared columns in their declared order."""
        path = self.path(table)
        if not path.exists():
            raise DatasetError(f"No {table} table at {path}")

        frame = pd.read_csv(path, dtype=schema.read_dtypes(table))
        missing = [c for c in schema.columns(table) if c not in frame.columns]
        if missing:
            raise DatasetError(f"{path} is missing column(s) {missing}")

        if frame.empty:
            # A header with no rows under it says nothing about its types, and
            # a table nobody has filled in yet is a normal state to be in —
            # not something to report as a dozen dtype failures.
            frame = frame.astype(schema.dtypes(table))

        return frame

    def write(self, table: str, frame: pd.DataFrame) -> Path:
        """Write one table deterministically; returns the path written."""
        frame = frame.copy()

        for column, decimals in schema.ROUNDING.get(table, {}).items():
            # to_numeric first: a column carrying unset cells arrives as
            # objects, and None has no __round__.
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round(
                decimals
            )

        frame = frame[schema.columns(table)]
        frame = frame.sort_values(schema.SORT_KEYS[table], kind="stable")

        path = self.path(table)
        path.parent.mkdir(parents=True, exist_ok=True)
        # na_rep is left at "" so an unset cell is empty rather than the string
        # "nan", which reads back as a value rather than as an absence.
        frame.to_csv(path, index=False, na_rep="", lineterminator="\n")
        logger.info("Wrote %d %s rows -> %s", len(frame), table, path)
        return path

    def read_splits(self) -> dict[str, list[str]]:
        """
        The clip ids in each split file, skipping blank and commented lines.

        A missing split file is an empty split, not an error: a dataset may
        legitimately have no test partition yet.
        """
        splits = {}
        for name in SPLITS:
            path = self.splits_dir / f"{name}.txt"
            if not path.exists():
                splits[name] = []
                continue
            splits[name] = [
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
        return splits

    def write_splits(self, splits: dict[str, list[str]]) -> None:
        """Write one clip id per line, sorted, one file per split."""
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        for name, clip_ids in splits.items():
            path = self.splits_dir / f"{name}.txt"
            path.write_text("".join(f"{clip_id}\n" for clip_id in sorted(clip_ids)))
            logger.info("Wrote %d clips -> %s", len(clip_ids), path)

    def validate(self) -> list[str]:
        """
        Every problem found, as human-readable lines. Empty means the dataset is
        consistent.

        Problems are collected rather than raised one at a time: fixing
        annotations is a batch job, and being told about one bad row per run
        turns an afternoon into a week.
        """
        clips = self.read("clips")
        events = self.read("events")

        problems = _schema_problems("clips", clips)
        problems += _source_problems(self.root, clips)
        problems += _schema_problems("events", events)
        problems += _cross_table_problems(clips, events)
        problems += _split_problems(clips, self.read_splits())

        # Court keypoints and tracks are optional: a dataset is useful before
        # they exist, and their absence is a stage not reached rather than a
        # fault.
        if self.path("court_points").exists():
            court_points = self.read("court_points")
            problems += _schema_problems("court_points", court_points)
            problems += _court_point_problems(clips, court_points)
            problems += _stale_derivation_problems(clips, events, court_points)

        if self.path("tracks").exists():
            tracks = self.read("tracks")
            problems += _schema_problems("tracks", tracks)
            problems += _track_problems(clips, events, tracks)

        return problems

    def validate_or_raise(self) -> None:
        problems = self.validate()
        if problems:
            raise DatasetError(
                f"{len(problems)} problem(s) in the dataset at {self.root}:\n  - "
                + "\n  - ".join(problems)
            )


def _schema_problems(table: str, frame: pd.DataFrame) -> list[str]:
    """Check one table against its schema, collecting every failure at once."""
    from pandera.errors import SchemaErrors

    try:
        schema.SCHEMAS[table].validate(frame, lazy=True)
    except SchemaErrors as errors:
        return [
            _schema_problem(table, failure)
            for failure in errors.failure_cases.itertuples()
        ]
    return []


def _schema_problem(table: str, failure) -> str:
    """
    One schema failure, as a line somebody can act on.

    The failing value is quoted only when there is one. An empty cell is the
    commonest failure of all — a column nobody has filled in yet — and asking
    pandas whether a missing value is truthy raises rather than answering.
    """
    where = failure.column or "the table"
    value = failure.failure_case

    if pd.isna(value):
        return f"{table}.csv: {where} is empty, and {failure.check} needs a value"

    return f"{table}.csv: {failure.check} failed for {where} (value {value!r})"


def _source_problems(root: Path, clips: pd.DataFrame) -> list[str]:
    """
    Recordings a clip points at that are not where it says.

    source_uri is what a reader follows to find the footage, so a path that
    resolves nowhere is worse than an empty cell: the empty cell says the video
    is not here, and the dangling one says it is. A URL is taken on trust —
    checking it would mean reaching out over the network to validate a table.
    """
    problems = []

    for clip in clips.itertuples():
        uri = clip.source_uri
        if pd.isna(uri) or str(uri).startswith(("http://", "https://")):
            continue

        if not (root / str(uri)).is_file():
            problems.append(
                f"clips.csv: clip {clip.clip_id!r} points at {uri!r}, which is "
                f"not a file under {root}"
            )

    return problems


def _cross_table_problems(clips: pd.DataFrame, events: pd.DataFrame) -> list[str]:
    """The checks that no single table can make about itself."""
    from ..court import CourtCameraMode, CourtModelType

    problems = []

    camera_modes = {mode.value for mode in CourtCameraMode}
    court_types = {kind.value for kind in CourtModelType}

    for clip in clips.itertuples():
        if clip.camera_mode not in camera_modes:
            problems.append(
                f"clips.csv: clip {clip.clip_id!r} has camera_mode "
                f"{clip.camera_mode!r}, expected one of {sorted(camera_modes)}"
            )
        if clip.court_type not in court_types:
            problems.append(
                f"clips.csv: clip {clip.clip_id!r} has court_type "
                f"{clip.court_type!r}, expected one of {sorted(court_types)}"
            )

    frame_counts = dict(zip(clips["clip_id"], clips["frame_count"]))

    for event in events.itertuples():
        if event.clip_id not in frame_counts:
            problems.append(
                f"events.csv: event {schema.event_label(event)} references clip "
                f"{event.clip_id!r}, which is not in clips.csv"
            )
        elif event.frame >= frame_counts[event.clip_id]:
            problems.append(
                f"events.csv: event {schema.event_label(event)} is at frame {event.frame}, "
                f"past the {frame_counts[event.clip_id]} frames of clip "
                f"{event.clip_id!r}"
            )

        # A verdict is made against a boundary, so one without the other is not
        # a call anybody can check or reproduce.
        if event.call in (Call.IN.value, Call.OUT.value) and pd.isna(event.court_type):
            problems.append(
                f"events.csv: event {schema.event_label(event)} is called "
                f"{event.call!r} without a court_type to call it against"
            )

        # Court metres are derived from the clicked pixel, so metres without
        # the pixel they came from cannot be recomputed or corrected.
        if not pd.isna(event.x_m) and pd.isna(event.x_px):
            problems.append(
                f"events.csv: event {schema.event_label(event)} has court coordinates "
                f"but no clicked pixel to derive them from"
            )

    return problems


def _court_point_problems(clips: pd.DataFrame, court_points: pd.DataFrame) -> list[str]:
    """
    Whether the annotated court keypoints describe a court that could exist.

    The reprojection check is the one worth having. Everything else catches
    typing; that one catches a confident click on the wrong intersection, which
    is the mistake a human reviewer will not see and the evaluation will
    inherit.
    """
    import numpy as np

    from .court import (
        MAX_RESIDUAL_PX,
        MIN_POINTS_FOR_HOMOGRAPHY,
        reprojection_residuals,
        vertex_count,
    )

    problems = []
    vertices = vertex_count()
    frame_counts = dict(zip(clips["clip_id"], clips["frame_count"]))
    static = dict(zip(clips["clip_id"], clips["camera_static"]))

    for clip_id, clip_points in court_points.groupby("clip_id", sort=True):
        if clip_id not in frame_counts:
            problems.append(f"court_points.csv: clip {clip_id!r} is not in clips.csv")
            continue

        keyframes = sorted(clip_points["frame"].unique())

        # One keyframe is a complete description only if the view never moved.
        if static.get(clip_id) and len(keyframes) > 1:
            problems.append(
                f"court_points.csv: clip {clip_id!r} is marked camera_static "
                f"but has {len(keyframes)} keyframes at {keyframes}"
            )
        if not static.get(clip_id, True) and len(keyframes) == 1:
            problems.append(
                f"court_points.csv: clip {clip_id!r} is not camera_static, so "
                f"one keyframe at frame {keyframes[0]} cannot describe it"
            )

        for frame, frame_points in clip_points.groupby("frame", sort=True):
            where = f"clip {clip_id!r} frame {frame}"

            if frame >= frame_counts[clip_id]:
                problems.append(
                    f"court_points.csv: {where} is past the "
                    f"{frame_counts[clip_id]} frames of the clip"
                )

            unknown = sorted(
                v for v in frame_points["vertex"] if not 1 <= v <= vertices
            )
            if unknown:
                problems.append(
                    f"court_points.csv: {where} names vertex/vertices {unknown}, "
                    f"but the court model has {vertices}"
                )
                continue

            visible = frame_points[frame_points["visible"]]

            placed = visible[["x_px", "y_px"]].notna().all(axis=1)
            if not placed.all():
                missing = sorted(visible.loc[~placed, "vertex"])
                problems.append(
                    f"court_points.csv: {where} marks vertex/vertices {missing} "
                    f"visible without a position"
                )
                continue

            hidden = frame_points[~frame_points["visible"]]
            positioned = hidden[["x_px", "y_px"]].notna().any(axis=1)
            if positioned.any():
                problems.append(
                    f"court_points.csv: {where} gives a position for "
                    f"vertex/vertices {sorted(hidden.loc[positioned, 'vertex'])}, "
                    f"which are marked not visible"
                )

            if len(visible) < MIN_POINTS_FOR_HOMOGRAPHY:
                problems.append(
                    f"court_points.csv: {where} has {len(visible)} visible "
                    f"vertices; {MIN_POINTS_FOR_HOMOGRAPHY} are needed to place "
                    f"the court, and more than that to tell a mistake from a fit"
                )
                continue

            residuals = reprojection_residuals(
                list(visible["vertex"]), visible[["x_px", "y_px"]].to_numpy()
            )
            if residuals is None:
                problems.append(
                    f"court_points.csv: {where} has no court that fits its "
                    f"vertices; they may be collinear or badly mismatched"
                )
                continue

            worst = int(np.argmax(residuals))
            if residuals[worst] > MAX_RESIDUAL_PX:
                problems.append(
                    f"court_points.csv: {where} places vertex "
                    f"{int(visible['vertex'].iloc[worst])} "
                    f"{residuals[worst]:.1f} px from where the other vertices "
                    f"put it (limit {MAX_RESIDUAL_PX:.0f} px)"
                )

    missing_courts = sorted(set(clips["clip_id"]) - set(court_points["clip_id"]))
    if missing_courts:
        problems.append(
            f"court_points.csv: {len(missing_courts)} clip(s) have no court "
            f"keypoints: {missing_courts}"
        )

    return problems


def _track_problems(
    clips: pd.DataFrame, events: pd.DataFrame, tracks: pd.DataFrame
) -> list[str]:
    """
    Whether the ground-truth track can stand in for the detector.

    The check that matters is the last one: a landing with no box on its own
    frame cannot be measured with the models taken out, because the thing the
    algorithm needs at that instant is missing from its input.
    """
    problems = []
    frame_counts = dict(zip(clips["clip_id"], clips["frame_count"]))
    sizes = dict(zip(clips["clip_id"], zip(clips["width"], clips["height"])))

    for clip_id, clip_tracks in tracks.groupby("clip_id", sort=True):
        if clip_id not in frame_counts:
            problems.append(f"tracks.csv: clip {clip_id!r} is not in clips.csv")
            continue

        past = clip_tracks[clip_tracks["frame"] >= frame_counts[clip_id]]
        if not past.empty:
            problems.append(
                f"tracks.csv: clip {clip_id!r} has {len(past)} box(es) past the "
                f"{frame_counts[clip_id]} frames of the clip, e.g. frame "
                f"{int(past['frame'].min())}"
            )

        inverted = clip_tracks[
            (clip_tracks["x2"] <= clip_tracks["x1"])
            | (clip_tracks["y2"] <= clip_tracks["y1"])
        ]
        if not inverted.empty:
            problems.append(
                f"tracks.csv: clip {clip_id!r} has {len(inverted)} box(es) with "
                f"no area, e.g. frame {int(inverted['frame'].min())}"
            )

        width, height = sizes[clip_id]
        outside = clip_tracks[
            (clip_tracks["x2"] > width) | (clip_tracks["y2"] > height)
        ]
        if not outside.empty:
            problems.append(
                f"tracks.csv: clip {clip_id!r} has {len(outside)} box(es) "
                f"reaching past its {width}x{height} frame, e.g. frame "
                f"{int(outside['frame'].min())}"
            )

    # Every landing needs the shuttle's position on its own frame, or the
    # algorithm cannot be run on ground truth there.
    annotated = set(zip(tracks["clip_id"], tracks["frame"]))
    missing = [
        schema.event_label(event)
        for event in events.itertuples()
        if event.clip_id in set(tracks["clip_id"])
        and (event.clip_id, event.frame) not in annotated
    ]
    if missing:
        problems.append(
            f"tracks.csv: {len(missing)} landing(s) have no ground-truth box on "
            f"their own frame: {', '.join(missing[:5])}"
            + (" ..." if len(missing) > 5 else "")
        )

    return problems


def _stale_derivation_problems(
    clips: pd.DataFrame, events: pd.DataFrame, court_points: pd.DataFrame
) -> list[str]:
    """
    Catch stale variables in a dataset that are no longer valid.
    """
    from .clips import is_empty, observed_camera_mode, observed_court_type

    problems = []
    placed = set(court_points["clip_id"])

    for clip in clips.itertuples():
        if clip.clip_id in placed:
            mode = observed_camera_mode(court_points, clip.clip_id)
            if (
                mode is not None
                and not is_empty(clip.camera_mode)
                and mode != clip.camera_mode
            ):
                problems.append(
                    f"clips.csv: clip {clip.clip_id!r} says camera_mode "
                    f"{clip.camera_mode!r}, but its court keypoints say "
                    f"{mode!r}"
                )

        court_type = observed_court_type(events, clip.clip_id)
        if (
            court_type is not None
            and not is_empty(clip.court_type)
            and court_type != clip.court_type
        ):
            problems.append(
                f"clips.csv: clip {clip.clip_id!r} says court_type "
                f"{clip.court_type!r}, but its landings were called against "
                f"{court_type!r}"
            )

    problems += _stale_position_problems(events, court_points)
    if problems:
        problems.append("Run 'sokil dataset refresh'.")

    return problems


def _stale_position_problems(
    events: pd.DataFrame, court_points: pd.DataFrame
) -> list[str]:
    """
    Find labeled hits that at the time of annotation didn't have court keypoints,
    and therefore, their positions couldn't be solved.
    """
    placed = set(court_points["clip_id"])

    stale = [
        schema.event_label(event)
        for event in events.itertuples()
        if event.clip_id in placed and not pd.isna(event.x_px) and pd.isna(event.x_m)
    ]
    if not stale:
        return []

    return [
        f"events.csv: {len(stale)} landing(s) have a click but no position, "
        f"though their clip has a court: {', '.join(stale[:5])}"
        + (" ..." if len(stale) > 5 else "")
    ]


def _split_problems(clips: pd.DataFrame, splits: dict[str, list[str]]) -> list[str]:
    """
    Splits are drawn on whole clips, and this is what keeps them that way.

    Frames from one rally are near-duplicates, so a clip appearing in two splits
    puts a frame's twin in both train and test and inflates every score measured
    afterwards. It is invisible in a diff, which is why it is checked here.
    """
    problems = []
    known = set(clips["clip_id"])
    seen: dict[str, str] = {}

    for name, clip_ids in splits.items():
        for clip_id in clip_ids:
            if clip_id not in known:
                problems.append(
                    f"splits/{name}.txt: clip {clip_id!r} is not in clips.csv"
                )
            if clip_id in seen:
                problems.append(
                    f"splits/{name}.txt: clip {clip_id!r} is also in the "
                    f"{seen[clip_id]} split; a clip belongs to exactly one"
                )
            else:
                seen[clip_id] = name

    unassigned = sorted(known - set(seen))
    if unassigned and seen:
        problems.append(
            f"clips.csv: {len(unassigned)} clip(s) are in no split: {unassigned}"
        )

    return problems
