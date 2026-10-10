"""
Building clips.csv from the recordings.
"""

import hashlib
import logging
import re
from pathlib import Path

import cv2
import pandas as pd

from . import schema

logger = logging.getLogger(__name__)

# Read in blocks rather than whole: a clip is bigger than it needs to be in
# memory just to be summed.
DIGEST_BLOCK = 1024 * 1024


def is_empty(value) -> bool:
    """
    Whether a cell has an answer in it.

    A column read back from CSV is empty as NaN or NA rather than None, and the
    kind depends on the column's type, so asking `is None` quietly decides that
    every blank cell is filled — which stops them being filled in and stops
    them being reported.
    """
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # Not a scalar, so it is something rather than nothing.
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(DIGEST_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def session_id(clip_id: str) -> str | None:
    """
    The session a clip belongs to, if its name carries a date.

    Clips recorded on one day at one hall are one session, which is the
    grouping a split is drawn on. A name with no date in it gets nothing rather
    than a guess.
    """
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", clip_id)
    if match is None:
        return None
    return "s-{}-{}-{}".format(*match.groups())


def probe(video_path: Path) -> dict:
    """What the clip says about itself."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        return {
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def observed_court_type(events: pd.DataFrame, clip_id: str) -> str | None:
    """
    The boundary this clip's landings were judged against.

    Taken from the calls rather than asked for again: the annotator already
    chose it per landing, and a clip whose landings disagree is a question for
    them, not something to settle by majority.
    """
    called = events[events["clip_id"] == clip_id]["court_type"].dropna()
    kinds = set(called.unique())

    if not kinds:
        return None
    if len(kinds) > 1:
        logger.warning(
            "Clip %r has landings judged as %s; leaving court_type empty",
            clip_id,
            sorted(kinds),
        )
        return None

    return kinds.pop()


def observed_camera_static(court_points: pd.DataFrame, clip_id: str) -> bool | None:
    """
    Whether the court was annotated as holding still.

    If there's only 1 keyframe, the camera was static; otherwise, moving
    """
    keyframes = court_points[court_points["clip_id"] == clip_id]["frame"]
    if keyframes.empty:
        return None

    return keyframes.nunique() == 1


def observed_camera_mode(court_points: pd.DataFrame, clip_id: str) -> str | None:
    """
    Which part of the court the camera was pointed at.

    Read from which vertices the annotator could see: a camera on one corridor
    sees that corridor's vertices and not the far side's, so the visible set
    identifies the mode without anybody being asked again. A set that does not
    clearly belong to one mode returns nothing, because the modes decide which
    boundaries the system may adjudicate and a wrong one adjudicates the wrong
    line.
    """
    from ..court import CourtCameraMode, CourtModel

    visible = set(
        court_points[(court_points["clip_id"] == clip_id) & court_points["visible"]][
            "vertex"
        ]
    )
    if not visible:
        return None

    model = CourtModel()
    candidates = {
        CourtCameraMode.LEFT_CORRIDOR.value: {
            1,
            2,
            9,
            10,
            11,
            12,
            19,
            20,
            21,
            22,
            29,
            30,
        },
        CourtCameraMode.RIGHT_CORRIDOR.value: {
            4,
            5,
            6,
            7,
            14,
            15,
            16,
            17,
            24,
            25,
            26,
            27,
        },
        CourtCameraMode.BOTTOM_CORRIDOR.value: set(range(1, 11)),
        CourtCameraMode.TOP_CORRIDOR.value: set(range(21, 31)),
        CourtCameraMode.FULL.value: set(range(1, len(model.vertices) + 1)),
    }

    # A corridor's own vertices must all be there. Extra ones are expected —
    # the centre line is visible from a corridor without belonging to it — so
    # the test is coverage of the mode, and the tightest mode covered wins.
    covered = {
        mode: vertices for mode, vertices in candidates.items() if vertices <= visible
    }
    if not covered:
        logger.warning(
            "Clip %r sees vertices %s, which is not one camera mode; "
            "leaving camera_mode empty",
            clip_id,
            sorted(visible),
        )
        return None

    return max(covered, key=lambda mode: len(covered[mode]))


# TODO: turn rows into data classes
# Also check if validation can be done at the data class level
def clip_rows(
    videos: dict[str, Path],
    events: pd.DataFrame | None = None,
    court_points: pd.DataFrame | None = None,
    existing: pd.DataFrame | None = None,
) -> list[dict]:
    """
    A clips.csv row per clip, with whatever can be established already filled.

    Anything a previous clips.csv already said is kept: this fills gaps and
    never overwrites an answer somebody gave by hand.
    """
    known = {}
    if existing is not None and not existing.empty:
        known = {row["clip_id"]: dict(row) for _, row in existing.iterrows()}

    rows = []
    for clip_id, video_path in sorted(videos.items()):
        row = {column: None for column in schema.columns("clips")}
        row.update(known.get(clip_id, {}))

        row["clip_id"] = clip_id
        row.update(probe(video_path))
        row["source_sha256"] = sha256(video_path)

        if is_empty(row.get("session_id")):
            row["session_id"] = session_id(clip_id)

        if is_empty(row.get("court_type")) and events is not None:
            row["court_type"] = observed_court_type(events, clip_id)

        if is_empty(row.get("camera_static")) and court_points is not None:
            row["camera_static"] = observed_camera_static(court_points, clip_id)

        if is_empty(row.get("camera_mode")) and court_points is not None:
            row["camera_mode"] = observed_camera_mode(court_points, clip_id)

        rows.append(row)

    return rows


def blank_columns(rows: list[dict]) -> dict[str, list[str]]:
    """Which clips are still missing which columns, for reporting."""
    missing = {}
    for row in rows:
        empty = [column for column, value in row.items() if is_empty(value)]
        if empty:
            missing[row["clip_id"]] = empty
    return missing
