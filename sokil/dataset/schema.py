"""
The annotation dataset's table schemas.
"""

from enum import Enum

import pandera.pandas as pa

# A clip id has to survive being a directory name, a CSV cell, a YOLO file stem
# and a URL, so it is restricted to what all four agree on.
CLIP_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"

SHA256_PATTERN = r"^[0-9a-f]{64}$"

# Where the source recording lives, resolvable from any machine: a path
# relative to the dataset root (the release carries the file), or a URL. An
# absolute path is rejected because it means nothing on anyone else's disk.
SOURCE_URI_PATTERN = r"^(https?://\S+|[^/\\][^\s]*)$"


# TODO: consider using this in the main package instead of raw strings
class Call(str, Enum):
    """
    The verdict on a landing.
    """

    IN = "in"
    OUT = "out"
    UNKNOWN = "unknown"


# TODO: consider using this in the main code package. Maybe sokil/schema.py
class Line(str, Enum):
    """Which line the landing was judged against."""

    SIDELINE = "sideline"
    BASELINE = "baseline"
    LONG_SERVICE = "long_service"
    SHORT_SERVICE = "short_service"
    CENTRE = "centre"


def _enum_values(enum: type[Enum]) -> list[str]:
    return [member.value for member in enum]


def _pattern_column(pattern: str, **kwargs) -> pa.Column:
    return pa.Column(str, pa.Check.str_matches(pattern), **kwargs)


def _enum_column(enum: type[Enum], **kwargs) -> pa.Column:
    return pa.Column(str, pa.Check.isin(_enum_values(enum)), **kwargs)


def _pixel_column(nullable: bool = True, **kwargs) -> pa.Column:
    return pa.Column(float, pa.Check.ge(0), nullable=nullable, **kwargs)


# Every clip that exists. Nothing else may reference a clip_id absent from here,
# which makes this file the dataset's manifest.
CLIPS = pa.DataFrameSchema(
    {
        "clip_id": _pattern_column(CLIP_ID_PATTERN, unique=True),
        # The recording this clip's frames were decoded from, kept as a digest
        # plus a location rather than as a copy. The digest is identity: it is
        # what proves a clip came from the footage it names, and it stays valid
        # whether the release ships the video, links it, or withholds it.
        "source_sha256": _pattern_column(SHA256_PATTERN),
        "source_uri": _pattern_column(SOURCE_URI_PATTERN, nullable=True),
        # Extracted frames, in cv2 decode order. Bounds every frame index, so a
        # landing labelled past the end of its clip is caught as the mistake it
        # is rather than silently dropping out of a join.
        "frame_count": pa.Column(int, pa.Check.gt(0)),
        "fps": pa.Column(float, pa.Check.gt(0)),
        "width": pa.Column(int, pa.Check.gt(0)),
        "height": pa.Column(int, pa.Check.gt(0)),
        # The grouping key splits are drawn on: clips recorded in one sitting
        # share a hall, a camera pose and a light, so scoring a model on a clip
        # whose siblings it trained on measures memory rather than generality.
        # An opaque id rather than a venue name — a split needs to know which
        # clips belong together, not where they were shot.
        "session_id": pa.Column(str),
        "camera_mode": pa.Column(str),
        "court_type": pa.Column(str),
        # Whether the camera held still for the whole clip. It is what tells a
        # reader that one court keyframe describes the clip completely rather
        # than approximately, and the pipeline solves the court once, so a clip
        # that fails it is out of scope until that changes.
        "camera_static": pa.Column(bool),
        "faces_blurred": pa.Column(bool),
        "license": pa.Column(str),
        "notes": pa.Column(str, nullable=True),
    },
    strict=True,
    ordered=True,
    name="clips",
)

# One row per landing. Sparse by nature — a few hundred rows against tens of
# thousands of frames — which is why these are rows of their own rather than
# columns on a per-frame table that would be empty on all but a handful.
EVENTS = pa.DataFrameSchema(
    {
        "clip_id": _pattern_column(CLIP_ID_PATTERN),
        "frame": pa.Column(int, pa.Check.ge(0)),
        "call": _enum_column(Call),
        # Which boundary the call was made against. The same landing is in for
        # doubles and out for singles, so a verdict without this is unreadable.
        # Empty where the call is unknown and no boundary was considered.
        "court_type": pa.Column(str, nullable=True),
        "line": _enum_column(Line, nullable=True),
        # Where the annotator clicked, in image pixels, and the same point in
        # court metres through the clip's court keypoints. Pixels are the
        # observation and metres are derived, which lets a benchmark report an
        # error in centimetres instead of a bare right or wrong.
        "x_px": _pixel_column(),
        "y_px": _pixel_column(),
        "x_m": pa.Column(float, nullable=True),
        "y_m": pa.Column(float, nullable=True),
        "annotator": pa.Column(str),
        "notes": pa.Column(str, nullable=True),
    },
    strict=True,
    ordered=True,
    # A landing is identified by which frame of which clip it is, the same way
    # a court keypoint is. Writing that pair out a third time as one string
    # would only be a value that can disagree with the two it came from.
    unique=["clip_id", "frame"],
    name="events",
)


class PointSource(str, Enum):
    """
    Where a court keypoint's position came from.

    Derived rather than annotated: the pre-annotation handed to CVAT is kept, so
    the builder can diff against it and see for itself which points a human
    moved. That makes automation bias measurable — a clip that comes back with
    thirty untouched points is visible as exactly that — without asking the
    annotator to remember anything.
    """

    SOLVER = "solver"
    SOLVER_ADJUSTED = "solver_adjusted"
    MANUAL = "manual"


# One row per court vertex per keyframe. Generated from the CVAT skeleton
# export, so it is rebuilt rather than edited.
#
# A row is read as valid from its frame until the next keyframe of the same
# clip. A static camera needs one keyframe at frame 0, which is the per-clip
# cost anyway; a moving camera needs one wherever the view changed, and nothing
# about the table changes. The column costs one integer now and saves a
# migration later, which is the right way round for a published schema.
COURT_POINTS = pa.DataFrameSchema(
    {
        "clip_id": _pattern_column(CLIP_ID_PATTERN),
        "frame": pa.Column(int, pa.Check.ge(0)),
        # CourtModel's own vertex numbering, 1-based, so a point joins directly
        # to the metric coordinate the court model already holds.
        "vertex": pa.Column(int, pa.Check.gt(0)),
        # Empty where the vertex is out of frame, which is normal: a corridor
        # camera mode sees a fraction of the court.
        "x_px": _pixel_column(),
        "y_px": _pixel_column(),
        "visible": pa.Column(bool),
        "source": _enum_column(PointSource),
    },
    strict=True,
    ordered=True,
    unique=["clip_id", "frame", "vertex"],
    name="court_points",
)

# One row per frame the shuttle was annotated on. Generated from the CVAT
# shuttle tracks, so it is rebuilt rather than edited.
#
# This is what lets the hit detection be measured without the models in the
# loop: substituted for the detector's output, it drives the same tracker code
# over the same frames, so the cork position and the blur are measured from
# real pixels and only the box differs. A figure from that does not move when
# a model is retrained, which is what makes tuning mean anything.
TRACKS = pa.DataFrameSchema(
    {
        "clip_id": _pattern_column(CLIP_ID_PATTERN),
        "frame": pa.Column(int, pa.Check.ge(0)),
        # The box as the tracker holds it: top left and bottom right, in
        # pixels of the frame as decoded. ShuttleTrack.xyxy is the same four
        # numbers, so nothing has to be converted at the boundary.
        "x1": _pixel_column(nullable=False),
        "y1": _pixel_column(nullable=False),
        "x2": _pixel_column(nullable=False),
        "y2": _pixel_column(nullable=False),
    },
    strict=True,
    ordered=True,
    # One shuttle per frame: the pipeline follows the single most confident
    # box, so ground truth with two would not say which one it should have
    # followed.
    unique=["clip_id", "frame"],
    name="tracks",
)

SCHEMAS = {
    "clips": CLIPS,
    "events": EVENTS,
    "court_points": COURT_POINTS,
    "tracks": TRACKS,
}

# What a table is sorted by before it is written, so a diff shows the rows that
# changed rather than the order pandas happened to produce.
SORT_KEYS = {
    "clips": ["clip_id"],
    "events": ["clip_id", "frame"],
    "court_points": ["clip_id", "frame", "vertex"],
    "tracks": ["clip_id", "frame"],
}

# Rounding applied on write. Pixels carry two decimals, past the precision of a
# hand-placed click; metres carry four, a tenth of a millimetre, well past what
# a homography recovers. Fixing this at write time keeps float formatting from
# showing up as diff noise.
ROUNDING = {
    "events": {"x_px": 2, "y_px": 2, "x_m": 4, "y_m": 4},
    "court_points": {"x_px": 2, "y_px": 2},
    "tracks": {"x1": 2, "y1": 2, "x2": 2, "y2": 2},
}


def columns(table: str) -> list[str]:
    """The table's column order, which is also the order it is written in."""
    return list(SCHEMAS[table].columns)


def dtypes(table: str) -> dict:
    """Every column's declared dtype, for typing a table that has no rows yet."""
    return {name: column.dtype.type for name, column in SCHEMAS[table].columns.items()}


def read_dtypes(table: str) -> dict[str, str]:
    """
    The dtypes a table has to be read with.

    Text columns are named explicitly because a CSV cannot say what an empty
    column held: a notes column nobody filled in comes back as float64 NaN and
    stops looking like text at all. Numbers and booleans are left to pandas,
    which reads them from the values.
    """
    return {
        name: "string"
        for name, column in SCHEMAS[table].columns.items()
        if str(column.dtype).startswith("str")
    }


def event_label(event) -> str:
    """
    How a landing is named in a message to a human.

    Built where it is needed rather than stored: it identifies a row by the two
    columns that already identify it, so there is nothing to fall out of step.
    """
    return f"{event.clip_id}:{event.frame}"
