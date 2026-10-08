"""
Turning a clicked landing into a position on the court.
"""

import logging

import numpy as np
import pandas as pd

from .court import image_to_court_homography
from .schema import Line

logger = logging.getLogger(__name__)

# The court model measures in centimetres; the dataset records landings in
# metres, which is the unit a line call is discussed in.
CM_PER_M = 100.0


def court_keyframe(
    court_points: pd.DataFrame, clip_id: str, frame: int
) -> pd.DataFrame | None:
    """
    The court keypoints in force at a frame: those of the latest keyframe at or
    before it, since a keyframe holds until the next one replaces it.
    """
    clip_points = court_points[court_points["clip_id"] == clip_id]
    if clip_points.empty:
        return None

    applicable = clip_points[clip_points["frame"] <= frame]
    if applicable.empty:
        # Annotated later than the landing: the first keyframe is still the
        # best description of a court nobody has seen move before it.
        applicable = clip_points[clip_points["frame"] == clip_points["frame"].min()]

    return applicable[applicable["frame"] == applicable["frame"].max()]


def image_to_court(court_points: pd.DataFrame, clip_id: str, frame: int):
    """
    A function mapping an image pixel to court metres, or None if this clip's
    keypoints cannot place the court.
    """
    keyframe = court_keyframe(court_points, clip_id, frame)
    if keyframe is None:
        logger.warning("No court keypoints for clip %r", clip_id)
        return None

    visible = keyframe[keyframe["visible"]]
    # The same fit the evaluation places its court with. Fitting one direction
    # and inverting the other gives answers that differ by centimetres — which
    # is the unit the landings are measured in, so the two would disagree about
    # the very thing they are there to measure.
    matrix = image_to_court_homography(
        list(visible["vertex"]), visible[["x_px", "y_px"]].to_numpy()
    )
    if matrix is None:
        logger.warning("Court keypoints for clip %r do not place a court", clip_id)
        return None

    def to_court(x_px: float, y_px: float) -> tuple[float, float] | None:
        projected = matrix @ np.array([x_px, y_px, 1.0])
        if np.isclose(projected[2], 0):
            return None
        x_cm, y_cm = projected[:2] / projected[2]
        return float(x_cm) / CM_PER_M, float(y_cm) / CM_PER_M

    return to_court


def boundary(court_type: str) -> tuple[float, float, float, float]:
    """
    The in-bounds rectangle in metres, as (x_min, x_max, y_min, y_max).

    Singles is narrower than doubles but the same length: the side corridors are
    out, the back corridor is not. Getting that the wrong way round would flip
    every call near a sideline.
    """
    from ..court import CourtModel

    model = CourtModel()
    corners = (
        model.singles_boundaries()
        if court_type == "singles"
        else model.doubles_boundaries()
    )
    points = np.array(corners, dtype=float) / CM_PER_M

    return (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
    )


def derive_call(x_m: float, y_m: float, court_type: str) -> str:
    """
    Where a landing falls against the boundary, as a second opinion on the call.

    A shuttle touching the line is in, so the bounds are inclusive.
    """
    from .schema import Call

    x_min, x_max, y_min, y_max = boundary(court_type)
    inside = x_min <= x_m <= x_max and y_min <= y_m <= y_max
    return Call.IN.value if inside else Call.OUT.value


def derive_line(x_m: float, y_m: float, court_type: str) -> str:
    """
    Which boundary line the landing was nearest, so an out call says where.

    Only the two lines a rally is judged against: service lines matter to a
    serve, which this dataset does not record.
    """
    x_min, x_max, y_min, y_max = boundary(court_type)

    to_side = min(abs(x_m - x_min), abs(x_m - x_max))
    to_base = min(abs(y_m - y_min), abs(y_m - y_max))

    return Line.SIDELINE.value if to_side <= to_base else Line.BASELINE.value
