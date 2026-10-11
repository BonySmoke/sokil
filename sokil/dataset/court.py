"""
TODO: consider merging this with sokil/court.py
Ground-truth court keypoints.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# The CVAT skeleton names its nodes after the court model's vertex numbers,
# except if aliases are provided. Record aliases in this dict.
# Numbering starts at 1 in CVAT.
VERTEX_ALIASES = {
    "bottom left": 1,
    "bottom right": 5,
    "top right": 26,
    "top left": 30,
}

# A homography needs at least four points
MIN_POINTS_FOR_HOMOGRAPHY = 4

# How far a vertex may sit from where the fitted homography puts it
MAX_RESIDUAL_PX = 5.0


class UnknownVertex(ValueError):
    """A keypoint names something that is not a court vertex."""


def vertex_count() -> int:
    from ..court import CourtModel

    return len(CourtModel().vertices)


def resolve_vertex(name: str) -> int:
    """
    The court vertex a CVAT skeleton node refers to, as a 1-based number.

    Node names come from whoever drew the skeleton, so both the numbers and the
    four corner names are accepted; everything downstream uses the number.
    """
    key = str(name).strip().lower()
    if key in VERTEX_ALIASES:
        return VERTEX_ALIASES[key]

    if key.isdigit():
        number = int(key)
        if 1 <= number <= vertex_count():
            return number

    raise UnknownVertex(
        f"{name!r} is neither a court vertex number (1-{vertex_count()}) "
        f"nor one of {sorted(VERTEX_ALIASES)}"
    )


def court_coordinates(vertices: list[int]) -> np.ndarray:
    """The court-model position, in centimetres, of each 1-based vertex number."""
    from ..court import vertex_coordinates

    return vertex_coordinates(vertices)


def fit_homography(vertices: list[int], points: np.ndarray):
    """
    The court->image homography these annotated vertices imply, or None.

    None means the points cannot define one: too few of them, or an arrangement
    a homography cannot be fitted to, such as vertices that all lie on one line.
    """
    import cv2

    if len(vertices) < MIN_POINTS_FOR_HOMOGRAPHY:
        return None

    source = court_coordinates(vertices).astype(np.float32)
    target = np.asarray(points, dtype=np.float32)

    # Plain least squares rather than RANSAC: these are hand-placed ground
    # truth, so an outlier is a mistake to report, not noise to discard.
    matrix, _ = cv2.findHomography(source, target, 0)
    return matrix


def reprojection_residuals(
    vertices: list[int], points: np.ndarray
) -> np.ndarray | None:
    """
    How far each annotated vertex sits from where the fitted homography puts it,
    in pixels, or None if no homography could be fitted.
    """
    matrix = fit_homography(vertices, points)
    if matrix is None:
        return None

    source = court_coordinates(vertices).astype(np.float64)
    homogeneous = np.hstack([source, np.ones((len(source), 1))])
    projected = homogeneous @ matrix.T

    w = projected[:, 2]
    if np.any(np.isclose(w, 0)):
        return None

    projected = projected[:, :2] / w[:, None]
    return np.linalg.norm(projected - np.asarray(points, dtype=np.float64), axis=1)


def image_to_court_homography(vertices: list[int], points: np.ndarray):
    """The image->court homography these vertices imply, or None."""
    from ..court import homography_from_vertices

    return homography_from_vertices(vertices, points)


def court_from_points(court_points, clip_id: str, frame: int = 0, court_config=None):
    """
    A Court placed by the annotated keypoints rather than solved from the video.

    This is the court the evaluation adjudicates against when it is measuring
    the algorithm rather than the solver: the geometry is a human's, so an
    error in the verdict is the algorithm's own.
    """
    from ..court import Court
    from .landing import court_keyframe

    keyframe = court_keyframe(court_points, clip_id, frame)
    if keyframe is None:
        raise ValueError(f"No court keypoints for clip {clip_id!r}")

    visible = keyframe[keyframe["visible"]]
    court = Court.from_points(
        list(visible["vertex"]),
        visible[["x_px", "y_px"]].to_numpy(),
        court_config=court_config,
    )
    if court is None:
        raise ValueError(f"Court keypoints for clip {clip_id!r} place no court")

    return court
