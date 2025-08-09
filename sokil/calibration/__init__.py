"""
Camera calibration: measuring what the lens does, once per camera.

The court solver reasons about straight lines, so a lens that bends them feeds
error into every measurement the review makes. Calibration measures that bend
from photographs of a checkerboard; the review then undoes it on every frame.

Two steps, in order:

1. Collect frames of the board from many angles — `select_frames` for the
   automatic pick, `select_frames_interactively` to choose them yourself.
2. Solve the lens model from them with `calibrate`, and save the result where
   `sokil review` can be pointed at it.
"""

from .compute import (
    CalibrationError,
    CalibrationResult,
    CheckerBoard,
    calibrate,
    distortion_preview,
    find_corners,
    image_paths,
    largest_detectable_board,
)
from .frames import (
    require_display,
    select_frames,
    select_frames_interactively,
)

__all__ = [
    "CalibrationError",
    "CalibrationResult",
    "CheckerBoard",
    "calibrate",
    "distortion_preview",
    "find_corners",
    "image_paths",
    "largest_detectable_board",
    "require_display",
    "select_frames",
    "select_frames_interactively",
]
