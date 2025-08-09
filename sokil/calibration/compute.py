"""
Camera intrinsics from checkerboard images.

A wide lens bends straight lines, and the court solver has only straight lines
to work with, so an uncorrected wide-angle camera inherits that error in every
measurement it makes. Calibrating measures the bend once per camera; the review
then undoes it on every frame.

This is per camera and per lens setting: recalibrate after changing the lens or
the zoom, but not after moving the camera.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# Below this many usable views the solution is under-constrained: the board has
# to be seen at enough different angles to separate lens distortion from the
# board simply being far away.
MINIMUM_VIEWS = 10


class CalibrationError(RuntimeError):
    """Calibration could not be completed."""


@dataclass(frozen=True)
class CheckerBoard:
    """
    The printed board being photographed.

    Corners are counted on the INSIDE of the board: a 10x7 square board has 9x6
    inner corners. Counting the squares instead is the usual reason nothing is
    ever detected.
    """

    cols: int = 8  # inner corners across the width
    rows: int = 5  # inner corners down the height
    square: float = 2.5  # side of one square, in cm

    @property
    def size(self) -> tuple[int, int]:
        return (self.cols, self.rows)

    def object_points(self) -> np.ndarray:
        """
        The corners in the board's own frame, in cm, all at z=0.

        Identical for every image — the board is rigid, so only its pose
        changes — which is what lets many views constrain one lens model.
        """
        points = np.zeros((self.cols * self.rows, 3), np.float32)
        points[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        return points * self.square

    def __str__(self) -> str:
        return f"{self.cols}x{self.rows} inner corners at {self.square}cm"


@dataclass
class CalibrationResult:
    """What a calibration run produced."""

    camera_matrix: np.ndarray  # K
    distortion: np.ndarray
    rms: float  # reprojection error, in pixels
    image_size: tuple[int, int]  # (width, height)
    used: list[Path]
    skipped: list[Path]

    @property
    def quality(self) -> str:
        """How much to trust this result, in words."""
        if self.rms < 0.5:
            return "excellent"
        if self.rms < 1.0:
            return "good"
        if self.rms <= 1.5:
            return "usable"
        return "poor"

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        """
        Write the two arrays the review expects.

        :returns: the intrinsics and distortion paths, in that order — the same
            order `sokil review --camera-intrinsics-path --camera-dist-path`
            takes them.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        intrinsics_path = directory / "camera_K.npy"
        distortion_path = directory / "camera_dist.npy"

        np.save(intrinsics_path, self.camera_matrix)
        np.save(distortion_path, self.distortion)

        logger.info("Wrote %s and %s", intrinsics_path, distortion_path)
        return intrinsics_path, distortion_path


def find_corners(image: np.ndarray, board: CheckerBoard) -> np.ndarray | None:
    """
    Locate the board's inner corners, to sub-pixel accuracy.

    :returns: the corners, or None when the board is not fully visible.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # This one detector is used everywhere, including to screen video frames.
    # A cheaper screen is tempting, but OpenCV's FAST_CHECK misses boards this
    # finds readily, and any disagreement between the two would select frames
    # that calibration then rejects.
    found, corners = cv2.findChessboardCornersSB(
        gray, board.size, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    return corners if found else None


def largest_detectable_board(
    image: np.ndarray, max_cols: int = 12, max_rows: int = 10
) -> tuple[int, int] | None:
    """
    The biggest checkerboard this image actually contains.

    Any sub-grid of a checkerboard is itself a valid checkerboard, so several
    sizes match and the largest is the real board. Slow, so this is only used
    to explain a total failure to detect — where the cause is almost always
    counting squares rather than inner corners.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    best = None
    for cols in range(3, max_cols + 1):
        for rows in range(3, max_rows + 1):
            if rows > cols:
                continue
            found, _ = cv2.findChessboardCornersSB(
                gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            )
            if found and (best is None or cols * rows > best[0] * best[1]):
                best = (cols, rows)

    return best


def image_paths(directory: str | Path) -> list[Path]:
    """Every calibration image in a directory, in a stable order."""
    directory = Path(directory)
    if not directory.is_dir():
        raise CalibrationError(f"No such directory: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
    )


def calibrate(
    images: str | Path | list[Path],
    board: CheckerBoard | None = None,
    minimum_views: int = MINIMUM_VIEWS,
    preview: bool = False,
) -> CalibrationResult:
    """
    Solve the lens model from a set of checkerboard photographs.

    :param images: a directory of images, or the image paths themselves.
    :param board: the printed board; defaults to 9x6 inner corners at 2.5cm.
    :param minimum_views: refuse to solve from fewer usable views than this.
    :param preview: draw each detection on screen; needs a display.
    :raises CalibrationError: too few usable views, or images of mixed sizes.
    """
    from .frames import require_display

    board = board or CheckerBoard()
    paths = image_paths(images) if isinstance(images, (str, Path)) else list(images)

    if not paths:
        raise CalibrationError(f"No images found in '{images}'")

    if preview:
        require_display("--preview")

    logger.info("Looking for a %s board in %d images", board, len(paths))

    object_points, image_points = [], []
    first_readable: np.ndarray | None = None
    used: list[Path] = []
    skipped: list[Path] = []
    sizes: set[tuple[int, int]] = set()

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("  skipped %s (unreadable)", path.name)
            skipped.append(path)
            continue

        height, width = image.shape[:2]
        sizes.add((width, height))
        if first_readable is None:
            first_readable = image

        corners = find_corners(image, board)
        if corners is None:
            logger.info("  skipped %s (board not fully visible)", path.name)
            skipped.append(path)
            continue

        object_points.append(board.object_points())
        image_points.append(corners)
        used.append(path)
        logger.info("  using %s", path.name)

        if preview:
            annotated = image.copy()
            cv2.drawChessboardCorners(annotated, board.size, corners, True)
            cv2.imshow("Corners", annotated)
            cv2.waitKey(500)

    if preview:
        cv2.destroyAllWindows()

    # Every view has to come from the same camera at the same settings, and one
    # lens model is solved for all of them. Mixed sizes mean mixed sources, and
    # would produce a plausible-looking solution that is wrong for both.
    if len(sizes) > 1:
        listed = ", ".join(f"{width}x{height}" for width, height in sorted(sizes))
        raise CalibrationError(
            f"The images are not all the same size ({listed}). Calibrate from "
            f"one camera at one setting."
        )

    if len(used) < minimum_views:
        message = (
            f"Only {len(used)} usable views — at least {minimum_views} are "
            f"needed. Recapture with the board fully in frame, held flat, at "
            f"more varied angles and distances."
        )

        # Detecting nothing at all is a different problem from detecting too
        # little: the board being described wrongly, not photographed badly.
        # Say what the images actually contain rather than leaving a dead end.
        if not used and first_readable is not None:
            logger.info("Nothing matched %s; looking for what these images hold", board)
            actual = largest_detectable_board(first_readable)
            message = (
                f"No image contained a {board}."
                + (
                    f" These images look like a {actual[0]}x{actual[1]} board — "
                    f"retry with --board-cols {actual[0]} --board-rows {actual[1]}."
                    if actual
                    else " No checkerboard could be found in them at all."
                )
                + " Remember to count inner corners, not squares."
            )

        raise CalibrationError(message)

    image_size = sizes.pop()
    logger.info("Calibrating from %d views at %dx%d", len(used), *image_size)

    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    result = CalibrationResult(
        camera_matrix=camera_matrix,
        distortion=distortion,
        rms=float(rms),
        image_size=image_size,
        used=used,
        skipped=skipped,
    )

    logger.info("Reprojection error %.4f px (%s)", result.rms, result.quality)
    if result.quality == "poor":
        logger.warning(
            "A high reprojection error usually means the board was bent, "
            "partly out of frame, or photographed from too few angles"
        )

    return result


def distortion_preview(result: CalibrationResult, step: int = 40) -> np.ndarray:
    """
    A picture of what the lens does, as a straight grid beside its bent self.

    Green is the ideal grid, red is where this lens would put it. The further
    red drifts from green towards the edges, the more the correction matters.
    """
    width, height = result.image_size

    map_x, map_y = cv2.initUndistortRectifyMap(
        result.camera_matrix,
        result.distortion,
        None,
        result.camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )

    grid = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, step):
        cv2.line(grid, (0, y), (width, y), (0, 255, 0), 1)
    for x in range(0, width, step):
        cv2.line(grid, (x, 0), (x, height), (0, 255, 0), 1)

    distorted = cv2.remap(grid, map_x, map_y, cv2.INTER_LINEAR)

    overlay = grid.copy()
    overlay[distorted[:, :, 1] > 0] = (0, 0, 255)
    return cv2.addWeighted(grid, 0.5, overlay, 0.5, 0)
