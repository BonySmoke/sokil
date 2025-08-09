"""
Choosing the frames to calibrate from.

Calibration needs the board seen from many different angles and distances: a
dozen near-identical views constrain the lens model no better than one does.
Two ways to get them, both from an ordinary video of someone walking the board
around in front of the camera.

`select_frames` does it automatically, keeping a frame only when the board's
pose differs enough from everything already kept. `select_frames_interactively`
hands the choice to you, which needs a display and is the fallback for footage
the detector struggles with.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .compute import CalibrationError, CheckerBoard, find_corners

logger = logging.getLogger(__name__)

# How far apart two views must be to count as different, as a fraction of the
# image diagonal averaged over the corners. Low enough to accept a modest
# change of angle, high enough to reject the board being held still.
DEFAULT_MIN_DIFFERENCE = 0.06

DEFAULT_TARGET_VIEWS = 20


def require_display(what: str = "this") -> None:
    """
    Fail early and clearly when there is no screen to draw on.

    Without this the failure surfaces from inside OpenCV, several frames into
    the run, as an error about a missing GUI backend.
    """
    probe = "__sokil_display_probe__"
    try:
        cv2.namedWindow(probe, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(probe)
    except cv2.error as error:
        raise CalibrationError(
            f"{what} needs a display, and this environment has none "
            f"(OpenCV: {error}). In a container, run the automatic selection "
            f"instead, or select frames on the host."
        ) from error


def _pose_difference(corners: np.ndarray, other: np.ndarray, diagonal: float) -> float:
    """
    How differently the board sits in two frames, as a fraction of the diagonal.

    Mean corner displacement captures every way a view can differ that matters
    here — distance, angle, position in frame — in one number. The corner order
    can come back reversed for a board seen from the other side, so both
    orderings are tried and the smaller distance wins; otherwise a mirrored
    detection of the same pose would read as a completely new view.
    """
    first = corners.reshape(-1, 2)
    second = other.reshape(-1, 2)

    forward = float(np.mean(np.linalg.norm(first - second, axis=1)))
    reversed_ = float(np.mean(np.linalg.norm(first - second[::-1], axis=1)))
    return min(forward, reversed_) / diagonal


def select_frames(
    video_path: str | Path,
    output_dir: str | Path,
    board: CheckerBoard | None = None,
    target: int = DEFAULT_TARGET_VIEWS,
    min_difference: float = DEFAULT_MIN_DIFFERENCE,
    stride: int = 5,
) -> list[Path]:
    """
    Pick a spread of board views out of a video, without supervision.

    :param target: stop once this many views have been kept.
    :param min_difference: how far a view must be from every kept view.
    :param stride: examine every Nth frame; consecutive frames of a handheld
        board are near-identical, so most are not worth decoding.
    :returns: the paths written, in the order they were found.
    """
    board = board or CheckerBoard()
    video_path, output_dir = Path(video_path), Path(output_dir)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise CalibrationError(f"Could not open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    kept: list[np.ndarray] = []
    written: list[Path] = []
    number = 0
    seen_board = 0

    logger.info(
        "Scanning %s for a %s board, every %d frames", video_path.name, board, stride
    )

    try:
        while len(kept) < target:
            if number % stride:
                if not capture.grab():
                    break
                number += 1
                continue

            success, frame = capture.read()
            if not success:
                break

            corners = find_corners(frame, board)
            if corners is None:
                number += 1
                continue

            seen_board += 1
            height, width = frame.shape[:2]
            diagonal = float(np.hypot(width, height))

            if any(
                _pose_difference(corners, previous, diagonal) < min_difference
                for previous in kept
            ):
                number += 1
                continue

            path = output_dir / f"frame_{number:06d}.jpg"
            cv2.imwrite(str(path), frame)
            kept.append(corners)
            written.append(path)
            logger.info("  kept frame %d (%d/%d)", number, len(kept), target)

            number += 1
    finally:
        capture.release()

    logger.info(
        "Kept %d of %d frames the board was found in, into %s",
        len(written),
        seen_board,
        output_dir,
    )

    if not written:
        raise CalibrationError(
            f"The board was never detected in {video_path.name}. Check that "
            f"--board-cols and --board-rows count INNER corners, and that the "
            f"whole board is visible."
        )

    if len(written) < target:
        logger.warning(
            "Only %d distinct views found, short of the %d asked for. Record "
            "the board at more angles and distances, or lower --min-difference.",
            len(written),
            target,
        )

    return written


def select_frames_interactively(
    video_path: str | Path,
    output_dir: str | Path,
    board: CheckerBoard | None = None,
) -> list[Path]:
    """
    Step through a video and save frames by hand.

    :param board: when given, detected corners are drawn, so a frame can be
        saved knowing the detector agrees the board is there.
    :returns: the paths written.
    """
    video_path, output_dir = Path(video_path), Path(output_dir)

    require_display("Interactive selection")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise CalibrationError(f"Could not open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    frame = None
    number = 0
    paused = False

    print(
        "\nControls:\n"
        "  SPACE  pause/play\n"
        "  d      next frame\n"
        "  a      previous frame\n"
        "  s      save this frame\n"
        "  q      quit\n"
    )

    try:
        while True:
            if not paused or frame is None:
                success, frame = capture.read()
                if not success:
                    break
                number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

            display = frame.copy()

            status = f"Frame {number}   saved {len(written)}"
            if board is not None:
                corners = find_corners(frame, board)
                if corners is not None:
                    cv2.drawChessboardCorners(display, board.size, corners, True)
                else:
                    status += "   (no board)"

            cv2.putText(
                display,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Frame Selector", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                paused = not paused
            elif key == ord("d"):
                paused = True
                success, frame = capture.read()
                if success:
                    number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            elif key == ord("a"):
                paused = True
                number = max(0, number - 2)
                capture.set(cv2.CAP_PROP_POS_FRAMES, number)
                success, frame = capture.read()
                if success:
                    number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            elif key == ord("s"):
                path = output_dir / f"frame_{number:06d}.jpg"
                cv2.imwrite(str(path), frame)
                written.append(path)
                logger.info("Saved %s", path)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    logger.info("Saved %d frames into %s", len(written), output_dir)
    return written
