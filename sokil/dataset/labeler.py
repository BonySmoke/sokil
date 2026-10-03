"""
TODO: make the labeling interface more user-friendly. Maybe research a small library that can help with this.
Labelling landings on a clip.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import schema
from .frames import VideoFrames
from .landing import derive_call, derive_line, image_to_court
from .schema import Call

logger = logging.getLogger(__name__)

# How often the window redraws while paused. A click has to appear without
# waiting for a keypress, and the loop is cheap — a cached frame and an overlay.
PAUSED_POLL_MS = 20

KEYS = """\
space play/pause   n/p frame   l/j +/-10   L/J +/-100
f     mark a landing on this frame
click place where the shuttle landed
i/o/u call it in / out / unknown
t     switch singles <-> doubles
r     remove this frame's landing
s     save        q save and quit"""


@dataclass
class Landing:
    """One landing, as the annotator has it so far."""

    frame: int
    call: str = Call.UNKNOWN.value
    x_px: float | None = None
    y_px: float | None = None


@dataclass
class Session:
    """
    The labelling state for one clip, with no screen attached.

    Kept apart from the window so that what a keypress does to the data can be
    tested without one.
    """

    clip_id: str
    court_type: str = "doubles"
    annotator: str = field(default_factory=lambda: os.environ.get("USER", "unknown"))
    landings: dict[int, Landing] = field(default_factory=dict)

    def mark(self, frame: int) -> Landing:
        return self.landings.setdefault(frame, Landing(frame=frame))

    def place(self, frame: int, x_px: float, y_px: float) -> Landing:
        landing = self.mark(frame)
        landing.x_px, landing.y_px = float(x_px), float(y_px)
        return landing

    def call(self, frame: int, call: str) -> Landing:
        landing = self.mark(frame)
        landing.call = call
        return landing

    def remove(self, frame: int) -> None:
        self.landings.pop(frame, None)

    def toggle_court_type(self) -> str:
        self.court_type = "singles" if self.court_type == "doubles" else "doubles"
        return self.court_type


def event_rows(session: Session, to_court=None) -> list[dict]:
    """
    The events.csv rows for a session's landings.

    Court metres and the line are derived here rather than asked for: both
    follow from the click once the court is known, and a number the annotator
    would have to estimate is a number that disagrees with itself.
    """
    rows = []

    for frame in sorted(session.landings):
        landing = session.landings[frame]
        x_m = y_m = None
        line = None

        if to_court is not None and landing.x_px is not None:
            court = to_court(landing.x_px, landing.y_px)
            if court is not None:
                x_m, y_m = court
                line = derive_line(x_m, y_m, session.court_type)

        rows.append(
            {
                "clip_id": session.clip_id,
                "frame": frame,
                "call": landing.call,
                "court_type": session.court_type,
                "line": line,
                "x_px": landing.x_px,
                "y_px": landing.y_px,
                "x_m": x_m,
                "y_m": y_m,
                "annotator": session.annotator,
                "notes": None,
            }
        )

    return rows


def merge_events(
    existing: pd.DataFrame, clip_id: str, rows: list[dict]
) -> pd.DataFrame:
    """
    Replace one clip's events with a freshly labelled set, leaving other clips
    alone: a labelling session is the last word on its own clip and says nothing
    about any other.
    """
    kept = existing[existing["clip_id"] != clip_id]
    fresh = pd.DataFrame(rows, columns=schema.columns("events"))
    return pd.concat([kept, fresh], ignore_index=True)


def load_session(
    events: pd.DataFrame, clip_id: str, annotator: str | None = None
) -> Session:
    """Resume from whatever this clip already has in events.csv."""
    session = Session(clip_id=clip_id)
    if annotator:
        session.annotator = annotator

    for event in events[events["clip_id"] == clip_id].itertuples():
        session.landings[int(event.frame)] = Landing(
            frame=int(event.frame),
            call=event.call,
            x_px=None if pd.isna(event.x_px) else float(event.x_px),
            y_px=None if pd.isna(event.y_px) else float(event.y_px),
        )
        if not pd.isna(event.court_type):
            session.court_type = event.court_type

    if session.landings:
        logger.info(
            "Resuming %s with %d landing(s) already labelled",
            clip_id,
            len(session.landings),
        )
    return session


def scale_point(
    x: float,
    y: float,
    shown: tuple[int, int],
    actual: tuple[int, int],
) -> tuple[float, float] | None:
    """
    A click in a resized window, as a pixel of the frame itself.

    The window is resizable, and OpenCV reports where the pointer was in the
    window rather than in the image. Left unscaled, a click on an enlarged
    window lands off the frame, so the landing is stored somewhere the shuttle
    could not be and nothing is drawn where the click was.

    Returns None for a click outside the image.
    """
    shown_w, shown_h = shown
    actual_w, actual_h = actual

    if shown_w <= 0 or shown_h <= 0:
        return None

    x_image = x * actual_w / shown_w
    y_image = y * actual_h / shown_h

    if not (0 <= x_image <= actual_w - 1 and 0 <= y_image <= actual_h - 1):
        return None

    return x_image, y_image


def _to_image_point(window: str, x: float, y: float, width: int, height: int):
    """Where a click in the window falls on the frame, or None if off it."""
    import cv2

    try:
        _, _, shown_w, shown_h = cv2.getWindowImageRect(window)
    except cv2.error:
        # No rect to be had (some backends); the window is then its own size.
        shown_w, shown_h = width, height

    return scale_point(x, y, (shown_w, shown_h), (width, height))


def _overlay(image, lines: list[tuple[str, tuple[int, int, int]]]):
    import cv2

    for row, (text, colour) in enumerate(lines):
        cv2.putText(
            image,
            text,
            (10, 30 + row * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
        )


def _status(session: Session, landing: Landing | None, to_court) -> list:
    """What the window shows about the frame in view."""
    white, grey, green, red, amber = (
        (255, 255, 255),
        (180, 180, 180),
        (0, 220, 0),
        (0, 0, 255),
        (0, 200, 255),
    )

    if landing is None:
        return [("no landing on this frame  -  f to mark one", grey)]

    called = landing.call in (Call.IN.value, Call.OUT.value)
    lines = [
        (
            f"LANDING  call: {landing.call.upper()}"
            + ("" if called else "   <-- press i / o / u"),
            white if called else amber,
        )
    ]

    if landing.x_px is None:
        lines.append(("click where the shuttle landed", amber))
        return lines

    court = to_court(landing.x_px, landing.y_px) if to_court else None
    if court is None:
        lines.append(("no court keypoints: position cannot be derived", amber))
        return lines

    x_m, y_m = court
    suggested = derive_call(x_m, y_m, session.court_type)
    line = derive_line(x_m, y_m, session.court_type)
    lines.append((f"at {x_m:.2f} m, {y_m:.2f} m  ({line})", grey))

    # Shown, never stored: the call is the annotator's. Only a call that was
    # actually made can disagree with the geometry — unknown is a verdict the
    # footage does not support, not a conflict with one — so an uncalled
    # landing gets the geometry's opinion and nothing more.
    if not called:
        lines.append((f"geometry says: {suggested.upper()}", grey))
        return lines

    agrees = suggested == landing.call
    lines.append(
        (
            f"geometry says: {suggested.upper()}"
            + ("" if agrees else "   <-- disagrees with the call"),
            green if agrees else red,
        )
    )
    return lines


def label_clip(
    video_path: Path,
    session: Session,
    court_points: pd.DataFrame | None = None,
    play_fps: float = 30.0,
    on_save=None,
) -> Session:
    """
    Run the labelling window over a clip; returns the session.

    Only the loop lives here: every keypress hands off to Session, which is
    where the rules are and where they are tested.

    :param on_save: called with the session on s and on quit, so a long
        labelling run is not one crash away from being lost.
    """
    import cv2

    clip = VideoFrames(video_path)

    to_court = None
    if court_points is not None:
        to_court = image_to_court(court_points, session.clip_id, 0)

    index = 0
    playing = False
    window = f"sokil - {session.clip_id}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        point = _to_image_point(window, x, y, clip.width, clip.height)
        if point is None:
            logger.warning("Click at (%d, %d) is outside the frame", x, y)
            return

        session.place(index, *point)
        logger.info(
            "Landing at frame %d placed at (%.0f, %.0f)", index, point[0], point[1]
        )

    cv2.setMouseCallback(window, on_mouse)
    print(KEYS)

    delay = max(1, int(1000 / play_fps))

    while True:
        display = clip[index]
        if display is None:
            # Past the real end of the clip, whatever the container claimed.
            index = max(index - 1, 0)
            playing = False
            continue

        display = display.copy()
        frame = index
        landing = session.landings.get(frame)

        if landing is not None and landing.x_px is not None:
            point = (int(landing.x_px), int(landing.y_px))
            cv2.drawMarker(display, point, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(display, point, 12, (0, 0, 255), 1)

        heading = (
            f"frame {frame}  ({index + 1}/{len(clip)})  "
            f"{'PLAYING' if playing else 'PAUSED'}  "
            f"{session.court_type}  landings: {len(session.landings)}"
        )
        _overlay(
            display,
            [(heading, (0, 220, 0)), *_status(session, landing, to_court)],
        )

        cv2.imshow(window, display)
        # Paused, this polls rather than blocks: a mouse click does not wake
        # waitKey, so a blocking wait would take the click and show nothing
        # until the next keypress.
        key = cv2.waitKey(delay if playing else PAUSED_POLL_MS) & 0xFF

        if key in (255, 0):
            if playing:
                if index < len(clip) - 1:
                    index += 1
                else:
                    playing = False
            continue

        if key == ord(" "):
            playing = not playing
        elif key == ord("n"):
            index = min(index + 1, len(clip) - 1)
        elif key == ord("p"):
            index = max(index - 1, 0)
        elif key in (83, ord("l")):
            index = min(index + 10, len(clip) - 1)
        elif key in (81, ord("j")):
            index = max(index - 10, 0)
        elif key == ord("L"):
            index = min(index + 100, len(clip) - 1)
        elif key == ord("J"):
            index = max(index - 100, 0)
        elif key == ord("f"):
            session.mark(frame)
        elif key == ord("i"):
            session.call(frame, Call.IN.value)
        elif key == ord("o"):
            session.call(frame, Call.OUT.value)
        elif key == ord("u"):
            session.call(frame, Call.UNKNOWN.value)
        elif key == ord("t"):
            session.toggle_court_type()
        elif key == ord("r"):
            session.remove(frame)
        elif key in (ord("s"), ord("q")):
            if on_save is not None:
                on_save(session)
            if key == ord("q"):
                break

    clip.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return session
