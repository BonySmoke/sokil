import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .court import Court, CourtModel
from .util import (
    draw_dashed_line,
    find_contour_intersection,
    zoom_on_object,
)

logger = logging.getLogger(__name__)

# Below this speed (px/frame) the shuttle is treated as stationary: its velocity
# direction is dominated by detection noise and would spin, so we don't project
# the cork along it or draw a direction arrow.
MIN_MOTION_SPEED = 3.0


@dataclass
class Hit:
    """
    A detected shuttle contact, attached to the track it occurred on.

    :param position: cork pixel position at contact (the point adjudicated).
    :param is_in: in/out verdict, None until HitJudge adjudicates it.
    """

    position: tuple[int, int]
    is_in: bool | None = None


@dataclass
class ShuttleTrack:
    """
    One shuttle detection and everything derived from it — without the pixels.
    """

    # frame number in cv2 decode order (the same axis CVAT annotates on)
    number: int
    xyxy: tuple[int, int, int, int]  # top left x, y and bottom right x, y
    speed: float  # px/frame, from the Kalman filter
    angle_change: float  # degrees turned compared to the previous detection
    velocity: tuple[float, float]  # EMA-smoothed velocity x, y
    timestamp: float  # seconds into the video
    blur: float  # Laplacian variance inside the bbox; needs pixels, so the
    # tracker measures it while the frame is still decoded
    shuttle_contour: np.ndarray | None = None
    # the cork measured from the pixels during the tracking pass
    measured_cork: tuple[int, int] | None = None
    # the hit detected on this frame (None if the frame is not a hit); carries
    # the contact position and its in/out verdict
    hit: Hit | None = None
    # the color of the track this frame is a part of, and the polyline drawn
    # for it — both assigned by the renderer from the detected trajectories
    track_color: tuple[int, int, int] | None = None
    track_points: list[tuple[int, int]] | None = None

    @property
    def bbox_center(self):
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = self.xyxy
        center_x = (top_left_x + bottom_right_x) / 2
        center_y = (top_left_y + bottom_right_y) / 2

        return (int(center_x), int(center_y))

    @property
    def bbox_width(self):
        top_left_x, _, bottom_right_x, _ = self.xyxy
        width = abs(bottom_right_x - top_left_x)

        return int(width)

    @property
    def bbox_height(self):
        _, top_left_y, _, bottom_right_y = self.xyxy
        height = abs(bottom_right_y - top_left_y)

        return int(height)

    @property
    def is_moving(self):
        """
        Whether the shuttle is moving fast enough for its velocity direction to
        be meaningful. A resting shuttle jitters, spinning the direction (and the
        cork position projected along it), so callers should treat it specially.
        """
        velocity_x, velocity_y = self.velocity
        return np.hypot(velocity_x, velocity_y) >= MIN_MOTION_SPEED

    @property
    def direction_start(self):
        return self.bbox_center

    @property
    def direction_end(self):
        center_x, center_y = self.bbox_center
        velocity_x, velocity_y = self.velocity

        # Normalize and scale to a fixed large length
        norm = np.sqrt(velocity_x**2 + velocity_y**2) + 1e-9
        scale = 1000  # always long enough to cross the contour

        end_x = center_x + (velocity_x / norm) * scale
        end_y = center_y + (velocity_y / norm) * scale

        return (int(end_x), int(end_y))

    def contour_from_bbox(self):
        x1, y1, x2, y2 = self.xyxy
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

    @property
    def cork_position(self):
        """
        Get the position of the shuttlecock cork position based on the direction
        This will be the contact point during the hit
        if we cannot find the cork position, we will return the bbox center
        """
        # A stationary shuttle has no meaningful heading; projecting along the
        # noisy velocity would make the cork jump around, so anchor it
        # to the detection center instead.
        if not self.is_moving:
            return self.bbox_center

        if self.measured_cork is not None:
            return self.measured_cork

        contour = self.shuttle_contour

        if contour is None:
            logger.debug("falling back to contour from bbox")
            contour = self.contour_from_bbox()

        cork_position = find_contour_intersection(
            contour=contour,
            direction_start=self.direction_start,
            direction_end=self.direction_end,
        )

        return cork_position if cork_position else self.bbox_center


def tracks_by_number(tracks: list[ShuttleTrack]) -> dict[int, ShuttleTrack]:
    """Index tracks by frame number so a decode pass can pair them up."""
    return {track.number: track for track in tracks}


@dataclass
class Frame:
    frame: cv2.typing.MatLike
    # frame number
    number: int

    def __repr__(self):
        return f"Frame {self.number}"

    def _draw_frame_number(self, frame: cv2.typing.MatLike):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # Green color (BGR)
        font_thickness = 2
        text_position = (10, 50)  # Top-left corner (x, y)

        # Put the frame number text on the image
        cv2.putText(
            frame,
            str(self.number),
            text_position,
            font,
            font_scale,
            font_color,
            font_thickness,
        )

    def draw(self, show_frame_number: bool = False):
        frame = self.frame.copy()

        if show_frame_number:
            self._draw_frame_number(frame)

        return frame


@dataclass
class ShuttleFrame(Frame):
    """
    A decoded image paired with the track measured on it.

    Only the renderer builds these, one at a time: the pixels come from a fresh
    decode pass and the measurements from the ShuttleTrack that pass produced.
    """

    track: ShuttleTrack
    court: Court | None = None
    court_contour: np.ndarray | None = None
    court_rectangle: np.ndarray | None = None  # 4 points of the court

    def draw(
        self,
        show_speed: bool = True,
        show_angle_change: bool = True,
        show_direction: bool = True,
        show_shuttle_contour: bool = False,
        zoom_in_on_hit: bool = False,
        show_frame_number: bool = False,
        show_track: bool = False,
        show_court_contour=True,
        show_court_rectangle=False,
    ):
        """
        Modify the frame with data such as speed, angle, etc.
        """
        frame = self.frame.copy()
        track = self.track

        if show_frame_number:
            self._draw_frame_number(frame)

        top_left_x, top_left_y, bottom_right_x, bottom_right_y = track.xyxy

        cv2.rectangle(
            frame,
            (int(top_left_x), int(top_left_y)),
            (int(bottom_right_x), int(bottom_right_y)),
            (0, 255, 0),
            1,
        )

        if show_angle_change:
            cv2.putText(
                frame,
                text=f"Angle change: {track.angle_change:.2f}",
                org=(int(top_left_x), int(top_left_y) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
            )

        if show_speed:
            cv2.putText(
                frame,
                text=f"Shuttle speed: {track.speed:.2f} px",
                org=(int(top_left_x), int(top_left_y) - 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=2,
            )

        if show_direction and track.is_moving:
            cv2.arrowedLine(
                frame,
                track.direction_start,
                track.direction_end,
                (0, 255, 0),
                thickness=2,
                tipLength=0.3,
            )

        if show_shuttle_contour and track.shuttle_contour is not None:
            cv2.drawContours(frame, [track.shuttle_contour], -1, (0, 0, 255), 1)

        contact_point = track.cork_position

        contact_point = contact_point if contact_point else track.bbox_center

        cv2.circle(frame, contact_point, radius=2, color=(255, 0, 0), thickness=2)

        if track.hit:
            color = (43, 240, 164) if track.hit.is_in else (0, 0, 255)
            text = "in" if track.hit.is_in else "out"
            contact_point = track.cork_position

            contact_point = contact_point if contact_point else track.bbox_center

            cv2.circle(frame, contact_point, radius=2, color=color, thickness=2)

            text_pos_x, text_pos_y = contact_point
            cv2.putText(
                frame,
                text=text,
                org=(text_pos_x, text_pos_y - 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=color,
                thickness=2,
            )

        if self.court_contour is not None and show_court_contour:
            if isinstance(self.court, Court):
                court_model = self.court.court_model
                _, visible_edges = court_model.get_court_model()
            else:
                court_model = CourtModel()
                visible_edges = court_model.edges

            visible = {tuple(sorted(e)) for e in visible_edges}
            for start_idx, end_idx in court_model.edges:
                p1 = tuple(self.court_contour[start_idx - 1])
                p2 = tuple(self.court_contour[end_idx - 1])
                if tuple(sorted((start_idx, end_idx))) in visible:
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
                else:
                    draw_dashed_line(frame, p1, p2, (0, 200, 0), thickness=1)

        if self.court_rectangle is not None and show_court_rectangle:
            overlay = frame.copy()

            cv2.polylines(overlay, [self.court_rectangle], True, (0, 255, 0), 3)
            cv2.fillPoly(overlay, [self.court_rectangle], (0, 128, 255))
            frame = cv2.addWeighted(overlay, 0.4, frame, 1 - 0.4, 0)

        if show_track and track.track_color and track.track_points:
            points = np.hstack(track.track_points).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame, [points], isClosed=False, color=track.track_color, thickness=2
            )

        if track.hit and zoom_in_on_hit:
            bbox = (top_left_x, top_left_y, track.bbox_width, track.bbox_height)
            frame = zoom_on_object(frame, bbox, margin_ratio=10)

        return frame
