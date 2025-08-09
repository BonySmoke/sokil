import logging
from functools import lru_cache

import cv2
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from shapely import Polygon

from .court import Court
from .frame import ShuttleFrame, ShuttleTrack
from .render import hit_closeup
from .util import infinite_line_points, zoom_inset

logger = logging.getLogger(__name__)

# BGR overlay colours. The two line families keep the same hues they have in
# tools/visualize_court_solver.py, so a red line means "court-length" in both
# walkthroughs.
COURT_LENGTH_BGR = (60, 76, 231)  # court-length ("vertical") family
CROSS_COURT_BGR = (219, 152, 52)  # cross-court ("horizontal") family
PER_FRAME_BGR = (150, 150, 150)  # a single frame's lines, behind the consensus
MODEL_BGR = (80, 200, 80)  # the court model projected by the homography
BOUNDARY_BGR = (40, 220, 255)  # the court boundary under the homography
MASK_BGR = (231, 76, 231)  # the segmentation mask
SHUTTLE_BGR = (0, 255, 0)
OBSERVABLE_BGR = (255, 200, 60)
INK_BGR = (255, 255, 255)


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    A TrueType face at `size` px.

    DejaVu Sans ships with matplotlib, which is already a dependency, so this
    needs no bundled asset and no system font lookup that can come back empty.
    """
    path = font_manager.findfont(
        font_manager.FontProperties(
            family="DejaVu Sans", weight="bold" if bold else "normal"
        )
    )
    return ImageFont.truetype(path, size)


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    """Break `text` into lines that each measure under max_width in `font`."""
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if current and font.getlength(candidate) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def draw_caption(
    image: np.ndarray,
    title: str,
    caption: str = "",
    opacity: float = 0.78,
    margin_fraction: float = 0.032,
) -> np.ndarray:
    """
    Burn a lower-third title and caption onto a copy of `image`.

    The text is burned in rather than overlaid at compose time so that it
    belongs to the stage: when the walkthrough wipes from one stage to the next,
    each caption travels with the picture it describes.
    """
    height, width = image.shape[:2]
    margin = int(width * margin_fraction)
    title_font = _font(max(14, int(height * 0.040)), bold=True)
    caption_font = _font(max(12, int(height * 0.027)))

    text_width = width - 2 * margin
    caption_lines = _wrap(caption, caption_font, text_width) if caption else []

    line_height = int(caption_font.size * 1.42)
    band_height = int(
        title_font.size * 1.5 + len(caption_lines) * line_height + margin * 1.2
    )
    band_top = max(0, height - band_height)

    out = image.copy()
    band = out[band_top:, :].astype(np.float32) * (1.0 - opacity)
    out[band_top:, :] = band.astype(np.uint8)
    cv2.line(out, (0, band_top), (width, band_top), SHUTTLE_BGR, 2)

    canvas = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)

    y = band_top + int(margin * 0.55)
    draw.text((margin, y), title, font=title_font, fill=(255, 255, 255))
    y += int(title_font.size * 1.45)
    for line in caption_lines:
        draw.text((margin, y), line, font=caption_font, fill=(214, 214, 210))
        y += line_height

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------


def draw_lines(image, lines, color, thickness=2):
    """Draw infinite Hesse-form lines (rho/theta dicts) across the image."""
    for line in lines:
        start, end = infinite_line_points(line["rho"], line["theta"])
        cv2.line(image, start, end, color, thickness)
    return image


def dim(image, strength: float = 0.55):
    """Darken an image so coloured overlays drawn on top of it read clearly."""
    return (image.astype(np.float32) * (1.0 - strength)).astype(np.uint8)


def mark_point(
    image: np.ndarray,
    point: tuple[int, int],
    radius_fraction: float = 0.022,
    color: tuple[int, int, int] = SHUTTLE_BGR,
    halo: tuple[int, int, int] = (0, 0, 0),
):
    """
    Ring a point so it stays readable at video scale.

    A thin bright ring on its own is a dozen pixels across at 480p and the
    encoder smears its edge into something that reads as two rings. The dark
    halo either side gives it an edge to hold on to, and the radius follows the
    frame so the marker is the same size at any resolution.
    """
    radius = max(4, int(min(image.shape[:2]) * radius_fraction))
    cv2.circle(image, point, radius + 2, halo, 1, cv2.LINE_AA)
    cv2.circle(image, point, radius, color, 2, cv2.LINE_AA)
    cv2.circle(image, point, radius - 2, halo, 1, cv2.LINE_AA)
    return image


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def stage_raw_frame(frame: np.ndarray) -> np.ndarray:
    """1. The clip as it comes off the camera."""
    return frame.copy()


def stage_detection(
    frame: np.ndarray, court: Court | None, track: ShuttleTrack | None
) -> np.ndarray:
    """2. The court segmentation mask and the shuttle's bounding box."""
    out = frame.copy()

    contour = court.get_contour() if court is not None else None
    if contour is not None:
        overlay = out.copy()
        cv2.drawContours(overlay, [contour], -1, MASK_BGR, cv2.FILLED)
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
        cv2.drawContours(out, [contour], -1, MASK_BGR, 2)

    if track is not None:
        x1, y1, x2, y2 = (int(value) for value in track.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), SHUTTLE_BGR, 2)
        out = zoom_inset(out, track.bbox_center)

    return out


def stage_court_edges(court: Court) -> np.ndarray | None:
    """3. The frame reduced to the painted lines the solver evaluates."""
    edges = court._court_edge_image()
    if edges is None:
        return None

    # a one-pixel edge survives neither the video encoder nor the eye
    thickened = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    return cv2.cvtColor(thickened, cv2.COLOR_GRAY2BGR)


def stage_line_clusters(
    frame: np.ndarray,
    court: Court,
    consensus_verticals: list | None = None,
    consensus_horizontals: list | None = None,
) -> np.ndarray | None:
    """
    4. The edges collapsed into one line per painted stripe.

    When the solver's consensus lines are passed in, this frame's own clusters
    are drawn behind them in grey: the consensus is what the homography is
    actually solved from, and it is the subset that persisted across frames.
    """
    verticals, horizontals = court.detect_court_lines()
    if not verticals and not horizontals:
        return None

    out = dim(frame)
    has_consensus = bool(consensus_verticals or consensus_horizontals)

    if has_consensus:
        draw_lines(out, verticals, PER_FRAME_BGR, 1)
        draw_lines(out, horizontals, PER_FRAME_BGR, 1)
        draw_lines(out, consensus_horizontals or [], CROSS_COURT_BGR, 2)
        draw_lines(out, consensus_verticals or [], COURT_LENGTH_BGR, 3)
    else:
        draw_lines(out, horizontals, CROSS_COURT_BGR, 2)
        draw_lines(out, verticals, COURT_LENGTH_BGR, 3)

    return out


def stage_homography(frame: np.ndarray, court: Court) -> np.ndarray | None:
    """5. The court model projected back onto the frame by the solved homography."""
    transformer = court.get_view_transformer()
    if transformer is None:
        return None

    out = dim(frame, 0.3)
    model = court.court_model

    for x in model.stripe_center_coords("x"):
        points = transformer.inverse_transform_points(
            np.array([[x, 0], [x, model.length]], np.float32)
        ).astype(int)
        cv2.line(out, tuple(points[0]), tuple(points[1]), MODEL_BGR, 2)

    for y in model.stripe_center_coords("y"):
        points = transformer.inverse_transform_points(
            np.array([[0, y], [model.width, y]], np.float32)
        ).astype(int)
        cv2.line(out, tuple(points[0]), tuple(points[1]), MODEL_BGR, 2)

    boundary = np.array(model.doubles_boundaries(), dtype=np.float32)
    cv2.polylines(
        out,
        [transformer.inverse_transform_points(boundary).astype(np.int32)],
        True,
        BOUNDARY_BGR,
        3,
    )
    return out


def stage_observable_area(
    frame: np.ndarray,
    court: Court,
    track: ShuttleTrack | None = None,
    padding: float = 50,
) -> np.ndarray | None:
    """
    6. The area a landing is allowed to be claimed in.

    This is the plausibility gate from Court.shuttle_inside_max_observable_area:
    the full court plus a margin, projected into the image. A direction change
    outside it belongs to another court or happened too high to be a landing.
    """
    transformer = court.get_view_transformer()
    if transformer is None:
        return None

    region = Polygon(court.court_model.doubles_boundaries()).buffer(padding)
    outline = transformer.inverse_transform_points(
        np.array(region.exterior.coords, dtype=np.float32)
    ).astype(np.int32)

    boundary = transformer.inverse_transform_points(
        np.array(court.court_model.doubles_boundaries(), dtype=np.float32)
    ).astype(np.int32)

    out = frame.copy()
    overlay = out.copy()
    cv2.fillPoly(overlay, [outline], OBSERVABLE_BGR)
    out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0)
    cv2.polylines(out, [outline], True, OBSERVABLE_BGR, 3)
    # the court itself, so the margin the region adds is visible as a margin
    cv2.polylines(out, [boundary], True, BOUNDARY_BGR, 2)

    if track is not None:
        mark_point(out, track.cork_position)

    return out


def stage_hit_point(
    frame: np.ndarray, court: Court | None, track: ShuttleTrack
) -> np.ndarray:
    """7. The contact frame: the flight, the cork position and the verdict."""
    court_contour = court.get_court_projection_points() if court is not None else None

    out = ShuttleFrame(
        frame=frame,
        number=track.number,
        track=track,
        court=court,
        court_contour=court_contour,
    ).draw(
        show_speed=False,
        show_angle_change=False,
        show_direction=False,
        show_shuttle_contour=True,
        show_track=True,
    )

    return zoom_inset(out, track.cork_position)


def stage_top_view(
    court: Court,
    track: ShuttleTrack,
    width: int,
    height: int,
) -> np.ndarray | None:
    """
    8. The landing seen from above: the umpire close-up the review video holds.

    A tight top-down window around the contact showing the line that decides the
    call, the shuttle at true scale and the margin in centimetres. The whole
    court would put the landing somewhere on the court, but at video resolution
    the few centimetres that decide the call are a pixel or two.
    """
    return hit_closeup(court=court, shuttle_track=track, width=width, height=height)
