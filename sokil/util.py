import logging
from collections import defaultdict
from itertools import combinations
from math import sqrt

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN, SpectralClustering

logger = logging.getLogger(__name__)


def draw_dashed_line(img, p1, p2, color, thickness=1, dash_len=12, gap_len=8):
    """
    Draw a dashed line, clipped to the image. Endpoints may lie far outside the
    frame (e.g. court keypoints projected beyond the camera view) — clipping
    first keeps the dash walk bounded.
    """
    h, w = img.shape[:2]
    inside, q1, q2 = cv2.clipLine(
        (0, 0, w, h), tuple(map(int, p1)), tuple(map(int, p2))
    )
    if not inside:
        return
    q1, q2 = np.array(q1, dtype=np.float64), np.array(q2, dtype=np.float64)
    length = float(np.hypot(*(q2 - q1)))
    if length < 1:
        return
    direction = (q2 - q1) / length
    pos = 0.0
    while pos < length:
        start = q1 + direction * pos
        end = q1 + direction * min(pos + dash_len, length)
        cv2.line(img, tuple(np.int32(start)), tuple(np.int32(end)), color, thickness)
        pos += dash_len + gap_len


def infinite_line_points(rho: float, theta: float, span: int = 3000):
    """
    Two far-apart points on the infinite line rho = x·cos(theta) + y·sin(theta).

    Detected court lines are stored in Hesse normal form, which has no
    endpoints; drawing one means picking a segment long enough to cross the
    frame, which is what span is for.
    """
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    return (
        (int(x0 + span * -b), int(y0 + span * a)),
        (int(x0 - span * -b), int(y0 - span * a)),
    )


def zoom_inset(
    image: np.ndarray,
    center: tuple[int, int],
    crop_fraction: float = 0.11,
    inset_fraction: float = 0.26,
    color: tuple[int, int, int] = (255, 255, 255),
):
    """
    Paste a magnified crop around `center` into the top-right corner.

    On a full HD frame the shuttle is a handful of pixels across, so a bounding
    box drawn around it is invisible at video scale. The inset shows the same
    pixels large enough to see, with a connector back to where they came from.

    :param crop_fraction: side of the crop, as a share of the frame's short
        side — so the same call frames the shuttle the same way at any
        resolution.
    """
    height, width = image.shape[:2]
    crop_size = max(24, int(min(height, width) * crop_fraction))
    half = crop_size // 2
    center_x, center_y = int(center[0]), int(center[1])

    x1 = int(np.clip(center_x - half, 0, max(0, width - crop_size)))
    y1 = int(np.clip(center_y - half, 0, max(0, height - crop_size)))
    x2, y2 = min(width, x1 + crop_size), min(height, y1 + crop_size)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image

    side = int(min(height, width) * inset_fraction)
    magnified = cv2.resize(crop, (side, side), interpolation=cv2.INTER_NEAREST)

    margin = int(width * 0.03)
    inset_x, inset_y = width - side - margin, margin

    out = image.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.line(out, (x2, y1), (inset_x, inset_y + side), color, 1)
    out[inset_y : inset_y + side, inset_x : inset_x + side] = magnified
    cv2.rectangle(out, (inset_x, inset_y), (inset_x + side, inset_y + side), color, 2)
    return out


def zoom_on_object(frame: cv2.typing.MatLike, bbox: tuple, margin_ratio=0.5):
    """
    Zoom in on a region around the object with optional surrounding margin.

    Args:
        frame: The original image (numpy array).
        bbox: Tuple (x, y, w, h) — bounding box.
        margin_ratio: Fraction of bbox size to include as margin (e.g., 0.5 = 50%).

    Returns:
        zoomed_frame: The zoomed-in image, resized to original frame size.
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox

    # Compute margins
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    # Compute expanded crop box (clamp to image bounds)
    x1 = max(x - mx, 0)
    y1 = max(y - my, 0)
    x2 = min(x + bw + mx, w)
    y2 = min(y + bh + my, h)

    # Crop and resize
    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return zoomed


def cross_2d(a, b) -> float:
    """
    Z-component of the cross product of two 2D vectors.
    (np.cross dropped 2D support in NumPy 2.0.)
    """
    return float(a[0] * b[1] - a[1] * b[0])


def line_intersection(p1, p2, q1, q2):
    """Returns the intersection point of line segments p1-p2 and q1-q2, if it exists."""
    p = np.array(p1, dtype=np.float32)
    r = np.array(p2, dtype=np.float32) - p
    q = np.array(q1, dtype=np.float32)
    s = np.array(q2, dtype=np.float32) - q

    r_cross_s = cross_2d(r, s)

    if r_cross_s == 0:
        return None  # Lines are parallel

    q_minus_p = q - p
    qmp_cross_r = cross_2d(q_minus_p, r)

    t = (
        cross_2d(q_minus_p, s) / r_cross_s
    )  # how far along line 1 the intersection point lies
    u = qmp_cross_r / r_cross_s  # how far along line 2 it lies

    # 0 is the start of the line, 1 is the end of it
    if 0 <= t <= 1 and 0 <= u <= 1:
        return tuple(p + t * r)  # Intersection point

    return None  # No intersection within segment bounds


def find_contour_intersection(contour, direction_start, direction_end):
    """
    Find the intersection of the contour with the direction line
    """
    points = contour.reshape(-1, 2)
    n = len(points)

    intersections = []

    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]

        intersection = line_intersection(direction_start, direction_end, p1, p2)
        if intersection is not None:
            intersections.append(intersection)

    if not intersections:
        return None

    # If multiple intersections, return the one closest to direction_end
    # (assuming direction points toward cork)
    intersections = np.array(intersections)
    distances = np.linalg.norm(intersections - direction_end, axis=1)
    closest_idx = np.argmin(distances)

    return tuple(map(int, intersections[closest_idx]))


def signed_angle_change(p1, p2, p3):
    """
    Calculate signed angle in degrees between two directed lines: p1→p2 and p2→p3.
    Positive = counterclockwise, Negative = clockwise
    """
    # Vector AB and BC
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)

    # No displacement on one side (e.g. a stationary/duplicate point) leaves the
    # angle undefined; report no change rather than dividing by zero.
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0

    # Normalize vectors
    v1_norm = v1 / n1
    v2_norm = v2 / n2

    # Compute angle and direction (cross product gives sign in 2D)
    angle_rad = np.arctan2(
        v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0],  # sin(θ) = cross product
        np.dot(v1_norm, v2_norm),  # cos(θ) = dot product
    )

    angle_deg = np.degrees(angle_rad)
    return angle_deg


def speed_estimation(
    track_history: list, box_areas: list, reference_area: float, fps: float
) -> float:
    """
    Estimate the speed of the object given the track history with its center location
    """
    if len(track_history) < 2 or len(box_areas) < 2:
        return 0.0
    meter_per_pixel = 0.001

    p0, p1 = track_history[0], track_history[-1]  # First and last points of track
    a0, a1 = box_areas[0], box_areas[-1]

    dt = len(track_history) / fps  # Time in seconds
    if dt == 0:
        return 0

    dx, dy = p1[0] - p0[0], p1[1] - p0[1]  # Pixel displacement
    pixel_distance = sqrt(dx * dx + dy * dy)  # Calculate pixel distance

    scale0 = reference_area / a0
    scale1 = reference_area / a1
    avg_scale = (scale0 + scale1) / 2

    adjusted_mpp = meter_per_pixel * avg_scale
    distance_meters = pixel_distance * adjusted_mpp
    speed_kmh = (distance_meters / dt) * 3.6

    return speed_kmh


def is_significant_speed_change(speed_history: list, threshold: int = 3) -> bool:
    """
    Given N speeds of frames, detect whether the last frame is an outlier which may indicate the hit
    """

    if len(speed_history) < 3:
        return False

    last_speed_record = speed_history[-1]
    reference_speed_records = speed_history[:-1]

    mean = np.mean(reference_speed_records)
    std = np.std(reference_speed_records)

    if std == 0:
        return False

    z_score = abs(last_speed_record - mean) / std
    return bool(z_score > threshold)


def is_speed_decreasing(speed_history: list, last_only: bool = True) -> bool:
    """
    Given N frames, find if the speed of the last starts to decrease
    :param last_only: if True should be returned ONLY if the speed of the last frame starts to decrease
        and the rest of the frames show speed increase or no change
    """
    if len(speed_history) < 3:
        return False

    if last_only:
        last_frame = speed_history[-1]
        context_frames = speed_history[:-1]
        for i in range(1, len(context_frames)):
            if context_frames[i - 1] > context_frames[i]:
                return False

        return context_frames[-1] > last_frame

    for i in range(1, len(speed_history)):
        if speed_history[i - 1] > speed_history[i]:
            return True

    return False


def is_significant_direction_change(
    query: float, candidates: list, threshold: int = 20
):
    """
    Check
    """
    changed_significantly = False

    for candidate in candidates:
        if abs(query - candidate) > threshold:
            changed_significantly = True

    return changed_significantly


def extract_contours(frame, background_subtractor):
    """
    Extract contours from the frame given the background subtractor
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    fg_mask = background_subtractor.apply(blur_gray)

    # decrease noise with thresholding
    _, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

    # erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # apply erosion
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def laplacian_blur_metric(image, bbox):
    """
    Calculate the amount of blur inside a bounding box on an image
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    # clamp: detections can touch (or slightly exceed) the frame edge, and an
    # empty ROI would blow up in cvtColor
    x1, x2 = max(0, int(x1)), min(width, int(x2))
    y1, y2 = max(0, int(y1)), min(height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = image[y1:y2, x1:x2]  # crop ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def segment_by_angle_spectralclustering(lines):
    """
    Robustly split Hough lines into horizontal and vertical clusters.
    Automatically falls back when spectral clustering is ill-posed.
    """
    n = len(lines)

    if n < 2:
        return [lines]

    angles = np.array([line[0][1] for line in lines])

    # ---- FAST deterministic fallback (small N) ----
    if n < 6:
        horizontal, vertical = [], []
        for line in lines:
            theta = line[0][1]
            if abs(theta - np.pi / 2) < np.pi / 4:
                horizontal.append(line)
            else:
                vertical.append(line)
        return [horizontal, vertical]

    # ---- Spectral clustering (safe region) ----
    # multiply the angles by two and find coordinates of that angle
    pts = np.column_stack([np.cos(2 * angles), np.sin(2 * angles)])

    n_samples = len(pts)
    n_neighbors = min(5, n_samples - 1)

    clustering = SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors", n_neighbors=n_neighbors
    )

    labels = clustering.fit_predict(pts)

    clusters = defaultdict(list)
    for i, line in enumerate(lines):
        clusters[labels[i]].append(line)
    clusters = list(clusters.values())

    return clusters


def intersection(line1, line2, angle_thresh=1e-3):
    """
    Finds the intersection point of two lines given in Hesse normal form (rho, theta).

    Returns:
        [[x, y]] if lines intersect within angle tolerance,
        None if lines are parallel or nearly parallel.
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    # Check if lines are parallel or nearly parallel
    if abs(np.sin(theta1 - theta2)) < angle_thresh:
        return None

    # Solve for intersection
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])

    b = np.array([rho1, rho2])  # shape (2,) instead of (2, 1)
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    return [[x0, y0]]


def segmented_intersections(lines):
    """
    Finds intersections between two groups of lines.
    Expects `lines` to be a list of 2 groups: [vertical_lines, horizontal_lines].
    """
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1 :]:
            for line1 in group:
                for line2 in next_group:
                    pt = intersection(line1, line2)
                    if pt is not None:
                        intersections.append(pt)
    return intersections


# Lines within this angle of each other count as the same direction. Defined
# here rather than in the signature so it is evaluated once, at import.
OUTER_LINE_ANGLE_THRESH = np.deg2rad(3)


def filter_outer_lines(lines, angle_thresh=OUTER_LINE_ANGLE_THRESH, rho_thresh=10):
    if lines is None:
        return None

    filtered = []
    used = set()

    # OpenCV's HoughLines returns lines sorted by the number of votes (strongest first).
    # We iterate through them and for each one, group similar lines and pick the strongest.
    for i, (rho_i, theta_i) in enumerate(lines[:, 0]):
        if i in used:
            continue

        # Since lines are sorted by votes, the first one we encounter for a new group
        # is the strongest representative for that physical court line.
        filtered.append([rho_i, theta_i])

        # Mark all similar lines as used so we don't process them as pivot lines
        for j, (rho_j, theta_j) in enumerate(lines[:, 0]):
            if j == i or j in used:
                continue
            if (
                abs(theta_i - theta_j) < angle_thresh
                and abs(rho_i - rho_j) < rho_thresh
            ):
                used.add(j)

    return np.array(filtered).reshape(-1, 1, 2)


def get_intersection_centers(intersections: list):
    """
    Given a list of line intersections, find their centers
    """
    if not intersections:
        return None
    intersection_points = np.array(intersections, dtype=np.float32).reshape(-1, 2)

    db = DBSCAN(eps=10, min_samples=1).fit(intersection_points)
    labels = db.labels_
    intersection_centers = np.array(
        [intersection_points[labels == i].mean(axis=0) for i in set(labels) if i != -1]
    )

    return intersection_centers


def classify_line_clusters(c1, c2):
    """
    Classify lines in the format (vertical, horizontal)
    """
    t1 = np.mean([l[0][1] for l in c1])
    t2 = np.mean([l[0][1] for l in c2])

    # abs(theta - pi/2) is ~0 for horizontal lines and ~pi/2 for vertical lines (theta ~0 or ~pi).
    # Therefore, the cluster with the larger distance from pi/2 is the vertical one.
    if abs(t1 - np.pi / 2) > abs(t2 - np.pi / 2):
        return c1, c2  # vertical, horizontal
    else:
        return c2, c1


def get_outermost_lines_cluster(lines, img_shape):
    """
    Finds the outermost lines in a cluster (horizontal or vertical)
    based on image coordinates, not rho.

    lines: list of np.array([[rho, theta]])
    img_shape: (height, width)

    Returns: (min_line, max_line)
    """
    if lines is None or len(lines) < 2:
        return None, None

    H, W = img_shape[:2]

    outer_values = []

    for line in lines:
        rho, theta = line[0]

        # Decide whether line is more vertical or horizontal
        if np.sin(theta) > np.cos(theta):  # horizontal-ish
            x = W / 2
            y = (rho - np.cos(theta) * x) / np.sin(theta)
            outer_values.append(y)
        else:  # vertical-ish
            y = H / 2
            x = (rho - np.sin(theta) * y) / np.cos(theta)
            outer_values.append(x)

    outer_values = np.array(outer_values)
    min_i = np.argmin(outer_values)
    max_i = np.argmax(outer_values)

    return lines[min_i], lines[max_i]


def get_court_rectangle(
    img_shape, horizontal_cluster, vertical_cluster
) -> np.ndarray | None:
    """
    Get 4 corners of a badminton court
    """
    if len(horizontal_cluster) < 2 or len(vertical_cluster) < 2:
        return None  # court not detectable this frame

    # 1. Find the two outermost lines in each group
    top_line, bottom_line = get_outermost_lines_cluster(horizontal_cluster, img_shape)
    left_line, right_line = get_outermost_lines_cluster(vertical_cluster, img_shape)

    if any(line is None for line in [top_line, bottom_line, left_line, right_line]):
        return None

    # 2. Compute intersections = rectangle corners
    pts = [
        intersection(top_line, left_line),
        intersection(top_line, right_line),
        intersection(bottom_line, right_line),
        intersection(bottom_line, left_line),
    ]

    if None in pts:
        return None

    return np.array(pts, dtype=np.int32)


def snap_to_intersection(source: list, target: list, max_radius=30):
    """
    Given 2 lists of points, snap source points to target if possible
    """
    target_pts = np.array(target, dtype=np.float32).reshape(-1, 2)  # (M,2)
    tree = KDTree(target_pts)

    source_pts = np.array(source)[:, 1:3].astype(np.float32)
    preds = np.atleast_2d(source_pts)
    snapped = []

    # Keep track of which intersections are already used
    used_mask = np.zeros(len(target_pts), dtype=bool)

    for i, pred in enumerate(preds):
        label, _, _, conf = source[i]
        dists, idxs = tree.query(
            pred, k=len(target_pts), distance_upper_bound=max_radius
        )
        # A single target means k=1, for which query returns scalars rather
        # than arrays; the zip below needs something iterable either way.
        dists, idxs = np.atleast_1d(dists), np.atleast_1d(idxs)
        valid = [
            (d, idx)
            for d, idx in zip(dists, idxs)
            if np.isfinite(d) and not used_mask[idx]
        ]

        if valid:  # only if within radius
            _, idx_min = min(valid, key=lambda x: x[0])
            used_mask[idx_min] = True
            snapped_point = target_pts[idx_min]

            snapped.append(
                (
                    label,
                    np.float32(snapped_point[0]),
                    np.float32(snapped_point[1]),
                    conf,
                )
            )
        else:
            snapped.append(source[i])

    return np.array(snapped, dtype=object)


def fig_to_numpy(fig):
    """
    Converts a Matplotlib figure to a NumPy array.
    """
    # Draw the canvas
    fig.canvas.draw()
    # Get the RGBA pixel data as a NumPy array
    buf = fig.canvas.buffer_rgba()
    # Convert to NumPy array
    x = np.asarray(buf, copy=True)
    plt.close(fig)

    return x


class Undistorter:
    """
    Undistorts a stream of same-sized frames efficiently.

    cv2.undistort rebuilds its remap tables on every call, which is wasteful
    for video: the tables depend only on the camera parameters and the frame
    size. This class builds the tables once, on the first frame, and applies
    the cheap cv2.remap for every frame after that.

    Output frames keep the input size: alpha=0 zooms the undistorted view just
    enough that every output pixel is valid, so no black borders appear and no
    crop is needed (downstream consumers like VideoWriter depend on a stable
    frame size). The cost is a thin band at the extreme edges of the field of
    view being scaled out.
    """

    def __init__(self, K: np.ndarray, dist: np.ndarray):
        self.K = K
        self.dist = dist
        self.map_x = None
        self.map_y = None

    def _build_maps(self, width: int, height: int):
        K_new, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, (width, height), alpha=0
        )
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, K_new, (width, height), cv2.CV_16SC2
        )

    def __call__(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if self.map_x is None:
            height, width = image.shape[:2]
            self._build_maps(width, height)
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)


def undistort(image: cv2.typing.MatLike, K: np.array, dist: np.array):
    """Undistort a single image, preserving its size (see Undistorter)."""
    return Undistorter(K, dist)(image)


# ---------------------------------------------------------------------------
# Geometric (keypoint-free) court line detection
#
# These helpers detect straight court lines directly from the edge image and
# match them to the known court model, so homography can be solved without the
# YOLO keypoint model. Lines are represented as dicts:
#   {"rho": float, "theta": float, "length": float, "ref": float,
#    "points": [(x1, y1), (x2, y2)]}
# where (rho, theta) is the Hesse normal form (theta in [0, pi), rho signed) so
# the existing `intersection()` keeps working.
# ---------------------------------------------------------------------------


def segment_to_hesse(x1, y1, x2, y2):
    """Convert a line segment to Hesse normal form (rho, theta), theta in [0, pi)."""
    dx, dy = x2 - x1, y2 - y1
    # theta is the angle of the line normal: perpendicular to the segment direction
    theta = (np.arctan2(dy, dx) + np.pi / 2) % np.pi
    rho = x1 * np.cos(theta) + y1 * np.sin(theta)
    return float(rho), float(theta)


def detect_line_segments(
    edges, threshold=80, min_line_length_ratio=0.2, max_line_gap=30
):
    """
    Detect straight line SEGMENTS in an edge image using the probabilistic
    Hough transform. Unlike infinite cv2.HoughLines, segments let us reject
    bent lines later by testing collinearity of their supporting points.
    """
    if edges is None or not edges.any():
        return []

    h, w = edges.shape[:2]
    min_line_length = int(min_line_length_ratio * min(h, w))

    segments = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if segments is None:
        return []

    # OpenCV 4 returns (N, 1, 4), OpenCV 5 returns (N, 4).
    lines = []
    for seg in np.asarray(segments).reshape(-1, 4):
        x1, y1, x2, y2 = map(float, seg)
        rho, theta = segment_to_hesse(x1, y1, x2, y2)
        lines.append(
            {
                "rho": rho,
                "theta": theta,
                "length": float(np.hypot(x2 - x1, y2 - y1)),
                "points": [(x1, y1), (x2, y2)],
            }
        )
    return lines


def line_reference_coord(rho, theta, img_shape, vertical):
    """
    Monotonic 1D coordinate used to order/cluster parallel lines:
    a vertical line's x at y=H/2, or a horizontal line's y at x=W/2.
    """
    h, w = img_shape[:2]
    if vertical:
        return (rho - (h / 2.0) * np.sin(theta)) / (np.cos(theta) + 1e-9)
    return (rho - (w / 2.0) * np.cos(theta)) / (np.sin(theta) + 1e-9)


def split_line_families(lines):
    """
    Split lines into (vertical, horizontal) families by orientation.
    A line is vertical-ish when its normal is near-horizontal (theta near 0/pi).
    """
    vertical, horizontal = [], []
    for line in lines:
        theta = line["theta"]
        if min(theta, np.pi - theta) < np.pi / 4:
            vertical.append(line)
        else:
            horizontal.append(line)
    return vertical, horizontal


def line_paint_contrast(line, gray, half_width=7, background_width=28, samples=64):
    """
    How much brighter the line is than the floor beside it.

    Court lines are painted white; a mat seam or a panel join is a step in an
    otherwise uniform floor. Both produce Canny edges, so the only thing that
    separates them is whether there is paint underneath. Sampling runs along the
    line's own normal, so it is correct at any orientation.

    :returns: (median peak brightness, median peak minus local background), or
        (None, None) when the line does not cross the image.
    """
    height, width = gray.shape[:2]
    normal = np.array([np.cos(line["theta"]), np.sin(line["theta"])])
    along = np.array([-normal[1], normal[0]])
    origin = line["rho"] * normal

    span = float(np.hypot(height, width))
    near = np.arange(-half_width, half_width + 1)
    far = np.arange(-background_width, background_width + 1)

    peaks, backgrounds = [], []
    for step in np.linspace(-span / 2, span / 2, samples):
        centre = origin + step * along
        for offsets, sink in ((near, peaks), (far, backgrounds)):
            points = centre + offsets[:, None] * normal
            xs = np.rint(points[:, 0]).astype(int)
            ys = np.rint(points[:, 1]).astype(int)
            inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if not inside.any():
                continue
            values = gray[ys[inside], xs[inside]]
            sink.append(values.max() if sink is peaks else np.median(values))

    if not peaks:
        return None, None

    peak = float(np.median(peaks))
    return peak, peak - float(np.median(backgrounds or [0.0]))


def keep_painted_lines(lines, gray, min_contrast=18.0, relative=0.45):
    """
    Drop clustered lines that have no paint under them.

    A line is kept when its paint contrast clears both an absolute floor and a
    fraction of the best contrast in the family. The relative test carries the
    decision — lighting varies between venues, while paint and floor within one
    frame do not — and the floor stops a family made only of seams from
    promoting its own best seam.
    """
    if not lines:
        return lines

    scored = []
    for line in lines:
        _, contrast = line_paint_contrast(line, gray)
        scored.append((line, -1.0 if contrast is None else contrast))

    best = max(contrast for _, contrast in scored)
    threshold = max(min_contrast, relative * best)

    kept = []
    for line, contrast in scored:
        if contrast >= threshold:
            kept.append(line)
        else:
            logger.debug(
                "Dropping line at %.1f: paint contrast %.1f below %.1f "
                "(no paint under it)",
                line["ref"],
                contrast,
                threshold,
            )
    return kept


def cluster_parallel_lines(
    lines, image, vertical, eps=12, straightness_tau=2.5, stripe_merge_eps=25
):
    """
    Cluster a family of parallel lines by perpendicular distance so each cluster
    is one physical court line, then fit a single STRAIGHT line per cluster.

    Collinear segments of the same painted line collapse into one representative;
    distinct lines (e.g. the doubles and singles sidelines) stay separate so inner
    boundaries are retained. Clusters whose supporting points are not straight
    (bent lines) are rejected rather than accepted as "most outer".

    Canny yields TWO edges per painted stripe (its left and right border), and
    near the camera the stripe is wider than `eps`, so a physical line often
    survives as a close pair of clusters. A final pass merges near-parallel
    pairs closer than `stripe_merge_eps` into the stripe centerline.

    Lines with no paint under them (mat seams, panel joins) are dropped before
    the stripe merge, so a seam running beside a sideline cannot be mistaken for
    its second edge.

    :param image: the frame the lines were found on, needed to tell paint from
        a seam.

    Returns representative lines sorted by reference coordinate.
    """
    if not lines:
        return []

    img_shape = image.shape
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    coords = np.array(
        [
            [line_reference_coord(ln["rho"], ln["theta"], img_shape, vertical)]
            for ln in lines
        ],
        dtype=np.float64,
    )
    labels = DBSCAN(eps=eps, min_samples=1).fit(coords).labels_

    result = []
    for label in set(labels):
        members = [lines[i] for i in range(len(lines)) if labels[i] == label]
        pts = np.array([p for m in members for p in m["points"]], dtype=np.float32)

        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        # Orthogonal residual of every supporting point to the fitted line.
        residuals = np.abs((pts[:, 0] - x0) * (-vy) + (pts[:, 1] - y0) * vx)
        rms = float(np.sqrt(np.mean(residuals**2)))
        if rms > straightness_tau or float(residuals.max()) > 2 * straightness_tau:
            continue  # bent line -> reject

        theta = float((np.arctan2(vy, vx) + np.pi / 2) % np.pi)
        rho = float(x0 * np.cos(theta) + y0 * np.sin(theta))
        result.append(
            {
                "rho": rho,
                "theta": theta,
                "length": float(sum(m["length"] for m in members)),
                "ref": float(line_reference_coord(rho, theta, img_shape, vertical)),
            }
        )

    result.sort(key=lambda ln: ln["ref"])
    result = keep_painted_lines(result, gray)
    return merge_stripe_edges(result, img_shape, vertical, stripe_merge_eps)


def _hesse_average(line_a, line_b):
    """
    Midpoint of two near-parallel lines in Hesse form.

    Deliberately unweighted: these are the two painted borders of one stripe, so
    its centreline is the geometric midpoint. Weighting by detected length would
    pull the centre toward whichever border happened to be traced further, and a
    long spurious edge can out-support a short real one several times over.

    Handles the theta wrap-around at 0/pi (a near-vertical pair can sit at
    theta≈0 and theta≈pi with opposite-sign rho yet be the same orientation).
    """
    r1, t1 = line_a["rho"], line_a["theta"]
    r2, t2 = line_b["rho"], line_b["theta"]
    if abs(t1 - t2) > np.pi / 2:
        t2 -= np.pi * np.sign(t2 - t1)
        r2 = -r2
    theta = 0.5 * (t1 + t2)
    rho = 0.5 * (r1 + r2)

    # Bring theta back into [0, pi). Shifting it by pi mirrors the line through
    # the origin unless rho is negated with it: (rho, theta) and (-rho,
    # theta+pi) are the same line, (rho, theta+pi) is a different one. Averaging
    # a near-vertical stripe's two edges lands just below zero routinely, so
    # getting this wrong moves a real sideline to the far side of the frame.
    while theta < 0:
        theta += np.pi
        rho = -rho
    while theta >= np.pi:
        theta -= np.pi
        rho = -rho

    return rho, theta


def merge_stripe_edges(lines, img_shape, vertical, stripe_merge_eps):
    """
    Merge adjacent near-parallel lines closer than `stripe_merge_eps` — the two
    Canny edges of one painted stripe — into its centerline. `lines` must be
    sorted by "ref". Genuinely distinct court lines (>= 46 cm apart) project far
    beyond the threshold, so they are untouched.
    """
    merged = []
    already_paired = False
    for line in lines:
        # a painted stripe contributes exactly two edges, so a line already
        # merged is closed: without this a third nearby line chains into the
        # pair and drags the centreline off the paint
        if (
            merged
            and not already_paired
            and (abs(line["ref"] - merged[-1]["ref"]) < stripe_merge_eps)
        ):
            prev = merged.pop()
            rho, theta = _hesse_average(prev, line)
            merged.append(
                {
                    "rho": float(rho),
                    "theta": float(theta),
                    "length": prev["length"] + line["length"],
                    "ref": float(line_reference_coord(rho, theta, img_shape, vertical)),
                }
            )
            already_paired = True
        else:
            merged.append(line)
            already_paired = False
    return merged


def consensus_lines(lines_per_frame, min_frame_fraction=0.4, cluster_eps=10.0):
    """
    Keep the court lines that persist across frames of a static camera.

    Real court lines appear at the same image position in every frame, while
    occlusions (players) and spurious detections come and go. Lines from all
    frames are clustered by their reference coordinate; a cluster is kept only
    when it contains lines from at least `min_frame_fraction` of the frames,
    and is represented by its median-position member.

    :param lines_per_frame: one list of line dicts (with a "ref" key) per frame.
    :returns: consensus line dicts sorted by position.
    """
    all_lines = []
    frame_index_of_line = []
    for frame_index, frame_lines in enumerate(lines_per_frame):
        for line in frame_lines:
            all_lines.append(line)
            frame_index_of_line.append(frame_index)

    if not all_lines:
        return []

    references = np.array([[line["ref"]] for line in all_lines], dtype=np.float64)
    labels = DBSCAN(eps=cluster_eps, min_samples=1).fit(references).labels_

    number_of_frames = len(lines_per_frame)
    result = []
    for label in set(labels):
        member_indices = [
            i for i, line_label in enumerate(labels) if line_label == label
        ]
        frames_seen = {frame_index_of_line[i] for i in member_indices}
        if len(frames_seen) < min_frame_fraction * number_of_frames:
            continue  # transient line (occlusion edge, shadow, noise)

        members = sorted(
            (all_lines[i] for i in member_indices), key=lambda line: line["ref"]
        )
        median_member = members[len(members) // 2]
        result.append(median_member)

    result.sort(key=lambda line: line["ref"])
    return result


def intersection_of(line_a, line_b):
    """Intersection of two line dicts, reusing `intersection()`. Returns [x, y] or None."""
    pt = intersection(
        [[line_a["rho"], line_a["theta"]]], [[line_b["rho"], line_b["theta"]]]
    )
    return pt[0] if pt is not None else None


def _family_assignments(detected, expected):
    """
    Order-preserving assignments of detected lines to expected model coords.

    Both sides may hold lines the other does not: a detected line can be
    spurious (mat seam, shadow, a stray stripe edge) and an expected line can
    be out of frame or undetected. So for every workable size k, every
    ascending choice of k detected lines is paired with every ascending choice
    of k expected coords. Larger k comes first — more matched lines pin the
    homography better. At least 2 lines are needed to span a homography grid.
    """
    assignments = []
    largest_size = min(len(detected), len(expected))
    for size in range(largest_size, 1, -1):
        for detected_subset in combinations(detected, size):
            for expected_subset in combinations(expected, size):
                assignments.append(list(zip(detected_subset, expected_subset)))
    return assignments


def identify_line_candidates(
    verticals, horizontals, expected_x, expected_y, max_candidates=500
):
    """
    Candidate assignments of detected lines to model lines, by order.

    A partial camera view rarely shows every model line (and Hough sometimes
    adds spurious ones), so instead of demanding an exact count match this
    enumerates order-preserving subset assignments per family and lets the
    caller validate each candidate homography (reprojection RMS + segmentation
    IoU) to pick the right one. Candidates using more lines are tried first.
    Returns a list of
    {"vertical": [(line, x_cm)...], "horizontal": [(line, y_cm)...]}.
    """
    candidates = []
    for vertical_assignment in _family_assignments(verticals, expected_x):
        for horizontal_assignment in _family_assignments(horizontals, expected_y):
            candidates.append(
                {"vertical": vertical_assignment, "horizontal": horizontal_assignment}
            )

    # try the assignments that explain the most lines first, and keep the
    # candidate list bounded
    candidates.sort(
        key=lambda c: len(c["vertical"]) + len(c["horizontal"]), reverse=True
    )
    return candidates[:max_candidates]


def correspondences_from_matches(matches):
    """
    Build image<->model point correspondences from matched lines: every
    vertical x horizontal grid intersection yields one correspondence.
    """
    source, target = [], []
    for vline, x_model in matches["vertical"]:
        for hline, y_model in matches["horizontal"]:
            pt = intersection_of(vline, hline)
            if pt is None:
                continue
            source.append([float(pt[0]), float(pt[1])])
            target.append([float(x_model), float(y_model)])
    return np.array(source, dtype=np.float32), np.array(target, dtype=np.float32)


# ---------------------------------------------------------------------------
# Trajectory-based hit detection helpers
# ---------------------------------------------------------------------------


def fit_velocity(times, points):
    """
    Least-squares velocity (vx, vy) in units per frame from (t, point) samples.

    Uses the time axis (frame numbers) rather than sample index so that gaps in
    the track don't distort the slope. Returns None if it can't be estimated.
    """
    if len(times) < 2:
        return None

    t = np.asarray(times, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    t = t - t.mean()
    denom = float(np.dot(t, t))
    if denom < 1e-9:
        return None

    vx = float(np.dot(t, pts[:, 0] - pts[:, 0].mean()) / denom)
    vy = float(np.dot(t, pts[:, 1] - pts[:, 1].mean()) / denom)
    return np.array([vx, vy])


def angle_between(v1, v2):
    """Unsigned angle between two 2D vectors in degrees ([0, 180]); 0 if degenerate."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))
