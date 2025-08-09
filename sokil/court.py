import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from shapely import Point, Polygon, box

from sokil.models import PoseEstimator, Segmenter

from .util import (
    classify_line_clusters,
    cluster_parallel_lines,
    consensus_lines,
    correspondences_from_matches,
    detect_line_segments,
    filter_outer_lines,
    get_court_rectangle,
    get_intersection_centers,
    identify_line_candidates,
    segment_by_angle_spectralclustering,
    segmented_intersections,
    snap_to_intersection,
    split_line_families,
)

logger = logging.getLogger(__name__)


class CourtCameraMode(Enum):
    FULL = "full"
    LEFT_CORRIDOR = "left_corridor"
    RIGHT_CORRIDOR = "right_corridor"
    BOTTOM_CORRIDOR = "bottom_corridor"
    TOP_CORRIDOR = "top_corridor"


class CourtModelType(Enum):
    SINGLES = "singles"
    DOUBLES = "doubles"


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.m, _ = cv2.findHomography(source, target, cv2.RANSAC)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

        # Calculate the inverse homography matrix
        self.m_inv = np.linalg.inv(self.m)
        if self.m_inv is None:
            raise ValueError("Inverse homography matrix could not be calculated.")

    @classmethod
    def from_matrix(cls, matrix) -> "ViewTransformer":
        """
        Build a transformer from a known image->court homography.

        Skips the point correspondences entirely, for a court that was solved
        earlier and saved rather than re-derived from this video.
        """
        transformer = cls.__new__(cls)
        transformer.m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        try:
            transformer.m_inv = np.linalg.inv(transformer.m)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"Homography is not invertible: {error}") from error
        return transformer

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)

    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(points, self.m_inv)
        return transformed_points.reshape(-1, 2).astype(np.float32)


@dataclass
class CourtConfig:
    snap_keypoints: bool = True
    keypoint_confidence: float = 0.5
    court_camera_mode: CourtCameraMode = CourtCameraMode.FULL
    court_model_type: CourtModelType = CourtModelType.DOUBLES

    # px within which two detected lines count as the same court line.
    line_cluster_eps: float = 6.0
    # max orthogonal RMS [px] of a cluster's supporting points about its fitted line;
    # above this the cluster is a bent line and is rejected
    line_straightness_tau: float = 2.5
    # px within which two parallel lines are the two borders of one stripe and are merged into its centerline
    stripe_merge_eps: float = 25.0


class Court:
    def __init__(
        self,
        image: cv2.typing.MatLike,
        segmentation_model: Segmenter,
        keypoint_model: PoseEstimator | None = None,
        court_config: CourtConfig | None = None,
    ):
        """
        :param model_path: a path to the model that predicts court keypoints
        """
        self.segmentation_model = segmentation_model
        self.keypoint_model = keypoint_model

        self.court_config = court_config if court_config else CourtConfig()

        self._image = image

        self._court_contour = None
        self._keypoints = ()
        self._view_transformer = None
        # segmentation IoU of the accepted geometric homography (None until solved);
        # used by Review to pick the most reliable solve across frames
        self.homography_iou: float | None = None

        self.court_model = CourtModel(
            court_camera_mode=self.court_config.court_camera_mode,
            court_model_type=self.court_config.court_model_type,
        )

    @classmethod
    def from_homography(
        cls,
        matrix,
        court_config: "CourtConfig | None" = None,
        image=None,
    ) -> "Court":
        """
        A court whose homography is given rather than solved.

        Everything the review needs downstream — projecting the model, judging a
        hit, drawing the close-up — goes through the view transformer, so no
        frame and no segmentation model are required.

        :param matrix: the 3x3 image->court homography.
        :param image: optional, only for helpers that draw on the frame.
        """
        court = cls(
            image=image,
            segmentation_model=None,
            keypoint_model=None,
            court_config=court_config,
        )
        court._view_transformer = ViewTransformer.from_matrix(matrix)
        return court

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, img: np.array):
        self._image = img

    def segment(self) -> list[np.ndarray]:
        """
        Segment the court out of the image; returns its mask polygons.
        """
        if self.image is None:
            raise ValueError("image property is not set")

        return self.segmentation_model.polygons(self.image)

    def inside(self, point: tuple[int]) -> bool:
        contour = self.get_contour()
        if contour is None:
            return None
        return cv2.pointPolygonTest(contour, point, False) >= 0

    def get_contour(self):
        """
        Get the contour of the court
        """
        # sort of cache
        if self._court_contour is not None:
            return self._court_contour

        if not self.image.any():
            raise ValueError("image property is not set")

        polygons = self.segment()
        if not polygons:
            return None

        court_shape = polygons[0]

        self._court_contour = court_shape.astype(np.int32).reshape(-1, 1, 2)

        return self._court_contour

    def get_keypoints(self):
        """
        Get court keypoints with >=specified confidence
        """
        if self._keypoints:
            return self._keypoints

        if self.image is None:
            raise ValueError("image property is not set")

        if self.keypoint_model is None:
            return ValueError("Keypoint model is not provided")

        keypoints, keypoint_confidence = self.keypoint_model.keypoints(self.image)

        if keypoints is None or not keypoints.any():
            return None, None

        self._keypoints = (keypoints, keypoint_confidence)

        return self._keypoints

    def get_view_transformer(self):
        """
        Solve the image->court homography.

        Primary path is geometric: detect straight court lines and match them to
        the known court model (no learned keypoints). The YOLO keypoint model is
        used only as a fallback when the geometric path fails or its result does
        not validate.
        """
        if self._view_transformer:
            return self._view_transformer

        transformer = self._geometric_view_transformer()
        if transformer is None and self.keypoint_model is not None:
            logger.warning(
                "Failed to construct a geometric view transformer, falling back to keypoints"
            )
            transformer = self._keypoint_view_transformer()

        if transformer is not None:
            self._view_transformer = transformer

        return transformer

    def _geometric_view_transformer(self):
        """
        Solve homography from straight court lines matched to the court model.
        Returns a validated ViewTransformer or None.
        """
        verticals, horizontals = self.detect_court_lines()
        return self._solve_homography_from_lines(verticals, horizontals)

    def detect_court_lines(self):
        """
        Detect the straight court lines on this frame.

        Returns (vertical_lines, horizontal_lines): clustered line dicts sorted
        by position (each dict has rho/theta/length/ref). Both lists are empty
        when the court edges cannot be extracted.
        """
        edges = self._court_edge_image()
        if edges is None:
            return [], []

        segments = detect_line_segments(edges)
        if not segments:
            return [], []

        verticals, horizontals = split_line_families(segments)
        clustering = {
            "eps": self.court_config.line_cluster_eps,
            "straightness_tau": self.court_config.line_straightness_tau,
            "stripe_merge_eps": self.court_config.stripe_merge_eps,
        }
        verticals = cluster_parallel_lines(
            verticals, self.image, vertical=True, **clustering
        )
        horizontals = cluster_parallel_lines(
            horizontals, self.image, vertical=False, **clustering
        )
        return verticals, horizontals

    def solve_homography_from_lines(self, verticals, horizontals) -> bool:
        """
        Solve and cache the homography from externally supplied court lines
        (e.g. consensus lines aggregated over many frames of a static camera).
        Returns True when a validated homography was found; its segmentation
        IoU is stored in homography_iou.
        """
        transformer = self._solve_homography_from_lines(verticals, horizontals)
        if transformer is None:
            return False
        self._view_transformer = transformer
        return True

    def _solve_homography_from_lines(self, verticals, horizontals):
        # Match against painted-stripe CENTERS: detection yields stripe
        # centerlines, while the model coords are the rule-book boundary edges.
        # Solving on centers puts the true boundaries on the painted edges.
        candidates = identify_line_candidates(
            verticals,
            horizontals,
            self.court_model.stripe_center_coords("x"),
            self.court_model.stripe_center_coords("y"),
        )

        # A partial view does not pin down WHICH model lines are visible, so
        # try every order-preserving assignment and keep the one whose
        # homography best matches the segmentation mask.
        best_transformer = None
        best_iou = -1.0
        for matches in candidates:
            source_pts, target_pts = correspondences_from_matches(matches)
            if len(source_pts) < 4:
                continue

            try:
                transformer = ViewTransformer(source=source_pts, target=target_pts)
            except ValueError:
                continue

            scored = self._transformer_score(transformer, source_pts, target_pts)
            if scored is not None and scored[1] > best_iou:
                best_transformer, best_iou = transformer, scored[1]

        if best_transformer is not None:
            self.homography_iou = best_iou

        return best_transformer

    def solve_geometric_homography(self) -> bool:
        """
        Solve the homography using ONLY the geometric path (no keypoint
        fallback) and cache it on success.

        Unlike get_view_transformer, a True result guarantees the homography
        passed validation (reprojection RMS + segmentation IoU, stored in
        homography_iou) — the keypoint fallback offers no such guarantee, so
        this is the method to use when the result is meant to be reused across
        frames.
        """
        transformer = self._geometric_view_transformer()
        if transformer is None:
            return False
        self._view_transformer = transformer
        return True

    def project_court_corners(self) -> np.ndarray:
        """
        Image positions of the court boundary corners under the solved
        homography. Two solves of a static camera should place these corners at
        the same pixels, which makes them a simple agreement test.
        """
        corners = np.array(self.court_model.doubles_boundaries(), dtype=np.float32)
        return self._view_transformer.inverse_transform_points(corners)

    def _transformer_score(
        self,
        transformer: ViewTransformer,
        source_pts,
        target_pts,
        max_rms=10.0,
        min_iou=0.5,
    ):
        """
        Sanity-check a homography: low reprojection error of the matched
        correspondences AND good overlap between the reprojected court boundary
        and the segmentation contour.

        :returns: (reprojection rms, segmentation IoU), or None when the
            homography fails either check. Both are needed by the caller's
            ranking; neither is a ranking key on its own — rms is measured on
            the points the homography was fitted to, so a minimal four-point
            assignment always reports ~0.
        """
        reprojected = transformer.inverse_transform_points(target_pts)
        rms = float(np.sqrt(np.mean(np.sum((reprojected - source_pts) ** 2, axis=1))))
        if rms > max_rms:
            return None

        contour = self.get_contour()
        if contour is None:
            # no segmentation to rank with; accept on RMS alone
            return rms, min_iou

        h, w = self.image.shape[:2]
        boundary = np.array(self.court_model.doubles_boundaries(), dtype=np.float32)
        projected = transformer.inverse_transform_points(boundary).astype(np.int32)

        court_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(court_mask, [projected], 255)
        seg_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(seg_mask, [contour], -1, 255, cv2.FILLED)

        intersection_area = np.logical_and(court_mask > 0, seg_mask > 0).sum()
        union_area = np.logical_or(court_mask > 0, seg_mask > 0).sum()
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return (rms, iou) if iou >= min_iou else None

    def _keypoint_view_transformer(self):
        """
        Fallback homography from the YOLO keypoint model: predictions are snapped
        to detected line intersections, then a homography is solved. Returns a
        ViewTransformer or None.
        """
        confidence = self.court_config.keypoint_confidence
        keypoints, keypoint_confidence = self.get_keypoints()

        if keypoints is None:
            return None

        keypoint_conf_filter = keypoint_confidence > confidence

        filtered_keypoints = [
            (i, x, y, c)
            for i, ((x, y), c) in enumerate(zip(keypoints, keypoint_confidence))
            if c > confidence
        ]

        hough_lines = self.court_hough_lines_from_yolo()
        if hough_lines is None:
            return None

        hough_outer_lines = filter_outer_lines(hough_lines)
        segmented = segment_by_angle_spectralclustering(hough_outer_lines)
        intersections = segmented_intersections(segmented)
        intersection_centers = get_intersection_centers(intersections)
        if intersection_centers is None:
            logger.warning(
                "Cannot get view transformer because no intersection centers have been found"
            )
            return None

        snapped_keypoints = snap_to_intersection(
            filtered_keypoints, intersection_centers, max_radius=60
        )

        source_pts = keypoints[keypoint_conf_filter].astype(np.float32)
        if self.court_config.snap_keypoints:
            source_pts = np.array(snapped_keypoints)[:, 1:3].astype(np.float32)

        # The keypoint model always predicts the full 30-vertex court, so the
        # confidence mask indexes the full vertex list (not the camera-mode
        # subset); occluded vertices are dropped by the confidence filter.
        vertices = self.court_model.vertices

        target_pts = np.array(vertices)[keypoint_conf_filter].astype(np.float32)

        required_points = 4
        if len(source_pts) < required_points or len(target_pts) < required_points:
            logger.warning(
                "Cannot get view transformer because at least 4 points are required"
            )
            return None

        try:
            return ViewTransformer(
                source=source_pts,
                target=target_pts,
            )
        except ValueError as error:
            logger.warning(
                "Cannot get view transformer from %d keypoints: %s",
                len(source_pts),
                error,
            )
            return None

    def _court_edge_image(self):
        """
        Build the masked edge image of the court: white court lines are enhanced
        with a morphological tophat, edges detected, then masked to the (dilated)
        court segmentation region. Shared by the geometric and keypoint paths.
        """
        img = self.image.copy()
        contour = self.get_contour()
        if contour is None or not contour.any():
            return None

        b_mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # Define the dilation kernel (adjust size as needed)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (20, 20)
        )  # larger kernel = more dilation

        # Apply dilation to make sure the entire court is covered
        dilated_mask = cv2.dilate(b_mask, kernel, iterations=1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Court lines are white/bright on dark background
        # Use morphological tophat to isolate thin bright lines
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        tophat_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_line)
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        tophat_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_line)
        enhanced = cv2.add(tophat_h, tophat_v)

        edges = cv2.Canny(enhanced, 50, 100)

        # Mask edges so that lines are found only within the segment
        return cv2.bitwise_and(edges, edges, mask=dilated_mask)

    def court_hough_lines_from_yolo(self, line_threshold=100):
        """
        Extract infinite Hough lines inside the court region.
        Used by the keypoint fallback path.
        """
        edges_masked = self._court_edge_image()
        if edges_masked is None:
            return None

        return cv2.HoughLines(edges_masked, 1, np.pi / 180, threshold=line_threshold)

    def get_court_projection_points(self):
        """
        Solve the H-matrix and project the court vertices onto the image.

        Always projects the full 30-vertex model in global order, so the result
        can be indexed directly by edge vertex IDs (1..30). Which edges are
        actually drawn is decided by the caller via the camera mode's edge subset
        (see CourtModel.get_court_model), so corridor modes don't render
        extrapolated far-side lines.
        """
        transformer = self.get_view_transformer()

        if transformer is None:
            return None

        vertices = self.court_model.vertices

        court_points_in_image = transformer.inverse_transform_points(
            points=np.array(vertices)
        )
        return court_points_in_image.astype(np.int32)

    def _in_region(self):
        """
        Build the IN region as an axis-aligned box bounded only by the perimeter
        lines this camera owns (see CourtModel.out_boundaries). Unowned
        (open) sides extend far into the unobserved court, so a shuttle deep in
        the court still counts as IN — we only ever call OUT on a watched edge.
        """
        big = 1e5
        lo_x, hi_x, lo_y, hi_y = -big, big, -big, big
        for axis, coord, out_sign in self.court_model.out_boundaries():
            if axis == "x":
                if out_sign < 0:
                    lo_x = coord
                else:
                    hi_x = coord
            else:
                if out_sign < 0:
                    lo_y = coord
                else:
                    hi_y = coord
        return box(lo_x, lo_y, hi_x, hi_y)

    def shuttle_intersects_court(self, shuttle_position: tuple) -> bool:
        """
        Predict whether the shuttle hits inside or outside the court.

        The shuttle's real radius is taken into account: if any
        part of the shuttle touches the IN region it counts as IN, even when its
        center is on the out-side of a line.
        :return bool: True if IN, False if OUT
        """

        transformer = self.get_view_transformer()

        if transformer is None:
            return None

        in_region = self._in_region()

        shuttle_array = np.array([shuttle_position], dtype=np.float32)

        shuttle_transformed = transformer.transform_points(points=shuttle_array)[0]

        shuttle_circle = Point(shuttle_transformed).buffer(
            self.court_model.shuttle_diameter / 2
        )

        return in_region.intersects(shuttle_circle)

    def shuttle_inside_max_observable_area(self, shuttle_position: tuple, padding=50):
        """
        Check if the shuttle is inside the area of the court + padding
        This way, we don't detect hits that happen too high or in different courts.

        Uses the full physical court extent (doubles outer rectangle) regardless
        of camera mode — this is a bounded plausibility gate, not the verdict.
        """
        transformer = self.get_view_transformer()

        if transformer is None:
            return None

        court_polygon = Polygon(self.court_model.doubles_boundaries())

        dilated_court_polygon = court_polygon.buffer(padding)

        shuttle_array = np.array([shuttle_position], dtype=np.float32)

        shuttle_transformed = transformer.transform_points(points=shuttle_array)[0]

        shuttle_circle = Point(shuttle_transformed).buffer(
            self.court_model.shuttle_diameter / 2
        )

        return dilated_court_polygon.intersects(shuttle_circle)

    def get_court_rectangle(self):
        """
        Get 4 outermost points of the court in the following order:
        - top left
        - top right
        - bottom right
        - bottom left
        """
        hough_lines = self.court_hough_lines_from_yolo()
        if hough_lines is None:
            return None

        hough_outer_lines = filter_outer_lines(hough_lines)
        segmented = segment_by_angle_spectralclustering(hough_outer_lines)

        vertical, horizontal = classify_line_clusters(*segmented)

        court_points = get_court_rectangle(self.image.shape, vertical, horizontal)

        if court_points is None:
            logger.warning(
                "Cannot construct the court rectangle because not enough lines were found"
            )
            return None

        return court_points

    def plot(
        self,
        shuttle_position: tuple | None = None,
        is_shuttle_in: bool | None = None,
        **kwargs,
    ):
        """
        Draw a badminton court
        """
        transformer = self.get_view_transformer()

        if not transformer:
            return None, None

        shuttle_array = np.array([shuttle_position], dtype=np.float32)

        shuttle_transformed = transformer.transform_points(points=shuttle_array)[0]

        fig, ax = draw_hit_closeup_matplotlib(
            config=self.court_model,
            shuttle_position=shuttle_transformed,
            is_shuttle_in=is_shuttle_in,
            **kwargs,
        )

        return fig, ax


@dataclass
class CourtModel:
    width: int = 610  # [cm]
    length: int = 1340  # [cm]
    side_corridor_width = 46  # [cm]
    back_corridor_width = 76  # [cm]
    net_corridor_width = 396  # [cm] 198 on each net side
    line_width = 4  # [cm]
    shuttle_diameter = 2.64  # [cm]
    court_model_type: CourtModelType = CourtModelType.DOUBLES
    court_camera_mode: CourtCameraMode = CourtCameraMode.FULL

    @property
    def vertices(self) -> list[tuple[int, int]]:
        return [
            (0, 0),  # 1
            (self.side_corridor_width, 0),  # 2
            (self.width / 2, 0),  # 3
            (self.width - self.side_corridor_width, 0),  # 4
            (self.width, 0),  # 5
            (self.width, self.back_corridor_width),  # 6
            ((self.width - self.side_corridor_width), self.back_corridor_width),  # 7
            ((self.width / 2), self.back_corridor_width),  # 8
            ((0 + self.side_corridor_width), self.back_corridor_width),  # 9
            (0, self.back_corridor_width),  # 10
            (0, (self.back_corridor_width + self.net_corridor_width)),  # 11
            (
                (0 + self.side_corridor_width),
                (self.back_corridor_width + self.net_corridor_width),
            ),  # 12
            (
                (self.width / 2),
                (self.back_corridor_width + self.net_corridor_width),
            ),  # 13
            (
                (self.width - self.side_corridor_width),
                (self.back_corridor_width + self.net_corridor_width),
            ),  # 14
            (self.width, self.back_corridor_width + self.net_corridor_width),  # 15
            (self.width, self.length / 2 + self.net_corridor_width / 2),  # 16
            (
                self.width - self.side_corridor_width,
                self.length / 2 + self.net_corridor_width / 2,
            ),  # 17
            (self.width / 2, self.length / 2 + self.net_corridor_width / 2),  # 18
            (
                0 + self.side_corridor_width,
                self.length / 2 + self.net_corridor_width / 2,
            ),  # 19
            (0, self.length / 2 + self.net_corridor_width / 2),  # 20
            (0, self.length - self.back_corridor_width),  # 21
            (
                0 + self.side_corridor_width,
                self.length - self.back_corridor_width,
            ),  # 22
            (self.width / 2, self.length - self.back_corridor_width),  # 23
            (
                self.width - self.side_corridor_width,
                self.length - self.back_corridor_width,
            ),  # 24
            (self.width, self.length - self.back_corridor_width),  # 25
            (self.width, self.length),  # 26
            (self.width - self.side_corridor_width, self.length),  # 27
            (self.width / 2, self.length),  # 28
            (0 + self.side_corridor_width, self.length),  # 29
            (0, self.length),  # 30
        ]

    edges: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 8),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (7, 8),
            (8, 9),
            (9, 2),
            (9, 10),
            (10, 1),
            (10, 11),
            (11, 12),
            (11, 20),
            (12, 9),
            (12, 13),
            (12, 19),
            (13, 8),
            (13, 14),
            (14, 7),
            (14, 15),
            (14, 17),
            (15, 6),
            (15, 16),
            (16, 25),
            (17, 16),
            (17, 18),
            (18, 23),
            (19, 18),
            (19, 20),
            (21, 20),
            (21, 22),
            (22, 19),
            (22, 23),
            (22, 29),
            (23, 24),
            (23, 28),
            (24, 17),
            (24, 25),
            (25, 26),
            (26, 27),
            (27, 24),
            (28, 27),
            (28, 29),
            (29, 30),
            (30, 21),
        ]
    )

    def _get_vertices_subset(self, vertex_ids: list[int]):
        vertices = []
        for i, vertex in enumerate(self.vertices):
            keypoint_index = i + 1
            if keypoint_index in vertex_ids:
                vertices.append(vertex)

        return vertices

    def doubles_boundaries(self):
        return [
            self.vertices[0],
            self.vertices[4],
            self.vertices[25],
            self.vertices[29],
        ]

    def singles_boundaries(self):
        return [
            self.vertices[1],
            self.vertices[3],
            self.vertices[26],
            self.vertices[28],
        ]

    def _bounding_rectangle(self, vertices):
        """Axis-aligned bounding rectangle (4 corners, CW from top-left) of a vertex subset."""
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    def out_boundaries(self):
        """
        Outer perimeter lines this camera adjudicates, as (axis, coord, out_sign):
        a shuttle is OUT when (value_on_axis - coord) * out_sign > 0.

        Each camera owns every visible outer line; sides that continue into the
        unobserved court are omitted, so they stay open (always IN). Sideline
        coordinates follow court_model_type — singles uses the inner sidelines
        because the doubles alley is out.
        """
        left_x = 0
        right_x = self.width

        if self.court_model_type == CourtModelType.SINGLES:
            left_x = self.side_corridor_width
            right_x = self.width - self.side_corridor_width

        left = ("x", left_x, -1)
        right = ("x", right_x, 1)
        near = ("y", 0, -1)
        far = ("y", self.length, 1)

        match self.court_camera_mode:
            case CourtCameraMode.FULL:
                return [left, right, near, far]
            case CourtCameraMode.LEFT_CORRIDOR:
                return [left, near, far]
            case CourtCameraMode.RIGHT_CORRIDOR:
                return [right, near, far]
            case CourtCameraMode.BOTTOM_CORRIDOR:
                return [near, left, right]
            case CourtCameraMode.TOP_CORRIDOR:
                return [far, left, right]

    def get_boundaries(self):
        """
        Closed boundary polygon of the observed court region. Superseded for the
        in/out verdict by out_boundaries() + Court._in_region() (open half-planes);
        retained for region/area drawing helpers.
        """
        match self.court_camera_mode:
            case CourtCameraMode.FULL:
                match self.court_model_type:
                    case CourtModelType.DOUBLES:
                        return self.doubles_boundaries()
                    case CourtModelType.SINGLES:
                        return self.singles_boundaries()
            case CourtCameraMode.LEFT_CORRIDOR:
                return self._bounding_rectangle(self.left_corridor_vertices())
            case CourtCameraMode.RIGHT_CORRIDOR:
                return self._bounding_rectangle(self.right_corridor_vertices())
            case CourtCameraMode.BOTTOM_CORRIDOR:
                return self._bounding_rectangle(self.bottom_corridor_vertices())
            case CourtCameraMode.TOP_CORRIDOR:
                return self._bounding_rectangle(self.top_corridor_vertices())

    def left_corridor_vertices(self):
        vertex_ids = [1, 2, 9, 10, 11, 12, 19, 20, 21, 22, 29, 30]
        return self._get_vertices_subset(vertex_ids)

    def right_corridor_vertices(self):
        vertex_ids = [4, 5, 6, 7, 14, 15, 16, 17, 24, 25, 26, 27]
        return self._get_vertices_subset(vertex_ids)

    def bottom_corridor_vertices(self):
        vertex_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return self._get_vertices_subset(vertex_ids)

    def top_corridor_vertices(self):
        vertex_ids = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        return self._get_vertices_subset(vertex_ids)

    def _filter_edges_for_vertices(
        self, vertices: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Returns edges where both endpoints are in the given vertex list."""
        vertex_set = set(map(tuple, vertices))
        all_vertices = self.vertices  # full list, 0-indexed

        return [
            (start, end)
            for start, end in self.edges
            if tuple(all_vertices[start - 1]) in vertex_set
            and tuple(all_vertices[end - 1]) in vertex_set
        ]

    def left_corridor_edges(self) -> list[tuple[int, int]]:
        return self._filter_edges_for_vertices(self.left_corridor_vertices())

    def right_corridor_edges(self) -> list[tuple[int, int]]:
        return self._filter_edges_for_vertices(self.right_corridor_vertices())

    def bottom_corridor_edges(self) -> list[tuple[int, int]]:
        return self._filter_edges_for_vertices(self.bottom_corridor_vertices())

    def top_corridor_edges(self) -> list[tuple[int, int]]:
        return self._filter_edges_for_vertices(self.top_corridor_vertices())

    def unique_line_coords(self, axis: str) -> list[float]:
        """
        Unique vertical (axis="x") or horizontal (axis="y") court-line
        coordinates [cm] for the active camera mode, ascending. These define
        exactly which/how many straight lines should be visible, so detected
        lines can be matched to the model by order.
        """
        vertices, _ = self.get_court_model()
        index = 0 if axis == "x" else 1
        return sorted({round(float(v[index]), 3) for v in vertices})

    def stripe_center_offset(self, axis: str, coord: float) -> float:
        """
        Offset [cm] from a model boundary coordinate to the CENTER of its
        painted stripe. Model coords are the rule-book measuring edges (lines
        belong to the court area they bound), so each stripe extends line_width
        from its boundary coord toward that area. Line detection sees stripe
        centerlines, so matching must compare against coord + this offset.
        """
        half = self.line_width / 2.0
        net_y = self.length / 2.0
        short_service_top = self.back_corridor_width + self.net_corridor_width

        if axis == "x":
            if coord == self.width / 2:
                return 0.0  # center line straddles the service courts
            # sidelines: stripe lies inside the court -> toward the middle
            return half if coord < self.width / 2 else -half

        # axis == "y"
        if coord in (0.0, self.length):
            return half if coord == 0.0 else -half  # back lines: stripe inside
        # service lines: stripe lies inside the service court it bounds,
        # i.e. away from the net for short service lines, toward the net for
        # long service lines
        if coord in (short_service_top, self.length - short_service_top):
            return -half if coord < net_y else half
        return half if coord < net_y else -half

    def stripe_center_coords(self, axis: str) -> list[float]:
        """unique_line_coords shifted to painted-stripe centers, ascending."""
        return sorted(
            coord + self.stripe_center_offset(axis, coord)
            for coord in self.unique_line_coords(axis)
        )

    def get_court_model(self):
        match self.court_camera_mode:
            case CourtCameraMode.FULL:
                return self.vertices, self.edges
            case CourtCameraMode.LEFT_CORRIDOR:
                return self.left_corridor_vertices(), self.left_corridor_edges()
            case CourtCameraMode.RIGHT_CORRIDOR:
                return self.right_corridor_vertices(), self.right_corridor_edges()
            case CourtCameraMode.BOTTOM_CORRIDOR:
                return self.bottom_corridor_vertices(), self.bottom_corridor_edges()
            case CourtCameraMode.TOP_CORRIDOR:
                return self.top_corridor_vertices(), self.top_corridor_edges()


def _draw_partial_court_overlay(
    ax, config: "CourtModel", bounds, out_color="red", out_alpha=0.3
):
    """
    Corridor-view overlay: shade the OUT region beyond each owned perimeter line
    in red, and draw a "court continues" arrow on each open side (the directions
    the real court extends but the camera cannot see).
    """
    xmin, xmax, ymin, ymax = bounds
    owned_sides = set()

    for axis, coord, out_sign in config.out_boundaries():
        owned_sides.add((axis, "lo" if out_sign < 0 else "hi"))

        if axis == "x" and out_sign < 0:  # OUT to the left
            rect = Rectangle((xmin, ymin), coord - xmin, ymax - ymin)
        elif axis == "x":  # OUT to the right
            rect = Rectangle((coord, ymin), xmax - coord, ymax - ymin)
        elif out_sign < 0:  # OUT below (near baseline)
            rect = Rectangle((xmin, ymin), xmax - xmin, coord - ymin)
        else:  # OUT above (far baseline)
            rect = Rectangle((xmin, coord), xmax - xmin, ymax - coord)

        rect.set(facecolor=out_color, alpha=out_alpha, edgecolor="none", zorder=0.5)
        ax.add_patch(rect)

    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    dx, dy = 0.18 * (xmax - xmin), 0.18 * (ymax - ymin)
    label = "court continues"

    arrows = {
        ("x", "hi"): {"xy": (xmax, cy), "xytext": (xmax - dx, cy), "rotation": 90},
        ("x", "lo"): {"xy": (xmin, cy), "xytext": (xmin + dx, cy), "rotation": 90},
        ("y", "hi"): {"xy": (cx, ymax), "xytext": (cx, ymax - dy), "rotation": 0},
        ("y", "lo"): {"xy": (cx, ymin), "xytext": (cx, ymin + dy), "rotation": 0},
    }

    for side, opts in arrows.items():
        if side in owned_sides:
            continue  # this edge is a judged boundary, not an open side
        ax.annotate(
            label,
            xy=opts["xy"],
            xytext=opts["xytext"],
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            rotation=opts["rotation"],
            arrowprops={"arrowstyle": "->", "color": "white"},
            zorder=3,
        )


def draw_badminton_court_matplotlib(
    config: CourtModel,
    shuttle_position: tuple | None = None,
    is_shuttle_in: bool | None = None,
    background_color: str = "forestgreen",
    line_color: str = "white",
    padding: int = 50,
    figsize: tuple = (10, 10),
    dpi: int = 100,
) -> np.ndarray:
    """
    draw badminton court with the shuttle position in it using matplotlib primitives
    this makes the image SVG-compatible
    :param shuttle_verdict: whether the shuttle is in or out
    """
    vertices, edges = config.get_court_model()

    # View bounds. For corridor modes, extend the open (unowned) sides so there
    # is room for the "court continues" arrows that mark the unseen court.
    all_x = [v[0] for v in vertices]
    all_y = [v[1] for v in vertices]
    xmin, xmax = min(all_x) - padding, max(all_x) + padding
    ymin, ymax = min(all_y) - padding, max(all_y) + padding

    is_partial = config.court_camera_mode != CourtCameraMode.FULL
    if is_partial:
        open_extra = 200
        owned_sides = {
            (axis, "lo" if out_sign < 0 else "hi")
            for axis, _, out_sign in config.out_boundaries()
        }
        if ("x", "hi") not in owned_sides:
            xmax += open_extra
        if ("x", "lo") not in owned_sides:
            xmin -= open_extra
        if ("y", "hi") not in owned_sides:
            ymax += open_extra
        if ("y", "lo") not in owned_sides:
            ymin -= open_extra

    bounds = (xmin, xmax, ymin, ymax)
    view_w, view_h = xmax - xmin, ymax - ymin

    # Size the figure to the court's own aspect ratio so the court fills the
    # output (no large letterbox margins) and stays crisp. The caller's figsize
    # height sets the scale; the width follows the court aspect.
    fig_height_in = figsize[1]
    fig_width_in = fig_height_in * (view_w / view_h)

    # 1 cm in the figure -> this many points (based on the actual view height)
    cm_to_points = (fig_height_in * 72) / view_h
    scaled_line_width = config.line_width * cm_to_points

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)

    # Set background color
    fig.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    if is_partial:
        _draw_partial_court_overlay(ax, config, bounds)

    for start, end in edges:
        x1, y1 = config.vertices[start - 1]
        x2, y2 = config.vertices[end - 1]

        # 2 cm is subtracted because the line width is 4 cm
        # and the line position is on the outer edge of the court
        # therefore, to get 2 cm from each side, we subtract half
        line_middle = config.line_width / 2
        ax.plot(
            [x1 - line_middle, x2 - line_middle],
            [y1 - line_middle, y2 - line_middle],
            color=line_color,
            linewidth=scaled_line_width,
            alpha=1,
            zorder=1,
        )

    if shuttle_position is not None:
        x, y = shuttle_position
        shuttle_radius = 2.64 / 2 * cm_to_points
        ax.add_patch(Circle((x, y), shuttle_radius, color="red"))

        # --- Inset axes for zooming in ---
        # Define the box for the inset axes. This is where the zoomed-in view will be.
        # [left, bottom, width, height] in axes coordinates (0 to 1)
        axins = ax.inset_axes([0.65, 0.65, 0.3, 0.3])

        # Set the limits of the inset axes to focus on the shuttle
        zoom_factor = 50
        x1_zoom, x2_zoom = x - zoom_factor, x + zoom_factor
        y1_zoom, y2_zoom = y - zoom_factor, y + zoom_factor

        # Calculate a NEW scaling for the zoomed-in plot
        # The new height in cm is the difference in y-limits of the inset
        height_cm_zoomed = y2_zoom - y1_zoom
        cm_to_inches_y_zoomed = (
            axins.get_figure().get_figheight()
            * axins.get_position().height
            / height_cm_zoomed
        )
        new_cm_to_points = cm_to_inches_y_zoomed * 72
        scaled_line_width_zoom = config.line_width * new_cm_to_points

        shuttle_radius_zoomed = 2.64 / 2 * new_cm_to_points

        # Plot the court lines on the inset axes
        for start, end in edges:
            x1, y1 = config.vertices[start - 1]
            x2, y2 = config.vertices[end - 1]
            axins.plot(
                [x1 - line_middle, x2 - line_middle],
                [y1 - line_middle, y2 - line_middle],
                color=line_color,
                linewidth=scaled_line_width_zoom,
                alpha=1,
                zorder=1,
            )

        # Add the shuttle to the inset axes
        axins.add_patch(Circle((x, y), shuttle_radius_zoomed, color="red"))

        axins.set_xlim(x1_zoom, x2_zoom)
        axins.set_ylim(y1_zoom, y2_zoom)

        if is_shuttle_in is not None:
            # Verdict label in a self-sizing box (bbox grows to fit the text, so
            # the text can never overflow it). Positioned in axes fractions near
            # the top of the inset, independent of the zoom scale.
            verdict = "IN" if is_shuttle_in else "OUT"

            axins.text(
                0.5,
                0.88,
                verdict,
                transform=axins.transAxes,
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "gold",
                    "edgecolor": "black",
                    "alpha": 0.9,
                },
                zorder=3,
            )

        # Adjust aspect ratio and turn off axes for the inset
        axins.set_aspect("equal", adjustable="box")
        axins.set_facecolor(background_color)
        axins.set_xticks([])
        axins.set_yticks([])

        # Draw lines connecting the main plot to the inset plot for context
        ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.set_xlim(xmin, xmax)
    # court coordinates grow DOWNWARD (like image coordinates): invert the
    # y-axis so the end of the court nearest the camera draws nearest the
    # bottom, the same way draw_hit_closeup_matplotlib orients its window
    ax.set_ylim(ymax, ymin)

    # Equal aspect ratio (so lines don’t stretch)
    ax.set_aspect("equal", adjustable="box")

    # Hide axes
    ax.axis("off")

    # Remove figure margins so the court fills the rasterized image edge-to-edge.
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig, ax


def draw_hit_closeup_matplotlib(
    config: CourtModel,
    shuttle_position: tuple,
    is_shuttle_in: bool | None = None,
    background_color: str = "forestgreen",
    line_color: str = "white",
    out_color: str = "red",
    figsize: tuple = (10, 10),
    dpi: int = 100,
    min_half_window: float = 15,
    max_half_window: float = 100,
    margin: float = 12,
    **_,
):
    """
    Umpire close-up of a single landing: a tight view centered on the shuttle
    showing the relevant boundary line(s), the shuttle at true scale, the OUT
    region, and the verdict + how far in/out it was (cm).

    The view window is sized to include the shuttle and the boundary line that
    decides the call, and matches the output (figsize) aspect ratio so it fills
    the frame edge-to-edge instead of letterboxing the whole court.
    """
    sx, sy = float(shuttle_position[0]), float(shuttle_position[1])
    radius = config.shuttle_diameter / 2
    fences = config.out_boundaries()

    # Signed out-distance of the shuttle center from each owned fence:
    # >0 means the center is on the OUT side of that line.
    fence_dists = []
    for axis, coord, out_sign in fences:
        value = sx if axis == "x" else sy
        fence_dists.append((value - coord) * out_sign)

    nearest_dist = min(abs(d) for d in fence_dists) if fence_dists else min_half_window

    # Window centered on the shuttle, big enough to reveal the nearest line,
    # clamped so the shuttle never shrinks to a dot or the line dwarfs it.
    aspect = figsize[0] / figsize[1]
    half_h = min(max_half_window, max(min_half_window, nearest_dist + margin))
    half_w = half_h * aspect

    fig_height_in = figsize[1]
    fig_width_in = fig_height_in * aspect
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    fig.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    xmin, xmax = sx - half_w, sx + half_w
    ymin, ymax = sy - half_h, sy + half_h

    # OUT shading beyond each owned fence that falls inside the window.
    for axis, coord, out_sign in fences:
        if axis == "x" and out_sign < 0 and coord > xmin:
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    min(coord, xmax) - xmin,
                    ymax - ymin,
                    facecolor=out_color,
                    alpha=0.3,
                    edgecolor="none",
                    zorder=0.5,
                )
            )
        elif axis == "x" and out_sign > 0 and coord < xmax:
            left = max(coord, xmin)
            ax.add_patch(
                Rectangle(
                    (left, ymin),
                    xmax - left,
                    ymax - ymin,
                    facecolor=out_color,
                    alpha=0.3,
                    edgecolor="none",
                    zorder=0.5,
                )
            )
        elif axis == "y" and out_sign < 0 and coord > ymin:
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    min(coord, ymax) - ymin,
                    facecolor=out_color,
                    alpha=0.3,
                    edgecolor="none",
                    zorder=0.5,
                )
            )
        elif axis == "y" and out_sign > 0 and coord < ymax:
            bottom = max(coord, ymin)
            ax.add_patch(
                Rectangle(
                    (xmin, bottom),
                    xmax - xmin,
                    ymax - bottom,
                    facecolor=out_color,
                    alpha=0.3,
                    edgecolor="none",
                    zorder=0.5,
                )
            )

    # Court lines at true scale (clipped to the window by the axis limits).
    # A line's painted body lies INSIDE the court, so its outer edge sits on the
    # nominal boundary coordinate — the same edge the in/out logic uses. Offset
    # each line toward the court interior by half its width so the shuttle's
    # position relative to the line reads correctly (a shuttle out by 1 cm sits
    # just outside the band, not in its middle).
    line_width_pts = config.line_width * (fig_height_in * 72) / (2 * half_h)
    line_middle = config.line_width / 2
    center_x, center_y = config.width / 2, config.length / 2
    for start, end in config.edges:
        x1, y1 = config.vertices[start - 1]
        x2, y2 = config.vertices[end - 1]
        if x1 == x2:  # vertical line -> offset along x toward the interior
            off_x, off_y = line_middle * np.sign(center_x - x1), 0.0
        elif y1 == y2:  # horizontal line -> offset along y toward the interior
            off_x, off_y = 0.0, line_middle * np.sign(center_y - y1)
        else:
            off_x, off_y = 0.0, 0.0
        ax.plot(
            [x1 + off_x, x2 + off_x],
            [y1 + off_y, y2 + off_y],
            color=line_color,
            linewidth=line_width_pts,
            zorder=1,
        )

    # Shuttle at true scale, with a center cross marking the exact contact point.
    ax.add_patch(
        Circle(
            (sx, sy),
            radius,
            facecolor="red",
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )
    )
    ax.plot(
        sx, sy, marker="+", color="black", markersize=8, markeredgewidth=1.5, zorder=5
    )

    # Verdict + precise gap, measured against the line that decides the call:
    # the most-violated fence when OUT, otherwise the closest one when IN.
    label = ""
    if fence_dists:
        decisive = max(fence_dists)  # largest signed out-distance
        if is_shuttle_in is False:
            label = f"OUT by {max(decisive - radius, 0):.1f} cm"
        elif is_shuttle_in is True:
            clearance = -decisive - radius
            label = (
                f"IN by {clearance:.1f} cm" if clearance >= 0 else "IN (touching line)"
            )
    elif is_shuttle_in is not None:
        label = "IN" if is_shuttle_in else "OUT"

    if label:
        ax.text(
            0.5,
            0.93,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="black",
            fontsize=13,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "gold",
                "edgecolor": "black",
                "alpha": 0.95,
            },
            zorder=6,
        )

    ax.set_xlim(xmin, xmax)
    # court coordinates grow DOWNWARD (like image coordinates): invert the
    # y-axis so a fence below the shuttle in the world draws below it here
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig, ax


@dataclass
class CourtSolverConfig:
    """Knobs for the session court solve (see CourtSolver)."""

    sample_seconds: float = 1.0  # spacing of the frames lines are detected on
    min_frame_fraction: float = 0.4  # share of frames a line must appear in
    cluster_eps: float = 10.0  # px within which lines count as the same line
    min_samples: int = 12  # Fewest frames to sample


def mask_completeness(court: "Court") -> float:
    """
    How much of the court this frame's segmentation actually sees.

    The camera is static, so the court's true footprint is the same on every
    frame: a smaller or more ragged mask means a player is standing in front of
    it. Area alone would favour a bloated mask and solidity alone a small tidy
    one, so the two are multiplied.

    :returns: 0.0 when the frame has no usable court mask.
    """
    contour = court.get_contour()
    if contour is None or len(contour) < 3:
        return 0.0

    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if area <= 0 or hull_area <= 0:
        return 0.0

    return float(area * (area / hull_area))


def save_homography(court: "Court", path: str | Path) -> Path:
    """
    Write a solved court's homography so later runs can reuse it.

    :returns: the path written.
    """
    transformer = court.get_view_transformer()
    if transformer is None:
        raise ValueError("Court has no homography to save")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ",\n".join(
        "    [" + ", ".join(f"{value:.10g}" for value in row) + "]"
        for row in transformer.m.tolist()
    )
    path.write_text('{\n  "image_to_court": [\n' + rows + "\n  ]\n}\n")
    logger.info("Court homography written -> %s", path)
    return path


def load_homography(path: str | Path) -> np.ndarray:
    """
    Read an image->court homography.

    Accepts either {"image_to_court": [[...]]} or a bare 3x3 array, so a matrix
    from anywhere can be dropped in.
    """
    document = json.loads(Path(path).read_text())
    matrix = document["image_to_court"] if isinstance(document, dict) else document
    return np.asarray(matrix, dtype=np.float64).reshape(3, 3)


class CourtSolver:
    """
    Finds the session court: solves the homography once, from consensus lines.

    The camera is static, so the real court lines sit at the same image
    position in every frame, while occlusions (players) and spurious lines
    (shadows, noise) come and go. Court lines are detected on frames sampled
    about a second apart, and only the lines that persist across many frames
    are kept, each at its median position. The homography is then solved ONCE
    from those consensus lines — far more reliable than any single frame, where
    an occluded or extra line can push the solver into a plausible-but-wrong
    assignment.

    Only the validated geometric path is used here; the keypoint fallback is
    never frozen because its result is not validated.
    """

    def __init__(
        self, court_detector_kwargs: dict, config: "CourtSolverConfig | None" = None
    ):
        self.court_detector_kwargs = court_detector_kwargs
        self.config = config or CourtSolverConfig()

        # What the last solve() saw on its way to the homography.
        # Keep the record for tracing and debugging
        self.sample_stride: int = 0
        self.sampled_frame_numbers: list[int] = []
        # (verticals, horizontals) as detected on each sampled frame, aligned
        # with sampled_frame_numbers
        self.lines_per_frame: list[tuple[list, list]] = []
        self.consensus_verticals: list = []
        self.consensus_horizontals: list = []
        # index into sampled_frame_numbers of the frame the homography was
        # solved on (the least occluded one), None until a court is solved
        self.solved_on_index: int | None = None

    def solve(self, source) -> "Court | None":
        """
        :param source: the VideoSource to sample frames from (decoded fresh —
            the solver needs about one frame per second, not the whole video).
        :returns: the solved and validated session Court, or None.
        """
        from rich.progress import track as progress_track

        logger.info("Solving the session court homography from consensus lines")

        stride = max(1, round((source.fps or 30) * self.config.sample_seconds))
        # For short clips, the number of samples may be too low to solve court reliably
        # if that's the case, fall back to the configuration value
        if source.frame_count > 0 and self.config.min_samples > 1:
            stride = max(1, min(stride, source.frame_count // self.config.min_samples))

        sampled = [(decoded.number, decoded.image) for decoded in source.sample(stride)]
        if not sampled:
            return None

        sampled_images = [image for _, image in sampled]

        self.sample_stride = stride
        self.sampled_frame_numbers = [number for number, _ in sampled]
        self.lines_per_frame = []
        self.consensus_verticals = []
        self.consensus_horizontals = []
        self.solved_on_index = None

        vertical_lines_per_frame = []
        horizontal_lines_per_frame = []

        # The frame kept here is the one the homography is finally solved on.
        # Its only job is to supply the segmentation mask that scores candidate
        # line assignments, so the least-occluded mask is what matters — a frame
        # with a player standing across the court measures every candidate
        # badly. Only the best is retained, so a long clip does not hold every
        # sampled frame in memory.
        session_court = None
        best_completeness = -1.0
        best_index = -1

        for index, image in enumerate(
            progress_track(sampled_images, description="Detecting court")
        ):
            court = Court(**self.court_detector_kwargs, image=image)
            vertical_lines, horizontal_lines = court.detect_court_lines()
            vertical_lines_per_frame.append(vertical_lines)
            horizontal_lines_per_frame.append(horizontal_lines)
            self.lines_per_frame.append((vertical_lines, horizontal_lines))

            completeness = mask_completeness(court)
            if completeness > best_completeness:
                session_court, best_completeness, best_index = (
                    court,
                    completeness,
                    index,
                )

        consensus_kwargs = {
            "min_frame_fraction": self.config.min_frame_fraction,
            "cluster_eps": self.config.cluster_eps,
        }
        consensus_verticals = consensus_lines(
            vertical_lines_per_frame, **consensus_kwargs
        )
        consensus_horizontals = consensus_lines(
            horizontal_lines_per_frame, **consensus_kwargs
        )

        self.consensus_verticals = consensus_verticals
        self.consensus_horizontals = consensus_horizontals

        if session_court is None:
            logger.warning("No sampled frame produced a usable court mask")
            return None

        logger.debug(
            "Solving on sampled frame %d of %d (least occluded court mask)",
            best_index + 1,
            len(sampled_images),
        )

        if not session_court.solve_homography_from_lines(
            consensus_verticals, consensus_horizontals
        ):
            logger.warning(
                "Could not solve a reliable court homography from consensus lines "
                "(%d vertical / %d horizontal lines persisted across %d sampled frames)",
                len(consensus_verticals),
                len(consensus_horizontals),
                len(sampled_images),
            )
            return None

        self.solved_on_index = best_index

        logger.info(
            "Session court frozen (segmentation IoU %.3f)",
            session_court.homography_iou,
        )
        return session_court
