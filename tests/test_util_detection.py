"""
Finding the court lines in an actual picture: the segment detector, the
clustering that collapses one painted stripe into one line, and the paint
contrast that tells a painted line from a mat seam.

The picture is drawn here — a dark floor with white stripes on it — so the
tests need no recorded frame and still run the real OpenCV code.
"""

import cv2
import numpy as np
import pytest

from sokil.court import Court, CourtConfig
from sokil.util import (
    cluster_parallel_lines,
    detect_line_segments,
    keep_painted_lines,
    line_paint_contrast,
    split_line_families,
)

WIDTH, HEIGHT = 400, 300
FLOOR = 40
PAINT = 235


# Wider than the DBSCAN eps that clusters line edges (12 px) and narrower than
# the stripe-merge threshold (25 px), so each stripe is detected as its own two
# painted borders and then merged back into one centreline — which is exactly
# what happens on a real frame where the court is near the camera.
STRIPE = 18


def court_image(vertical_xs=(80, 300), horizontal_ys=(70, 230), stripe=STRIPE):
    """A dark floor with bright painted stripes at the given positions."""
    image = np.full((HEIGHT, WIDTH, 3), FLOOR, dtype=np.uint8)
    for x in vertical_xs:
        image[:, x : x + stripe] = PAINT
    for y in horizontal_ys:
        image[y : y + stripe, :] = PAINT
    return image


def edges_of(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)


@pytest.fixture
def painted() -> np.ndarray:
    return court_image()


class TestDetectLineSegments:
    def test_finds_the_painted_stripes(self, painted):
        segments = detect_line_segments(edges_of(painted))

        assert segments
        assert {"rho", "theta", "length", "points"} <= set(segments[0])

    def test_both_orientations_are_present(self, painted):
        segments = detect_line_segments(edges_of(painted))

        vertical, horizontal = split_line_families(segments)

        assert vertical
        assert horizontal

    def test_an_empty_edge_image_yields_nothing(self):
        assert detect_line_segments(np.zeros((HEIGHT, WIDTH), np.uint8)) == []

    def test_no_edge_image_at_all_yields_nothing(self):
        assert detect_line_segments(None) == []

    def test_a_line_shorter_than_the_minimum_is_ignored(self):
        image = np.full((HEIGHT, WIDTH, 3), FLOOR, dtype=np.uint8)
        image[100:118, 10:25] = PAINT  # a 15 px stub

        assert detect_line_segments(edges_of(image), min_line_length_ratio=0.5) == []


class TestClusterParallelLines:
    def cluster(self, image, vertical=True):
        segments = detect_line_segments(edges_of(image))
        families = split_line_families(segments)
        family = families[0] if vertical else families[1]
        return cluster_parallel_lines(family, image, vertical=vertical)

    def test_one_painted_stripe_becomes_one_line(self):
        image = court_image(vertical_xs=(200,), horizontal_ys=())

        clustered = self.cluster(image)

        assert len(clustered) == 1

    def test_the_line_sits_on_the_middle_of_the_stripe(self):
        image = court_image(vertical_xs=(200,), horizontal_ys=())

        (line,) = self.cluster(image)

        assert line["ref"] == pytest.approx(200 + STRIPE / 2, abs=2.0)

    def test_two_distinct_stripes_stay_two_lines(self):
        image = court_image(vertical_xs=(80, 300), horizontal_ys=())

        clustered = self.cluster(image)

        assert len(clustered) == 2

    def test_the_result_is_ordered_by_position(self):
        image = court_image(vertical_xs=(80, 200, 300), horizontal_ys=())

        refs = [line["ref"] for line in self.cluster(image)]

        assert refs == sorted(refs)

    def test_horizontal_stripes_cluster_the_same_way(self):
        image = court_image(vertical_xs=(), horizontal_ys=(70, 230))

        assert len(self.cluster(image, vertical=False)) == 2

    def test_nothing_detected_clusters_to_nothing(self, painted):
        assert cluster_parallel_lines([], painted, vertical=True) == []


class TestPaintContrast:
    def line_at(self, x: float) -> dict:
        # theta = 0 is a vertical line at x = rho
        return {"rho": float(x), "theta": 0.0, "length": float(HEIGHT), "ref": float(x)}

    def test_a_painted_line_stands_out_from_the_floor(self):
        gray = cv2.cvtColor(
            court_image(vertical_xs=(200,), horizontal_ys=()), cv2.COLOR_BGR2GRAY
        )

        peak, contrast = line_paint_contrast(self.line_at(203), gray)

        assert peak == pytest.approx(PAINT, abs=5)
        assert contrast > 100

    def test_bare_floor_has_no_paint_under_it(self):
        gray = cv2.cvtColor(
            court_image(vertical_xs=(200,), horizontal_ys=()), cv2.COLOR_BGR2GRAY
        )

        _, contrast = line_paint_contrast(self.line_at(100), gray)

        assert contrast == pytest.approx(0.0, abs=2)

    def test_a_line_that_never_crosses_the_image_has_no_measurement(self):
        gray = np.full((HEIGHT, WIDTH), FLOOR, dtype=np.uint8)

        assert line_paint_contrast(self.line_at(10_000), gray) == (None, None)


class TestKeepPaintedLines:
    def test_a_seam_beside_a_painted_line_is_dropped(self):
        image = court_image(vertical_xs=(200,), horizontal_ys=())
        # a faint step in the floor, the kind a mat join leaves
        image[:, 300:] = FLOOR + 6
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines = [
            {"rho": 203.0, "theta": 0.0, "length": 300.0, "ref": 203.0},
            {"rho": 300.0, "theta": 0.0, "length": 300.0, "ref": 300.0},
        ]

        kept = keep_painted_lines(lines, gray)

        assert [line["ref"] for line in kept] == [203.0]

    def test_a_family_of_seams_alone_is_dropped_entirely(self):
        image = np.full((HEIGHT, WIDTH, 3), FLOOR, dtype=np.uint8)
        image[:, 200:] = FLOOR + 5
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines = [{"rho": 200.0, "theta": 0.0, "length": 300.0, "ref": 200.0}]

        assert keep_painted_lines(lines, gray) == []

    def test_nothing_to_filter_passes_through(self):
        gray = np.full((HEIGHT, WIDTH), FLOOR, dtype=np.uint8)

        assert keep_painted_lines([], gray) == []


class FullMaskSegmenter:
    """A segmenter whose mask is the whole frame, so nothing is cropped away."""

    def polygons(self, image, confidence=None):
        height, width = image.shape[:2]
        return [
            np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            )
        ]


class TestCourtDetectCourtLines:
    def test_finds_both_families_on_a_painted_court(self, painted):
        court = Court(image=painted, segmentation_model=FullMaskSegmenter())

        verticals, horizontals = court.detect_court_lines()

        assert len(verticals) == 2
        assert len(horizontals) == 2

    def test_the_lines_come_back_ordered(self, painted):
        court = Court(image=painted, segmentation_model=FullMaskSegmenter())

        verticals, _ = court.detect_court_lines()

        assert [line["ref"] for line in verticals] == sorted(
            line["ref"] for line in verticals
        )

    def test_a_frame_with_no_court_mask_finds_no_lines(self, painted):
        class NoMask:
            def polygons(self, image, confidence=None):
                return []

        court = Court(image=painted, segmentation_model=NoMask())

        assert court.detect_court_lines() == ([], [])

    def test_a_blank_floor_has_no_lines_to_find(self):
        blank = np.full((HEIGHT, WIDTH, 3), FLOOR, dtype=np.uint8)
        court = Court(image=blank, segmentation_model=FullMaskSegmenter())

        verticals, horizontals = court.detect_court_lines()

        assert verticals == []
        assert horizontals == []


class TestCourtConfigClustering:
    """
    The line-clustering thresholds on CourtConfig, and the failure they are
    tuned against: Canny reports both borders of a painted stripe, so an eps
    wide enough to swallow the pair fits one line across both, which then fails
    the straightness test and drops the court line altogether.
    """

    NARROW_STRIPE = 8  # narrower than an eps of 12, wider than one of 6

    def court(self, image, **config_kwargs) -> Court:
        return Court(
            image=image,
            segmentation_model=FullMaskSegmenter(),
            court_config=CourtConfig(**config_kwargs),
        )

    def striped(self, stripe: int) -> np.ndarray:
        return court_image(vertical_xs=(200,), horizontal_ys=(), stripe=stripe)

    def test_the_defaults_are_the_tuned_pipeline_values(self):
        config = CourtConfig()

        assert config.line_cluster_eps == 6.0
        assert config.line_straightness_tau == 2.5
        assert config.stripe_merge_eps == 25.0

    def test_the_configured_thresholds_reach_the_clustering(self, monkeypatch):
        seen = []

        def spy(lines, image, vertical, **kwargs):
            seen.append(kwargs)
            return []

        monkeypatch.setattr("sokil.court.cluster_parallel_lines", spy)
        config = {
            "line_cluster_eps": 3.0,
            "line_straightness_tau": 1.5,
            "stripe_merge_eps": 40.0,
        }

        self.court(self.striped(STRIPE), **config).detect_court_lines()

        # once for each family, both with the configured values
        assert len(seen) == 2
        assert (
            seen
            == [{"eps": 3.0, "straightness_tau": 1.5, "stripe_merge_eps": 40.0}] * 2
        )

    def test_a_narrow_stripe_survives_the_default_eps(self):
        verticals, _ = self.court(self.striped(self.NARROW_STRIPE)).detect_court_lines()

        assert len(verticals) == 1
        assert verticals[0]["ref"] == pytest.approx(
            200 + self.NARROW_STRIPE / 2, abs=2.0
        )

    def test_an_eps_wider_than_the_stripe_loses_the_line_entirely(self):
        # the regression this default guards: both borders land in one cluster,
        # the fit straddles them, and the straightness test throws it away
        verticals, _ = self.court(
            self.striped(self.NARROW_STRIPE), line_cluster_eps=12.0
        ).detect_court_lines()

        assert verticals == []

    def test_the_straightness_threshold_is_what_rejects_the_swallowed_pair(self):
        # same too-wide eps, but tolerant enough to accept the straddling fit
        verticals, _ = self.court(
            self.striped(self.NARROW_STRIPE),
            line_cluster_eps=12.0,
            line_straightness_tau=8.0,
        ).detect_court_lines()

        assert len(verticals) == 1

    def test_a_stripe_wider_than_either_eps_is_found_regardless(self):
        wide = self.striped(STRIPE)

        assert len(self.court(wide).detect_court_lines()[0]) == 1
        assert len(self.court(wide, line_cluster_eps=12.0).detect_court_lines()[0]) == 1

    def test_both_families_are_clustered_with_the_same_thresholds(self):
        verticals, horizontals = self.court(court_image()).detect_court_lines()

        assert len(verticals) == 2
        assert len(horizontals) == 2
