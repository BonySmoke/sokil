"""
Drawing the court: the top-down overview and the umpire close-up of a landing.

Both build a matplotlib figure, so the tests assert on the figure's own
geometry — the window it framed, the patches it drew — rather than on pixels.
"""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from sokil.court import (
    CourtCameraMode,
    CourtModel,
    CourtModelType,
    draw_badminton_court_matplotlib,
    draw_hit_closeup_matplotlib,
)


@pytest.fixture
def model() -> CourtModel:
    return CourtModel()


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TestCourtOverview:
    def test_draws_the_whole_court_with_a_margin(self, model):
        padding = 50

        _, ax = draw_badminton_court_matplotlib(model, padding=padding, dpi=50)

        assert ax.get_xlim() == pytest.approx((-padding, model.width + padding))

    def test_the_shuttle_is_marked_where_it_landed(self, model):
        _, ax = draw_badminton_court_matplotlib(
            model, shuttle_position=(300, 600), is_shuttle_in=True, dpi=50
        )

        centres = [
            patch.get_center() for patch in ax.patches if hasattr(patch, "get_center")
        ]
        assert (300.0, 600.0) in centres

    def test_the_verdict_is_labelled_on_the_zoomed_inset(self, model):
        def verdicts(ax):
            # the label lives on the zoomed inset, a child of the court axes
            return {
                text.get_text()
                for inset in ax.child_axes
                for text in inset.texts
                if text.get_text() in {"IN", "OUT"}
            }

        _, inside = draw_badminton_court_matplotlib(
            model, shuttle_position=(300, 600), is_shuttle_in=True, dpi=50
        )
        _, outside = draw_badminton_court_matplotlib(
            model, shuttle_position=(300, 600), is_shuttle_in=False, dpi=50
        )
        _, undecided = draw_badminton_court_matplotlib(
            model, shuttle_position=(300, 600), dpi=50
        )

        assert verdicts(inside) == {"IN"}
        assert verdicts(outside) == {"OUT"}
        assert verdicts(undecided) == set()

    def test_a_corridor_view_opens_out_toward_the_court_it_cannot_see(self, model):
        corridor = CourtModel(court_camera_mode=CourtCameraMode.LEFT_CORRIDOR)

        _, corridor_ax = draw_badminton_court_matplotlib(corridor, dpi=50)

        # the right side is unwatched, so the view extends past the corridor
        assert corridor_ax.get_xlim()[1] > corridor.side_corridor_width + 50

    def test_the_figure_takes_the_courts_own_aspect(self, model):
        fig, _ = draw_badminton_court_matplotlib(model, figsize=(10, 10), dpi=50)

        width_in, height_in = fig.get_size_inches()

        assert width_in / height_in == pytest.approx(
            (model.width + 100) / (model.length + 100), rel=0.01
        )

    def test_a_singles_court_can_be_drawn_too(self):
        singles = CourtModel(court_model_type=CourtModelType.SINGLES)

        _, ax = draw_badminton_court_matplotlib(singles, dpi=50)

        assert ax.lines or ax.patches


class TestHitCloseup:
    def test_the_window_is_centred_on_the_shuttle(self, model):
        _, ax = draw_hit_closeup_matplotlib(model, (300, 600), dpi=50)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        assert (xmin + xmax) / 2 == pytest.approx(300.0)
        assert (min(ymin, ymax) + max(ymin, ymax)) / 2 == pytest.approx(600.0)

    def test_a_landing_near_a_line_zooms_in_tight(self, model):
        _, near = draw_hit_closeup_matplotlib(model, (5, 600), dpi=50)
        _, far = draw_hit_closeup_matplotlib(model, (300, 600), dpi=50)

        def height(ax):
            low, high = ax.get_ylim()
            return abs(high - low)

        assert height(near) < height(far)

    def test_the_window_never_shrinks_below_the_minimum(self, model):
        _, ax = draw_hit_closeup_matplotlib(
            model, (0, 600), dpi=50, min_half_window=15, margin=0
        )

        low, high = ax.get_ylim()
        assert abs(high - low) >= 2 * 15

    def test_the_window_never_grows_past_the_maximum(self, model):
        _, ax = draw_hit_closeup_matplotlib(
            model, (305, 670), dpi=50, max_half_window=100
        )

        low, high = ax.get_ylim()
        assert abs(high - low) <= 2 * 100

    def test_the_out_region_is_shaded_beside_the_line(self, model):
        _, ax = draw_hit_closeup_matplotlib(model, (5, 600), dpi=50)

        assert ax.patches

    def test_the_window_matches_the_output_aspect(self, model):
        _, ax = draw_hit_closeup_matplotlib(model, (300, 600), figsize=(16, 9), dpi=50)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        assert abs(xmax - xmin) / abs(ymax - ymin) == pytest.approx(16 / 9, rel=0.01)

    def test_a_verdict_is_written_onto_the_close_up(self, model):
        _, inside = draw_hit_closeup_matplotlib(
            model, (300, 600), is_shuttle_in=True, dpi=50
        )
        _, outside = draw_hit_closeup_matplotlib(
            model, (-20, 600), is_shuttle_in=False, dpi=50
        )

        assert [text.get_text() for text in inside.texts] != [
            text.get_text() for text in outside.texts
        ]

    def test_a_corridor_camera_only_shades_the_lines_it_watches(self):
        corridor = CourtModel(court_camera_mode=CourtCameraMode.LEFT_CORRIDOR)

        _, ax = draw_hit_closeup_matplotlib(corridor, (300, 600), dpi=50)

        assert ax.get_xlim()[0] < 300


class TestCourtPlot:
    def test_a_court_plots_the_landing_through_its_homography(self, identity_court):
        fig, ax = identity_court.plot(shuttle_position=(300, 600), is_shuttle_in=True)

        assert fig is not None
        xmin, xmax = ax.get_xlim()
        assert (xmin + xmax) / 2 == pytest.approx(300.0)

    def test_a_court_with_no_homography_plots_nothing(self, frame):
        from sokil.court import Court

        class NoMask:
            def polygons(self, image, confidence=None):
                return []

        court = Court(image=frame, segmentation_model=NoMask())

        assert court.plot(shuttle_position=(10, 10)) == (None, None)


class TestCourtRectangle:
    def test_finds_the_four_outer_corners_of_a_painted_court(self):
        from sokil.court import Court

        width, height, floor, paint = 400, 300, 40, 235
        image = np.full((height, width, 3), floor, dtype=np.uint8)
        for x in (60, 330):
            image[:, x : x + 6] = paint
        for y in (50, 250):
            image[y : y + 6, :] = paint

        class FullMask:
            def polygons(self, img, confidence=None):
                h, w = img.shape[:2]
                return [
                    np.array(
                        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                        dtype=np.float32,
                    )
                ]

        corners = Court(
            image=image, segmentation_model=FullMask()
        ).get_court_rectangle()

        assert corners is not None
        assert corners.reshape(-1, 2).shape == (4, 2)
        xs = sorted({int(x) for x, _ in corners.reshape(-1, 2)})
        assert xs[0] == pytest.approx(60, abs=8)
        assert xs[-1] == pytest.approx(336, abs=8)

    def test_a_frame_with_no_mask_has_no_rectangle(self, frame):
        from sokil.court import Court

        class NoMask:
            def polygons(self, image, confidence=None):
                return []

        assert (
            Court(image=frame, segmentation_model=NoMask()).get_court_rectangle()
            is None
        )
