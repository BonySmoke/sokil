"""
Turning a clicked landing into a position on the court, and a call against it.

The unit change is the trap here: the court model measures in centimetres and
the dataset records metres, so an error of a hundred looks entirely plausible
right up until every landing is out.
"""

import numpy as np
import pandas as pd
import pytest

from sokil.court import CourtModel
from sokil.dataset.landing import (
    boundary,
    court_keyframe,
    derive_call,
    derive_line,
    derive_positions,
    image_to_court,
)

HOMOGRAPHY = np.array([[1.6, 0.35, 240.0], [0.0, 0.55, 120.0], [0.0, 0.00045, 1.0]])
VERTICES = [1, 5, 26, 30, 11, 15, 3, 28]


def to_image(x_cm: float, y_cm: float) -> tuple[float, float]:
    projected = HOMOGRAPHY @ np.array([x_cm, y_cm, 1.0])
    return tuple(projected[:2] / projected[2])


def court_points(clip_id: str = "clip_a", frames: tuple[int, ...] = (0,)):
    model = CourtModel().vertices
    rows = []
    for frame in frames:
        for vertex in VERTICES:
            x_px, y_px = to_image(*model[vertex - 1])
            rows.append(
                {
                    "clip_id": clip_id,
                    "frame": frame,
                    "vertex": vertex,
                    "x_px": x_px,
                    "y_px": y_px,
                    "visible": True,
                    "source": "manual",
                }
            )
    return pd.DataFrame(rows)


class TestBoundary:
    def test_doubles_is_the_full_court_in_metres(self):
        assert boundary("doubles") == (0.0, 6.1, 0.0, 13.4)

    def test_singles_is_narrower_but_just_as_long(self):
        x_min, x_max, y_min, y_max = boundary("singles")

        assert round(x_max - x_min, 2) == 5.18
        assert (y_min, y_max) == (0.0, 13.4)


class TestCall:
    def test_the_middle_of_the_court_is_in(self):
        assert derive_call(3.05, 6.7, "doubles") == "in"

    def test_past_the_sideline_is_out(self):
        assert derive_call(6.2, 6.7, "doubles") == "out"

    def test_the_doubles_alley_is_out_for_singles(self):
        # The same landing, judged against two boundaries: this is why a
        # verdict without a court_type cannot be read.
        assert derive_call(0.2, 6.7, "doubles") == "in"
        assert derive_call(0.2, 6.7, "singles") == "out"

    def test_a_shuttle_on_the_line_is_in(self):
        assert derive_call(6.1, 6.7, "doubles") == "in"


class TestLine:
    def test_a_landing_near_the_side_is_judged_on_the_sideline(self):
        assert derive_line(6.05, 6.7, "doubles") == "sideline"

    def test_a_landing_near_the_back_is_judged_on_the_baseline(self):
        assert derive_line(3.05, 13.3, "doubles") == "baseline"

    def test_the_singles_sideline_is_the_inner_one(self):
        # 0.5 m is inside the doubles court but a few centimetres from the
        # singles sideline, so which line it is judged on depends on the type.
        assert derive_line(0.5, 6.7, "singles") == "sideline"


class TestImageToCourt:
    def test_a_click_comes_back_as_the_court_position_it_came_from(self):
        to_court = image_to_court(court_points(), "clip_a", 0)

        x_m, y_m = to_court(*to_image(305.0, 670.0))

        assert x_m == pytest.approx(3.05, abs=0.01)
        assert y_m == pytest.approx(6.70, abs=0.01)

    def test_the_court_corners_land_on_zero_and_the_court_size(self):
        to_court = image_to_court(court_points(), "clip_a", 0)
        model = CourtModel()

        far = to_court(*to_image(model.width, model.length))

        assert far[0] == pytest.approx(6.10, abs=0.01)
        assert far[1] == pytest.approx(13.40, abs=0.01)

    def test_a_clip_with_no_keypoints_has_no_transform(self):
        assert image_to_court(court_points(), "clip_ghost", 0) is None

    def test_points_that_place_no_court_have_no_transform(self):
        points = court_points()
        points.loc[:, "visible"] = False

        assert image_to_court(points, "clip_a", 0) is None


class TestKeyframeChoice:
    def test_uses_the_keyframe_in_force_at_the_frame(self):
        points = court_points(frames=(0, 500, 900))

        chosen = court_keyframe(points, "clip_a", 700)

        assert set(chosen["frame"]) == {500}

    def test_a_landing_before_the_first_keyframe_uses_it_anyway(self):
        points = court_points(frames=(300, 900))

        chosen = court_keyframe(points, "clip_a", 10)

        assert set(chosen["frame"]) == {300}

    def test_no_keyframes_for_a_clip_is_no_court(self):
        assert court_keyframe(court_points(), "clip_ghost", 0) is None


class TestDerivePositions:
    """
    Keeping derived positions in step with the court they came from.

    x_m and y_m are worked out from the click, so storing them once and never
    refreshing them is what lets a landing go quietly stale: clicked before its
    court existed, or derived through a court that has since been corrected.
    """

    def events_for(self, clip_id: str, rows: list[dict]):
        from sokil.dataset.schema import columns

        return pd.DataFrame(
            [
                {
                    **{column: None for column in columns("events")},
                    "clip_id": clip_id,
                    "court_type": "doubles",
                    "call": "in",
                    "annotator": "test",
                    **row,
                }
                for row in rows
            ],
            columns=columns("events"),
        )

    def test_a_landing_with_no_position_gets_one(self):
        x_px, y_px = to_image(305.0, 670.0)
        events = self.events_for(
            "clip_a", [{"frame": 10, "x_px": x_px, "y_px": y_px, "x_m": None}]
        )

        refreshed, changed = derive_positions(events, court_points())

        assert changed == 1
        assert refreshed.iloc[0]["x_m"] == pytest.approx(3.05, abs=0.01)

    def test_a_landing_derived_through_an_old_court_is_corrected(self):
        x_px, y_px = to_image(305.0, 670.0)
        events = self.events_for(
            "clip_a",
            [{"frame": 10, "x_px": x_px, "y_px": y_px, "x_m": 99.0, "y_m": 99.0}],
        )

        refreshed, changed = derive_positions(events, court_points())

        assert changed == 1
        assert refreshed.iloc[0]["y_m"] == pytest.approx(6.70, abs=0.01)

    def test_a_landing_already_right_is_left_alone(self):
        x_px, y_px = to_image(305.0, 670.0)
        events = self.events_for("clip_a", [{"frame": 10, "x_px": x_px, "y_px": y_px}])
        once, _ = derive_positions(events, court_points())

        twice, changed = derive_positions(once, court_points())

        assert changed == 0
        assert twice.iloc[0]["x_m"] == once.iloc[0]["x_m"]

    def test_the_click_is_never_touched(self):
        # The click is the observation, and the only part a human made.
        x_px, y_px = to_image(305.0, 670.0)
        events = self.events_for("clip_a", [{"frame": 10, "x_px": x_px, "y_px": y_px}])

        refreshed, _ = derive_positions(events, court_points())

        assert refreshed.iloc[0]["x_px"] == x_px
        assert refreshed.iloc[0]["y_px"] == y_px

    def test_the_line_is_re_derived_too(self):
        x_px, y_px = to_image(605.0, 670.0)
        events = self.events_for(
            "clip_a", [{"frame": 10, "x_px": x_px, "y_px": y_px, "line": "baseline"}]
        )

        refreshed, _ = derive_positions(events, court_points())

        assert refreshed.iloc[0]["line"] == "sideline"

    def test_a_clip_with_no_court_is_left_as_it_is(self):
        events = self.events_for(
            "clip_ghost", [{"frame": 10, "x_px": 100.0, "y_px": 100.0}]
        )

        refreshed, changed = derive_positions(events, court_points())

        assert changed == 0
        assert pd.isna(refreshed.iloc[0]["x_m"])

    def test_a_landing_with_no_click_cannot_be_placed(self):
        events = self.events_for("clip_a", [{"frame": 10, "x_px": None, "y_px": None}])

        refreshed, changed = derive_positions(events, court_points())

        assert changed == 0
        assert pd.isna(refreshed.iloc[0]["x_m"])

    def test_other_clips_are_not_disturbed(self):
        x_px, y_px = to_image(305.0, 670.0)
        events = pd.concat(
            [
                self.events_for("clip_a", [{"frame": 1, "x_px": x_px, "y_px": y_px}]),
                self.events_for("clip_b", [{"frame": 2, "x_px": 50.0, "y_px": 50.0}]),
            ],
            ignore_index=True,
        )

        refreshed, _ = derive_positions(events, court_points())

        assert pd.isna(refreshed[refreshed["clip_id"] == "clip_b"].iloc[0]["x_m"])
