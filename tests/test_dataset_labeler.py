"""
The labelling session: what a keypress does to the data.

The window is not tested — a cv2 loop needs a screen — so everything that
decides what gets written lives in Session and the row builders, and that is
what these cover.
"""

import pandas as pd
import pytest
from test_dataset_landing import court_points, to_image

from sokil.dataset.labeler import (
    Session,
    _status,
    event_rows,
    load_session,
    merge_events,
    scale_point,
)
from sokil.dataset.schema import columns


@pytest.fixture
def session() -> Session:
    return Session(clip_id="clip_a", court_type="doubles", annotator="oleh")


def to_court_stub(x_px, y_px):
    """A court where one pixel is one centimetre, to keep expectations readable."""
    return x_px / 100.0, y_px / 100.0


class TestSession:
    def test_marking_a_frame_starts_an_unknown_call(self, session):
        landing = session.mark(250)

        assert landing.frame == 250
        assert landing.call == "unknown"

    def test_calling_a_frame_marks_it_too(self, session):
        session.call(250, "out")

        assert session.landings[250].call == "out"

    def test_a_click_places_the_landing(self, session):
        session.place(250, 812.5, 604.25)

        assert (session.landings[250].x_px, session.landings[250].y_px) == (
            812.5,
            604.25,
        )

    def test_a_second_click_moves_it_rather_than_adding_one(self, session):
        session.place(250, 10, 10)
        session.place(250, 20, 20)

        assert len(session.landings) == 1
        assert session.landings[250].x_px == 20

    def test_a_call_survives_being_re_placed(self, session):
        session.call(250, "in")
        session.place(250, 20, 20)

        assert session.landings[250].call == "in"

    def test_removing_a_frame_forgets_it(self, session):
        session.call(250, "in")
        session.remove(250)

        assert session.landings == {}

    def test_removing_a_frame_with_no_landing_is_harmless(self, session):
        session.remove(999)

        assert session.landings == {}

    def test_the_court_type_toggles(self, session):
        assert session.toggle_court_type() == "singles"
        assert session.toggle_court_type() == "doubles"


class TestEventRows:
    def test_derives_the_position_in_metres_from_the_click(self, session):
        session.place(250, 305.0, 670.0)
        session.call(250, "in")

        row = event_rows(session, to_court_stub)[0]

        assert (row["x_m"], row["y_m"]) == (3.05, 6.7)
        assert (row["x_px"], row["y_px"]) == (305.0, 670.0)

    def test_derives_which_line_the_landing_was_judged_on(self, session):
        session.place(250, 605.0, 670.0)

        assert event_rows(session, to_court_stub)[0]["line"] == "sideline"

    def test_keeps_the_call_the_annotator_made(self, session):
        # Geometry would say out; the human said in. The human wins, and the
        # window is where the disagreement gets shown.
        session.place(250, 900.0, 670.0)
        session.call(250, "in")

        assert event_rows(session, to_court_stub)[0]["call"] == "in"

    def test_a_landing_with_no_click_still_records_the_call(self, session):
        session.call(250, "unknown")

        row = event_rows(session, to_court_stub)[0]

        assert row["call"] == "unknown"
        assert row["x_px"] is None
        assert row["x_m"] is None

    def test_without_a_court_there_are_no_metres(self, session):
        session.place(250, 305.0, 670.0)

        row = event_rows(session, None)[0]

        assert row["x_px"] == 305.0
        assert row["x_m"] is None
        assert row["line"] is None

    def test_rows_come_out_in_frame_order(self, session):
        for frame in (900, 250, 500):
            session.mark(frame)

        assert [row["frame"] for row in event_rows(session)] == [250, 500, 900]

    def test_every_declared_column_is_written(self, session):
        session.mark(250)

        assert set(event_rows(session)[0]) == set(columns("events"))

    def test_uses_the_real_court_when_given_one(self, session):
        from sokil.dataset.landing import image_to_court

        to_court = image_to_court(court_points(), "clip_a", 0)
        session.place(250, *to_image(305.0, 670.0))

        row = event_rows(session, to_court)[0]

        assert row["x_m"] == pytest.approx(3.05, abs=0.01)
        assert row["y_m"] == pytest.approx(6.70, abs=0.01)


class TestMerge:
    def existing(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    **{c: None for c in columns("events")},
                    "clip_id": "clip_b",
                    "frame": 10,
                    "call": "in",
                    "annotator": "oleh",
                },
            ],
            columns=columns("events"),
        )

    def test_leaves_other_clips_alone(self, session):
        session.call(250, "out")

        merged = merge_events(self.existing(), "clip_a", event_rows(session))

        assert sorted(merged["clip_id"]) == ["clip_a", "clip_b"]

    def test_replaces_the_clip_being_labelled(self):
        existing = self.existing()
        session = Session(clip_id="clip_b", annotator="oleh")
        session.call(99, "out")

        merged = merge_events(existing, "clip_b", event_rows(session))

        assert list(merged["frame"]) == [99]

    def test_a_clip_labelled_to_nothing_loses_its_rows(self):
        merged = merge_events(self.existing(), "clip_b", [])

        assert merged.empty


class TestResume:
    def test_reads_back_what_was_labelled_before(self, session):
        session.place(250, 812.5, 604.25)
        session.call(250, "out")
        events = merge_events(
            pd.DataFrame(columns=columns("events")),
            "clip_a",
            event_rows(session, to_court_stub),
        )

        resumed = load_session(events, "clip_a")

        assert resumed.landings[250].call == "out"
        assert resumed.landings[250].x_px == 812.5
        assert resumed.court_type == "doubles"

    def test_an_unlabelled_clip_starts_empty(self):
        events = pd.DataFrame(columns=columns("events"))

        assert load_session(events, "clip_a").landings == {}


class TestClickScaling:
    """
    A click arrives in window coordinates, and the window is resizable. Without
    scaling, clicking on an enlarged window stores a landing off the frame — so
    nothing is drawn where the click was and the position is wrong.
    """

    def test_an_unresized_window_maps_one_to_one(self):
        assert scale_point(320, 240, (640, 480), (640, 480)) == (320, 240)

    def test_an_enlarged_window_maps_back_down(self):
        assert scale_point(640, 480, (1280, 960), (640, 480)) == (320, 240)

    def test_a_shrunken_window_maps_back_up(self):
        assert scale_point(160, 120, (320, 240), (640, 480)) == (320, 240)

    def test_a_click_past_the_edge_is_rejected(self):
        assert scale_point(1279, 100, (640, 480), (640, 480)) is None

    def test_a_click_before_the_origin_is_rejected(self):
        assert scale_point(-5, 100, (640, 480), (640, 480)) is None

    def test_the_far_corner_is_still_on_the_frame(self):
        assert scale_point(639, 479, (640, 480), (640, 480)) == (639, 479)

    def test_a_window_with_no_size_yet_is_not_guessed_at(self):
        assert scale_point(10, 10, (0, 0), (640, 480)) is None

    def test_a_stretched_window_scales_each_axis_on_its_own(self):
        assert scale_point(640, 240, (1280, 480), (640, 480)) == (320, 240)


class TestStatusLine:
    """
    What the window says about a landing. Unknown is a verdict the footage does
    not support, not a conflict with the geometry, so it must not be reported
    as a disagreement — a warning that is always on tells you nothing.
    """

    def text(self, session, landing):
        return " | ".join(text for text, _ in _status(session, landing, to_court_stub))

    def test_an_uncalled_landing_is_not_a_disagreement(self, session):
        session.place(250, 305.0, 670.0)

        assert "disagrees" not in self.text(session, session.landings[250])

    def test_an_uncalled_landing_is_prompted_for_a_call(self, session):
        session.place(250, 305.0, 670.0)

        assert "press i / o / u" in self.text(session, session.landings[250])

    def test_a_call_matching_the_geometry_is_not_flagged(self, session):
        session.place(250, 305.0, 670.0)  # 3.05 m, 6.70 m: in
        session.call(250, "in")

        assert "disagrees" not in self.text(session, session.landings[250])

    def test_a_call_against_the_geometry_is_flagged(self, session):
        session.place(250, 305.0, 670.0)  # in
        session.call(250, "out")

        assert "disagrees" in self.text(session, session.landings[250])

    def test_calling_out_what_the_geometry_calls_out_agrees(self, session):
        session.place(250, 900.0, 670.0)  # 9.00 m: past the sideline
        session.call(250, "out")

        assert "disagrees" not in self.text(session, session.landings[250])

    def test_an_unclicked_landing_asks_for_the_click(self, session):
        session.mark(250)

        assert "click where the shuttle landed" in self.text(
            session, session.landings[250]
        )

    def test_no_landing_says_so(self, session):
        assert "no landing on this frame" in self.text(session, None)
