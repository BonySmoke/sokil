"""
The --explain walkthrough: the slider transition, and which stages can be drawn
from what a landing actually has.
"""

import numpy as np
import pytest

from sokil.explanation import STAGES, Explainer, Stage, build_stages, wipe
from sokil.frame import Hit
from sokil.render import RenderConfig
from sokil.visualization import draw_caption

WIDTH, HEIGHT = 120, 90


@pytest.fixture
def before() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)


@pytest.fixture
def after() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)


class TestWipe:
    def test_no_progress_shows_the_frame_before(self, before, after):
        assert np.array_equal(wipe(before, after, 0.0), before)

    def test_full_progress_shows_the_frame_after(self, before, after):
        assert np.array_equal(wipe(before, after, 1.0), after)

    def test_halfway_splits_the_frame_down_the_middle(self, before, after):
        out = wipe(before, after, 0.5, divider_width=0)

        assert np.array_equal(out[:, : WIDTH // 2], after[:, : WIDTH // 2])
        assert np.array_equal(out[:, WIDTH // 2 :], before[:, WIDTH // 2 :])

    def test_the_divider_marks_the_split(self, before, after):
        out = wipe(before, after, 0.5, divider_width=4, divider_color=(0, 0, 255))

        column = out[HEIGHT // 2, WIDTH // 2]
        assert tuple(column) == (0, 0, 255)

    def test_progress_beyond_the_ends_is_clamped(self, before, after):
        assert np.array_equal(wipe(before, after, -3.0), before)
        assert np.array_equal(wipe(before, after, 7.0), after)

    def test_the_input_frames_are_never_modified(self, before, after):
        original = before.copy()

        wipe(before, after, 0.5)

        assert np.array_equal(before, original)


class FakeSegmenter:
    """A court segmenter that finds nothing, so the frame-local stages fail."""

    def polygons(self, image, confidence=None):
        return []


class TestBuildStages:
    def track_at(self, make_track, position=(60, 45)):
        track = make_track(
            number=12,
            measured_cork=position,
            velocity=(10.0, 10.0),
            xyxy=(55, 40, 65, 50),
        )
        track.hit = Hit(position=position, is_in=True)
        return track

    def test_without_a_frame_court_only_the_session_stages_are_drawn(
        self, identity_court, make_track
    ):
        frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)

        stages = build_stages(frame, self.track_at(make_track), identity_court)

        keys = [stage.key for stage in stages]
        assert "edges" not in keys
        assert "clusters" not in keys
        assert "raw" in keys
        assert "top_view" in keys

    def test_the_stages_keep_the_documented_order(self, identity_court, make_track):
        frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)

        stages = build_stages(frame, self.track_at(make_track), identity_court)

        order = [key for key, _, _ in STAGES]
        assert [stage.key for stage in stages] == [
            key for key in order if key in {stage.key for stage in stages}
        ]

    def test_every_stage_is_a_captioned_frame_sized_image(
        self, identity_court, make_track
    ):
        frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)

        stages = build_stages(frame, self.track_at(make_track), identity_court)

        for stage in stages:
            assert isinstance(stage, Stage)
            assert stage.image.shape == frame.shape
            assert stage.title
            # the caption is burned in, so the image is not the raw frame
            assert not np.array_equal(stage.image, frame)

    def test_a_frame_court_that_segments_nothing_skips_its_stages(
        self, identity_court, make_track, caplog
    ):
        from sokil.court import Court

        frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
        frame_court = Court(image=frame, segmentation_model=FakeSegmenter())

        stages = build_stages(
            frame, self.track_at(make_track), identity_court, frame_court=frame_court
        )

        assert "edges" not in {stage.key for stage in stages}
        assert "could not be drawn" in caplog.text


class TestExplainerFrames:
    """
    The Explainer segments the contact frame itself. There is no checkpoint
    here, so it is handed a segmenter that finds nothing: the frame-local
    stages are then skipped and the session stages carry the walkthrough.
    """

    @pytest.fixture
    def landing(self, make_track):
        frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
        track = make_track(
            number=12,
            measured_cork=(60, 45),
            velocity=(10.0, 10.0),
            xyxy=(55, 40, 65, 50),
        )
        track.hit = Hit(position=(60, 45), is_in=True)
        return frame, track

    def explainer(self, **config_kwargs) -> Explainer:
        return Explainer(
            court_detector_kwargs={"segmentation_model": FakeSegmenter()},
            config=RenderConfig(**config_kwargs),
        )

    def test_the_walkthrough_holds_each_stage_for_its_configured_beat(
        self, identity_court, landing
    ):
        frame, track = landing
        explainer = self.explainer(
            explain_freeze_seconds=1.0,
            explain_stage_seconds=2.0,
            explain_transition_seconds=0.0,
        )

        images = list(explainer.frames(frame, track, identity_court, fps=1.0))
        stages = build_stages(frame, track, identity_court)

        # the first stage is held for the freeze beat, every other for a full one
        assert len(images) == 1 + 2 * (len(stages) - 1)

    def test_a_transition_is_inserted_between_stages(self, identity_court, landing):
        frame, track = landing
        beats = {"explain_freeze_seconds": 1.0, "explain_stage_seconds": 1.0}

        with_wipe = self.explainer(explain_transition_seconds=1.0, **beats)
        without_wipe = self.explainer(explain_transition_seconds=0.0, **beats)

        assert len(list(with_wipe.frames(frame, track, identity_court, 1.0))) > len(
            list(without_wipe.frames(frame, track, identity_court, 1.0))
        )

    def test_every_frame_is_the_size_of_the_clip(self, identity_court, landing):
        frame, track = landing

        for image in self.explainer().frames(frame, track, identity_court, fps=2.0):
            assert image.shape == frame.shape

    def test_the_landing_is_explained_from_the_undrawn_frame(
        self, identity_court, landing
    ):
        frame, track = landing
        original = frame.copy()

        list(self.explainer().frames(frame, track, identity_court, fps=2.0))

        assert np.array_equal(frame, original)


class TestDrawCaption:
    def test_the_caption_is_burned_into_a_copy(self):
        image = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)

        captioned = draw_caption(image, "Title", "A caption.")

        assert captioned.shape == image.shape
        assert np.array_equal(image, np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8))
        assert not np.array_equal(captioned, image)

    def test_a_long_caption_is_wrapped_rather_than_clipped(self):
        image = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)

        short = draw_caption(image, "Title", "Short.")
        long = draw_caption(image, "Title", "A much longer caption " * 6)

        # the band grows upward to fit the wrapped lines
        assert (long != image).any(axis=2).sum() > (short != image).any(axis=2).sum()

    def test_a_title_alone_is_enough(self):
        image = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)

        assert draw_caption(image, "Title").shape == image.shape
