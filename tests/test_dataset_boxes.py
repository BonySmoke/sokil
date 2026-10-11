"""
Driving the tracker from annotated boxes rather than a detector.
"""

import pandas as pd
import pytest
from test_tracker import FakeSource

from sokil.dataset.boxes import ShuttleBoxSource
from sokil.tracker import ShuttleTracker
from sokil.video import DecodedFrame


def frame(number: int) -> DecodedFrame:
    import numpy as np

    blank = np.zeros((96, 128, 3), np.uint8)
    return DecodedFrame(
        number=number, image=blank, model_input=blank, timestamp=number / 120
    )


def tracks_for(clip_id: str, boxes: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "clip_id": clip_id,
                "frame": f,
                "x1": b[0],
                "y1": b[1],
                "x2": b[2],
                "y2": b[3],
            }
            for f, b in boxes.items()
        ]
    )


class TestShuttleBoxSource:
    def test_hands_back_the_annotated_box(self):
        source = ShuttleBoxSource(tracks_for("clip_a", {5: (10, 20, 30, 40)}), "clip_a")

        assert source.best_box(frame=5) == ((10.0, 20.0, 30.0, 40.0), 1.0)

    def test_the_pixels_are_never_consulted(self):
        # It answers with no image at all, which is the point: the box did not
        # come from looking at anything.
        source = ShuttleBoxSource(tracks_for("clip_a", {5: (10, 20, 30, 40)}), "clip_a")

        assert source.best_box(image=None, frame=5) is not None

    def test_a_frame_with_no_row_has_no_box(self):
        source = ShuttleBoxSource(tracks_for("clip_a", {5: (10, 20, 30, 40)}), "clip_a")

        assert source.best_box(frame=0) is None

    def test_being_asked_without_a_frame_is_a_caller_s_mistake(self):
        # Not a frame with no shuttle: a caller that never passed the metadata.
        source = ShuttleBoxSource(tracks_for("clip_a", {5: (10, 20, 30, 40)}), "clip_a")

        with pytest.raises(ValueError, match="needs the frame number"):
            source.best_box(image=None)

    def test_only_the_clip_asked_for_is_loaded(self):
        tracks = pd.concat(
            [
                tracks_for("clip_a", {0: (1, 2, 3, 4)}),
                tracks_for("clip_b", {0: (9, 9, 9, 9), 1: (9, 9, 9, 9)}),
            ]
        )

        assert len(ShuttleBoxSource(tracks, "clip_a")) == 1

    def test_a_clip_with_no_track_says_so(self):
        with pytest.raises(ValueError, match="No ground-truth track"):
            ShuttleBoxSource(tracks_for("clip_a", {0: (1, 2, 3, 4)}), "clip_b")

    def test_it_drives_the_tracker(self):
        tracks = tracks_for(
            "clip_a", {i: (16 + i * 4, 46, 34 + i * 4, 64) for i in range(6)}
        )

        result = ShuttleTracker(detector=ShuttleBoxSource(tracks, "clip_a")).track(
            FakeSource(6)
        )

        assert [t.number for t in result] == list(range(6))
        assert all(t.blur > 0 for t in result)
