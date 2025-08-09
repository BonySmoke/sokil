"""
ShuttleTracker: the per-frame detection pass. The detector is replaced by a
scripted fake, so what is tested is the motion the tracker measures from a
known sequence of boxes.
"""

import numpy as np
import pytest

from sokil import tracker as tracker_module
from sokil.tracker import ShuttleTracker, TrackerConfig
from sokil.video import DecodedFrame

FPS = 30.0
WIDTH, HEIGHT = 160, 120


class ScriptedDetector:
    """Returns the box scripted for each frame, in order; None means no shuttle."""

    def __init__(self, model, confidence=None, device=None):
        self.model = model
        self.confidence = confidence
        self.device = device
        self.boxes = []
        self.seen = []

    def best_box(self, image, confidence=None):
        self.seen.append(image)
        index = len(self.seen) - 1
        if index >= len(self.boxes) or self.boxes[index] is None:
            return None
        return self.boxes[index], 0.9


class FakeSource:
    def __init__(self, count: int, moving: bool = True):
        self.frame_count = count
        self.fps = FPS
        self._count = count
        self._moving = moving

    def __iter__(self):
        for number in range(self._count):
            image = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
            if self._moving:
                # a bright blob that walks across the frame, so the cork
                # estimator has something to separate from the background
                x = 20 + number * 4
                image[50:60, x : x + 10] = 240
            yield DecodedFrame(
                number=number,
                image=image,
                model_input=image,
                timestamp=round(number / FPS, 2),
            )

    def describe(self):
        return "fake source"


@pytest.fixture
def tracker(monkeypatch):
    monkeypatch.setattr(tracker_module, "ShuttleDetector", ScriptedDetector)
    return ShuttleTracker({"model": "weights.pt"})


class TestConstruction:
    def test_the_detector_gets_the_configured_device_and_confidence(self, monkeypatch):
        monkeypatch.setattr(tracker_module, "ShuttleDetector", ScriptedDetector)
        config = TrackerConfig(device="cuda:0", confidence=0.7)

        tracker = ShuttleTracker({"model": "weights.pt"}, config)

        assert tracker.detector.device == "cuda:0"
        assert tracker.detector.confidence == 0.7

    def test_the_kalman_filter_is_set_up_for_position_and_velocity(self, tracker):
        kalman = tracker._new_kalman()

        assert kalman.measurementMatrix.shape == (2, 4)
        assert kalman.transitionMatrix.shape == (4, 4)


class TestTracking:
    def boxes_for(self, count, step=4):
        # a little wider than the blob itself, the way a detection is, so the
        # box contains the blob's edge and the blur metric has detail to measure
        return [(16 + i * step, 46, 34 + i * step, 64) for i in range(count)]

    def test_one_track_per_frame_the_shuttle_was_found_on(self, tracker):
        tracker.detector.boxes = self.boxes_for(6)

        tracks = tracker.track(FakeSource(6))

        assert len(tracks) == 6
        assert [track.number for track in tracks] == list(range(6))

    def test_frames_without_a_detection_are_skipped(self, tracker):
        tracker.detector.boxes = [
            self.boxes_for(6)[0],
            None,
            None,
            self.boxes_for(6)[3],
        ]

        tracks = tracker.track(FakeSource(4))

        assert [track.number for track in tracks] == [0, 3]

    def test_a_clip_with_no_shuttle_at_all_tracks_nothing(self, tracker):
        tracker.detector.boxes = [None] * 5

        assert tracker.track(FakeSource(5)) == []

    def test_the_detector_sees_the_model_input_of_every_frame(self, tracker):
        tracker.detector.boxes = self.boxes_for(5)

        tracker.track(FakeSource(5))

        assert len(tracker.detector.seen) == 5

    def test_the_box_is_recorded_as_integers(self, tracker):
        tracker.detector.boxes = [(20.7, 50.2, 30.9, 60.4)]

        (track,) = tracker.track(FakeSource(1))

        assert track.xyxy == (20, 50, 30, 60)

    def test_the_timestamp_comes_from_the_decoded_frame(self, tracker):
        tracker.detector.boxes = self.boxes_for(3)

        tracks = tracker.track(FakeSource(3))

        assert [track.timestamp for track in tracks] == [0.0, 0.03, 0.07]

    def test_a_steady_flight_settles_on_its_true_velocity(self, tracker):
        tracker.detector.boxes = self.boxes_for(20, step=4)

        tracks = tracker.track(FakeSource(20))

        assert tracks[-1].velocity == pytest.approx((4.0, 0.0), abs=0.1)
        assert tracks[-1].speed > 0

    def test_the_first_two_frames_have_no_turn_to_measure(self, tracker):
        tracker.detector.boxes = self.boxes_for(4)

        tracks = tracker.track(FakeSource(4))

        assert tracks[0].angle_change == 0.0
        assert tracks[1].angle_change == 0.0

    def test_a_turn_in_the_flight_is_measured(self, tracker):
        tracker.detector.boxes = [
            (20, 50, 30, 60),
            (40, 50, 50, 60),
            (60, 50, 70, 60),
            (60, 90, 70, 100),  # a sharp turn downward
        ]

        tracks = tracker.track(FakeSource(4))

        assert abs(tracks[-1].angle_change) == pytest.approx(90.0, abs=1.0)

    def test_the_blur_is_measured_from_the_pixels(self, tracker):
        tracker.detector.boxes = self.boxes_for(4)

        tracks = tracker.track(FakeSource(4))

        # the synthetic blob has a hard edge inside the box
        assert tracks[-1].blur > 0

    def test_the_cork_is_measured_once_there_is_enough_history(self, tracker):
        tracker.detector.boxes = self.boxes_for(8)

        tracks = tracker.track(FakeSource(8))

        assert tracks[0].measured_cork is None
        assert any(track.measured_cork is not None for track in tracks[4:])
