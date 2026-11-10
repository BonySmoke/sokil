"""
The YOLO wrappers. No checkpoint is loaded: ultralytics.YOLO is replaced by a
fake, so what is tested is the shape each wrapper hands back to the pipeline.
"""

import sys
import types
from typing import ClassVar

import numpy as np
import pytest

from sokil.models import (
    Model,
    PoseEstimator,
    Segmenter,
    ShuttleDetector,
    _to_numpy,
)


class FakeTensor:
    """Stands in for a torch tensor: it has to be detached before use."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.detached = False

    def detach(self):
        self.detached = True
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = FakeTensor(xyxy)
        self.conf = None if conf is None else FakeTensor(conf)

    def __len__(self):
        return len(self.xyxy.values)


class FakeResult:
    def __init__(self, boxes=None, masks=None, keypoints=None):
        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints


class FakeYOLO:
    """Records how it was called and replays whatever results a test set up."""

    instances: ClassVar[list["FakeYOLO"]] = []

    def __init__(self, path, task=None):
        self.path = path
        self.task = task
        self.calls = []
        self.results = []
        FakeYOLO.instances.append(self)

    def predict(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return self.results


@pytest.fixture(autouse=True)
def fake_ultralytics(monkeypatch):
    """Make `from ultralytics import YOLO` resolve to the fake."""
    FakeYOLO.instances = []
    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return FakeYOLO


class TestToNumpy:
    def test_a_tensor_is_detached(self):
        tensor = FakeTensor([1, 2, 3])

        assert _to_numpy(tensor).tolist() == [1, 2, 3]
        assert tensor.detached is True

    def test_none_passes_through(self):
        assert _to_numpy(None) is None

    def test_a_plain_array_passes_through(self):
        array = np.array([1, 2])

        assert _to_numpy(array) is array


class TestModel:
    def test_the_checkpoint_is_loaded_for_the_subclasss_task(self):
        detector = ShuttleDetector("weights.pt")

        assert detector.model.task == "detect"
        assert detector.model.path == "weights.pt"

    def test_a_path_object_is_accepted(self, tmp_path):
        detector = ShuttleDetector(tmp_path / "weights.pt")

        assert detector.model_path == str(tmp_path / "weights.pt")

    def test_the_configured_confidence_is_passed_to_every_prediction(self):
        model = Model("weights.pt", confidence=0.4)
        model.model.results = [FakeResult()]

        model.predict(np.zeros((4, 4, 3), np.uint8))

        assert model.model.calls[0][1]["conf"] == 0.4

    def test_frames_are_letterboxed_to_a_full_square(self):
        model = Model("weights.pt", imgsz=320)
        model.model.results = [FakeResult()]

        model.predict(np.zeros((4, 4, 3), np.uint8))

        assert model.model.calls[0][1]["imgsz"] == 320
        assert model.model.calls[0][1]["rect"] is False

    def test_a_per_call_confidence_overrides_it(self):
        model = Model("weights.pt", confidence=0.4)
        model.model.results = [FakeResult()]

        model.predict(np.zeros((4, 4, 3), np.uint8), confidence=0.9)

        assert model.model.calls[0][1]["conf"] == 0.9

    def test_the_device_is_only_named_when_one_was_chosen(self):
        without = Model("weights.pt")
        without.model.results = [FakeResult()]
        without.predict(np.zeros((4, 4, 3), np.uint8))

        chosen = Model("weights.pt", device="cpu")
        chosen.model.results = [FakeResult()]
        chosen.predict(np.zeros((4, 4, 3), np.uint8))

        assert "device" not in without.model.calls[0][1]
        assert chosen.model.calls[0][1]["device"] == "cpu"

    def test_an_empty_result_list_is_no_prediction(self):
        model = Model("weights.pt")
        model.model.results = []

        assert model.predict(np.zeros((4, 4, 3), np.uint8)) is None

    def test_the_repr_names_the_checkpoint(self):
        assert repr(ShuttleDetector("weights.pt")) == "ShuttleDetector('weights.pt')"


class TestLifecycle:
    """A model holds its device only between load() and release()."""

    def test_nothing_is_loaded_until_it_is_asked_for(self):
        model = Model("weights.pt")

        assert not model.loaded
        assert FakeYOLO.instances == []

    def test_loading_warms_it_up_on_a_blank_frame_of_its_size(self):
        model = Model("weights.pt", imgsz=320).load()

        image, kwargs = model.model.calls[0]
        assert model.loaded
        assert image.shape == (320, 320, 3)
        assert kwargs["imgsz"] == 320

    def test_loading_twice_warms_it_up_once(self):
        model = Model("weights.pt").load()
        model.load()

        assert len(FakeYOLO.instances) == 1
        assert len(model.model.calls) == 1

    def test_releasing_drops_the_model_and_a_later_use_loads_it_again(self):
        model = Model("weights.pt").load()

        model.release()

        assert not model.loaded
        model.model.results = [FakeResult()]
        assert len(FakeYOLO.instances) == 2

    def test_releasing_what_was_never_loaded_is_harmless(self):
        Model("weights.pt").release()

    def test_a_with_block_loads_on_entry_and_releases_on_exit(self):
        with Model("weights.pt") as model:
            assert model.loaded

        assert not model.loaded


class TestStagesOwnTheirModels:
    def test_the_court_solver_loads_its_segmenter_for_the_block(self):
        from sokil.court import CourtSolver

        segmenter, keypoints = Segmenter("court.pt"), PoseEstimator("pose.pt")
        solver = CourtSolver(
            {"segmentation_model": segmenter, "keypoint_model": keypoints}
        )

        with solver:
            assert segmenter.loaded
            # optional, so left to load on first use
            assert not keypoints.loaded

        assert not segmenter.loaded

    def test_the_tracker_loads_its_detector_for_the_block(self):
        from sokil.tracker import ShuttleTracker

        tracker = ShuttleTracker({"model": "shuttle.pt"})

        with tracker:
            assert tracker.detector.loaded

        assert not tracker.detector.loaded

    def test_a_tracker_fed_ground_truth_has_nothing_to_load(self):
        from sokil.tracker import ShuttleTracker

        with ShuttleTracker(detector=object()) as tracker:
            assert tracker.detector is not None


class TestShuttleDetector:
    def test_reports_the_most_confident_box_not_the_first(self):
        detector = ShuttleDetector("weights.pt")
        detector.model.results = [
            FakeResult(
                boxes=FakeBoxes(
                    xyxy=[[0, 0, 10, 10], [50, 50, 60, 60]], conf=[0.3, 0.9]
                )
            )
        ]

        assert detector.best_box(np.zeros((4, 4, 3), np.uint8)) == (
            (50.0, 50.0, 60.0, 60.0),
            0.9,
        )

    def test_no_detection_at_all(self):
        detector = ShuttleDetector("weights.pt")
        detector.model.results = [FakeResult(boxes=None)]

        assert detector.best_box(np.zeros((4, 4, 3), np.uint8)) is None

    def test_an_empty_box_list(self):
        detector = ShuttleDetector("weights.pt")
        detector.model.results = [FakeResult(boxes=FakeBoxes(xyxy=[], conf=[]))]

        assert detector.best_box(np.zeros((4, 4, 3), np.uint8)) is None

    def test_a_result_without_confidences_falls_back_to_the_first_box(self):
        detector = ShuttleDetector("weights.pt")
        detector.model.results = [
            FakeResult(boxes=FakeBoxes(xyxy=[[1, 2, 3, 4]], conf=None))
        ]

        box, confidence = detector.best_box(np.zeros((4, 4, 3), np.uint8))

        assert box == (1.0, 2.0, 3.0, 4.0)
        assert np.isnan(confidence)


class TestSegmenter:
    def test_returns_every_mask_outline(self):
        segmenter = Segmenter("weights.pt")
        masks = types.SimpleNamespace(xy=[np.array([[0, 0], [5, 0], [5, 5]])])
        segmenter.model.results = [FakeResult(masks=masks)]

        polygons = segmenter.polygons(np.zeros((4, 4, 3), np.uint8))

        assert len(polygons) == 1
        assert polygons[0].shape == (3, 2)

    def test_nothing_segmented_is_an_empty_list_not_none(self):
        segmenter = Segmenter("weights.pt")
        segmenter.model.results = [FakeResult(masks=None)]

        assert segmenter.polygons(np.zeros((4, 4, 3), np.uint8)) == []


class TestPoseEstimator:
    def test_returns_the_first_instances_points_and_confidences(self):
        estimator = PoseEstimator("weights.pt")
        keypoints = types.SimpleNamespace(
            xy=FakeTensor([[[1, 2], [3, 4]], [[9, 9], [8, 8]]]),
            conf=FakeTensor([[0.8, 0.2], [0.1, 0.1]]),
        )
        estimator.model.results = [FakeResult(keypoints=keypoints)]

        points, confidences = estimator.keypoints(np.zeros((4, 4, 3), np.uint8))

        assert points.tolist() == [[1, 2], [3, 4]]
        assert confidences.tolist() == pytest.approx([0.8, 0.2])

    def test_no_keypoints_at_all(self):
        estimator = PoseEstimator("weights.pt")
        estimator.model.results = [FakeResult(keypoints=None)]

        assert estimator.keypoints(np.zeros((4, 4, 3), np.uint8)) == (None, None)

    def test_an_empty_keypoint_set(self):
        estimator = PoseEstimator("weights.pt")
        keypoints = types.SimpleNamespace(xy=FakeTensor([]), conf=None)
        estimator.model.results = [FakeResult(keypoints=keypoints)]

        assert estimator.keypoints(np.zeros((4, 4, 3), np.uint8)) == (None, None)

    def test_keypoints_without_confidences(self):
        estimator = PoseEstimator("weights.pt")
        keypoints = types.SimpleNamespace(xy=FakeTensor([[[1, 2]]]), conf=None)
        estimator.model.results = [FakeResult(keypoints=keypoints)]

        points, confidences = estimator.keypoints(np.zeros((4, 4, 3), np.uint8))

        assert points.tolist() == [[1, 2]]
        assert confidences is None
