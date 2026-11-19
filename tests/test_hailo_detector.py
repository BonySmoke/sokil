"""
The HailoRT shuttle detector, against a fake hailo_platform: what is tested is
what the host does around the chip — the letterbox going in, the batching, and
the NMS output coming back into frame pixels.
"""

import sys
import types

import numpy as np
import pytest

from sokil.models import (
    LETTERBOX_FILL,
    HailoShuttleDetector,
    ShuttleDetector,
    best_nms_box,
    is_hailo_model,
    letterbox,
)
from sokil.tracker import ShuttleTracker, TrackerConfig, shuttle_detector

INPUT = (640, 640)


class TestLetterbox:
    def test_a_640_by_480_frame_is_padded_top_and_bottom(self):
        image = np.zeros((480, 640, 3), np.uint8)

        padded, (scale, left, top) = letterbox(image, INPUT)

        assert padded.shape == (640, 640, 3)
        assert (scale, left, top) == (1.0, 0, 80)
        assert tuple(padded[0, 0]) == LETTERBOX_FILL
        assert tuple(padded[80, 0]) == (0, 0, 0)
        assert tuple(padded[560, 0]) == LETTERBOX_FILL

    def test_the_frame_is_handed_over_as_rgb(self):
        image = np.zeros((480, 640, 3), np.uint8)
        image[..., 0] = 255  # blue in OpenCV's BGR

        padded, _ = letterbox(image, INPUT)

        assert tuple(padded[300, 300]) == (0, 0, 255)

    def test_a_larger_frame_is_scaled_down_to_fit(self):
        padded, (scale, left, top) = letterbox(
            np.zeros((960, 1280, 3), np.uint8), INPUT
        )

        assert padded.shape == (640, 640, 3)
        assert scale == 0.5
        assert (left, top) == (0, 80)


class TestBestNmsBox:
    def test_a_box_comes_back_in_frame_pixels(self):
        # (y1, x1, y2, x2, score), normalised to the 640x640 input
        detections = [np.array([[0.25, 0.5, 0.375, 0.625, 0.9]])]

        box, score = best_nms_box(detections, INPUT, (1.0, 0, 80), (480, 640), 0.5)

        assert box == pytest.approx((320.0, 80.0, 400.0, 160.0))
        assert score == pytest.approx(0.9)

    def test_the_most_confident_box_wins_across_classes(self):
        detections = [
            np.array([[0.1, 0.1, 0.2, 0.2, 0.6], [0.3, 0.3, 0.4, 0.4, 0.8]]),
            np.array([[0.5, 0.5, 0.6, 0.6, 0.7]]),
        ]

        _, score = best_nms_box(detections, INPUT, (1.0, 0, 80), (480, 640), 0.5)

        assert score == pytest.approx(0.8)

    def test_below_the_confidence_is_no_shuttle(self):
        detections = [np.array([[0.25, 0.5, 0.375, 0.625, 0.4]])]

        assert best_nms_box(detections, INPUT, (1.0, 0, 80), (480, 640), 0.5) is None

    def test_an_empty_frame_is_no_shuttle(self):
        assert (
            best_nms_box([np.empty((0, 5))], INPUT, (1.0, 0, 80), (480, 640), 0.5)
            is None
        )

    def test_a_box_in_the_padding_is_clipped_to_the_frame(self):
        detections = [np.array([[0.0, 0.0, 0.2, 0.2, 0.9]])]

        box, _ = best_nms_box(detections, INPUT, (1.0, 0, 80), (480, 640), 0.5)

        assert box[1] == 0.0


class FakeHailo:
    """Just enough of hailo_platform to drive the detector without a device."""

    def __init__(self, box=(0.25, 0.5, 0.375, 0.625, 0.9)):
        self.box = np.array([box])
        self.batches = []
        self.batch_size = None
        self.closed = False

    def module(self):
        fake = self
        module = types.ModuleType("hailo_platform")

        class Info:
            def __init__(self, name, shape):
                self.name, self.shape = name, shape

        class HEF:
            def __init__(self, path):
                self.path = path

            def get_input_vstream_infos(self):
                return [Info("in", (640, 640, 3))]

            def get_output_vstream_infos(self):
                return [Info("nms", ())]

        class Params:
            batch_size = 1

        class ConfigureParams:
            @staticmethod
            def create_from_hef(hef, interface):
                return {"net": Params()}

        class Context:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                fake.closed = True

        class NetworkGroup:
            def activate(self, params):
                return Context()

            def create_params(self):
                return None

        class VDevice(Context):
            def configure(self, hef, params):
                fake.batch_size = params["net"].batch_size
                return [NetworkGroup()]

        class InferVStreams(Context):
            def __init__(self, network_group, inputs, outputs):
                pass

            def infer(self, data):
                batch = data["in"]
                # as HailoRT does for a multi-context network
                if len(batch) % fake.batch_size:
                    raise ValueError(
                        f"frames count must be a multiple of the batch size "
                        f"({len(batch)} % {fake.batch_size} != 0)"
                    )
                fake.batches.append(batch.shape)
                return {"nms": [[fake.box] for _ in range(len(batch))]}

        class VStreamParams:
            @staticmethod
            def make(network_group, format_type):
                return None

        module.HEF = HEF
        module.ConfigureParams = ConfigureParams
        module.VDevice = VDevice
        module.InferVStreams = InferVStreams
        module.InputVStreamParams = VStreamParams
        module.OutputVStreamParams = VStreamParams
        module.FormatType = types.SimpleNamespace(UINT8="uint8", FLOAT32="float32")
        module.HailoStreamInterface = types.SimpleNamespace(PCIe="pcie")
        return module


@pytest.fixture
def hailo(monkeypatch, tmp_path):
    fake = FakeHailo()
    monkeypatch.setitem(sys.modules, "hailo_platform", fake.module())
    model = tmp_path / "train-hailo8_hailo_model"
    model.mkdir()
    (model / "train-hailo8.hef").write_bytes(b"")
    return fake, model


def frame():
    return np.zeros((480, 640, 3), np.uint8)


class TestHailoShuttleDetector:
    def test_the_network_is_configured_for_the_batch_size(self, hailo):
        fake, model = hailo

        HailoShuttleDetector(model, batch_size=4).load()

        assert fake.batch_size == 4

    def test_frames_go_to_the_chip_in_whole_batches(self, hailo):
        fake, model = hailo
        detector = HailoShuttleDetector(model, batch_size=2).load()
        fake.batches.clear()

        boxes = detector.best_boxes([frame(), frame(), frame()])

        # the short last batch is filled up, and its filler answer dropped
        assert fake.batches == [(2, 640, 640, 3), (2, 640, 640, 3)]
        assert len(boxes) == 3
        assert detector.inferences == 3

    def test_the_warm_up_sends_a_whole_batch(self, hailo):
        fake, model = hailo

        HailoShuttleDetector(model, batch_size=4).load()

        assert fake.batches == [(4, 640, 640, 3)]

    def test_a_single_frame_is_sent_as_a_whole_batch(self, hailo):
        fake, model = hailo
        detector = HailoShuttleDetector(model, batch_size=4).load()
        fake.batches.clear()

        assert detector.best_box(frame()) is not None
        assert fake.batches == [(4, 640, 640, 3)]

    def test_boxes_come_back_in_frame_pixels(self, hailo):
        _, model = hailo

        box, score = HailoShuttleDetector(model).best_box(frame())

        assert box == pytest.approx((320.0, 80.0, 400.0, 160.0))
        assert score == pytest.approx(0.9)

    def test_the_load_is_counted_for_the_run_stats(self, hailo):
        _, model = hailo

        detector = HailoShuttleDetector(model).load()

        assert detector.loads == 1

    def test_release_gives_the_device_back(self, hailo):
        fake, model = hailo
        detector = HailoShuttleDetector(model).load()

        detector.release()

        assert fake.closed
        assert not detector.loaded

    def test_a_batch_size_below_one_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="batch_size"):
            HailoShuttleDetector(tmp_path, batch_size=0)


class TestChoosingTheDetector:
    def test_a_hailo_export_is_recognised_by_its_name(self):
        assert is_hailo_model("models/train-hailo8_hailo_model")
        assert not is_hailo_model("models/train.pt")

    def test_a_hef_gets_the_hailo_detector_with_the_batch_size(self):
        detector = shuttle_detector(
            "models/train-hailo8_hailo_model", TrackerConfig(batch_size=4)
        )

        assert isinstance(detector, HailoShuttleDetector)
        assert detector.batch_size == 4

    def test_other_weights_keep_the_ultralytics_detector(self):
        detector = shuttle_detector("train.pt", TrackerConfig())

        assert type(detector) is ShuttleDetector


class TestTrackerBatching:
    class BatchingDetector:
        """Answers in batches, recording how the frames arrived."""

        batch_size = 3

        def __init__(self):
            self.calls = []

        def best_boxes(self, images, confidence=None, frames=None):
            self.calls.append(list(frames))
            return [((10.0 + f, 10.0, 20.0 + f, 20.0), 0.9) for f in frames]

    class Source:
        def __init__(self, count):
            self.count = count
            self.frame_count = count

        def __iter__(self):
            from sokil.video import DecodedFrame

            for number in range(self.count):
                image = np.zeros((120, 160, 3), np.uint8)
                yield DecodedFrame(
                    number=number,
                    image=image,
                    model_input=image,
                    timestamp=number / 30,
                )

        def describe(self):
            return "fake source"

    def test_frames_reach_a_batching_detector_in_batches(self):
        detector = self.BatchingDetector()

        tracks = ShuttleTracker(detector=detector).track(self.Source(7))

        assert detector.calls == [[0, 1, 2], [3, 4, 5], [6]]
        assert [track.number for track in tracks] == list(range(7))
        assert tracks[6].xyxy == (16, 10, 26, 20)
