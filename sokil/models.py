"""
The YOLO models the pipeline infers with.

One class per task, so callers ask for the shape they want — a box, a set of
polygons, a set of keypoints — rather than reaching into a raw result.

A model holds its device only while it is loaded. An accelerator such as a
Hailo-8 serves one model at a time, and a small GPU may fit only one, so each
stage loads its models on the way in and releases them on the way out:

    with Segmenter(path) as segmenter:   # loads and warms up, before any frame
        segmenter.polygons(frame)
                                         # released here

Used without `with`, a model loads on its first prediction and keeps its device.
"""

import gc
import logging
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Self

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Model:
    """A loaded YOLO checkpoint that runs one frame at a time."""

    task: str | None = None

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        device: str | None = None,
        imgsz: int = 640,
    ):
        self.model_path = str(model_path)
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz
        self._model = None

        # What sokil.timing reports: loads include the warm-up frame, and
        # inferences time the predict call alone.
        self.loads = 0
        self.load_seconds = 0.0
        self.inferences = 0
        self.inference_seconds = 0.0

    @property
    def model(self):
        """The YOLO model, loaded on first use if load() was not called."""
        if self._model is None:
            from ultralytics import YOLO

            logger.debug("Loading %s", self.model_path)
            self._model = YOLO(self.model_path, task=self.task)
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Self:
        """
        Load the model and run one blank frame through it.

        Ultralytics sets up an exported model only on
        its first prediction, so run a blank prediction for faster predictions on real data
        """
        if not self.loaded:
            started = time.perf_counter()
            self._predict(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))
            self.loads += 1
            self.load_seconds += time.perf_counter() - started
        return self

    def release(self) -> None:
        """
        Give the device back for the next model.

        Ultralytics' Hailo backend closes its device when it is collected, and
        torch keeps freed GPU memory cached until asked to return it.
        """
        if not self.loaded:
            return

        logger.debug("Releasing %s", self.model_path)
        self._model = None
        gc.collect()

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> Self:
        return self.load()

    def __exit__(self, *exc_info) -> None:
        self.release()

    def predict(self, image: np.ndarray, confidence: float | None = None):
        """
        Run one OpenCV frame through the model; returns its Results, or None.

        The frame is passed as a numpy array, which YOLO reads as BGR — the same
        order OpenCV decodes into — so no colour conversion is needed here.
        """
        model = self.model
        started = time.perf_counter()
        result = self._predict(image, confidence, model)
        self.inferences += 1
        self.inference_seconds += time.perf_counter() - started
        return result

    def _predict(self, image: np.ndarray, confidence: float | None = None, model=None):
        kwargs = {
            "conf": self.confidence if confidence is None else confidence,
            "imgsz": self.imgsz,
            "rect": False,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        results = (model or self.model).predict(image, **kwargs)
        return results[0] if results else None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model_path!r})"


class ShuttleDetector(Model):
    """Object detection: reports the single best box on a frame."""

    task = "detect"

    def best_box(
        self,
        image: np.ndarray,
        confidence: float | None = None,
        frame: int | None = None,
    ) -> tuple[tuple[float, float, float, float], float] | None:
        """
        The highest-confidence box as (xyxy, confidence), or None.

        The pipeline follows one shuttle, so it wants the most confident box
        rather than whichever came first: box order is a post-processing detail.

        :param frame: frame ID. Used for getting ground truth labels during validation.
        """
        result = self.predict(image, confidence)

        if result is None or result.boxes is None or not len(result.boxes):
            return None

        boxes = result.boxes
        xyxy = np.asarray(_to_numpy(boxes.xyxy), dtype=float).reshape(-1, 4)
        confidences = _to_numpy(boxes.conf)

        if confidences is None:
            index, best = 0, float("nan")
        else:
            confidences = np.asarray(confidences, dtype=float).reshape(-1)
            index = int(np.argmax(confidences))
            best = float(confidences[index])

        x1, y1, x2, y2 = (float(value) for value in xyxy[index])
        return (x1, y1, x2, y2), best


class Segmenter(Model):
    """Instance segmentation: reports mask polygons in pixel coordinates."""

    task = "segment"

    def polygons(
        self, image: np.ndarray, confidence: float | None = None
    ) -> list[np.ndarray]:
        """Every mask outline found on the frame, in image pixel coordinates."""
        result = self.predict(image, confidence)

        if result is None or result.masks is None:
            return []
        return [np.asarray(polygon) for polygon in result.masks.xy]


class PoseEstimator(Model):
    """Keypoint detection: reports the first instance's points."""

    task = "pose"

    def keypoints(
        self, image: np.ndarray, confidence: float | None = None
    ) -> tuple[np.ndarray, np.ndarray | None] | tuple[None, None]:
        """The first instance's (points, confidences), or (None, None)."""
        result = self.predict(image, confidence)

        if result is None or result.keypoints is None:
            return None, None

        points = _to_numpy(result.keypoints.xy)
        if points is None or not len(points):
            return None, None

        confidences = _to_numpy(result.keypoints.conf)
        return (
            np.asarray(points[0]),
            np.asarray(confidences[0]) if confidences is not None else None,
        )


# What Ultralytics pads a letterboxed frame with, so the HEF sees the border it
# was calibrated on.
LETTERBOX_FILL = (114, 114, 114)


def is_hailo_model(path: str | Path) -> bool:
    """Whether a model path is a Hailo export (<stem>-hailo8_hailo_model/)."""
    return Path(path).name.endswith("_hailo_model")


class HailoShuttleDetector(Model):
    """
    The shuttle detector on a Hailo accelerator, driven through HailoRT directly.

    HailoRT processes batches at much higher speeds than one frame per call like Ultralitics does.

    Here the network is configured for `batch_size` frames, the frame is only
    letterboxed on the host, and the HEF's own NMS output is read directly, so
    the CPU work per frame is a pad and a few array lookups.
    """

    task = "detect"

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        device: str | None = None,
        imgsz: int = 640,
        batch_size: int = 16,
    ):
        super().__init__(model_path, confidence=confidence, device=device, imgsz=imgsz)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.batch_size = batch_size
        self._stack: ExitStack | None = None
        self._pipeline = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._input_size: tuple[int, int] = (imgsz, imgsz)  # (height, width)

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> Self:
        """Configure the network for the batch size and warm it up."""
        if self.loaded:
            return self

        from hailo_platform import (  # pyright: ignore[reportMissingImports] -- HailoRT exists only on the Pi
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        started = time.perf_counter()
        hef_file = next(Path(self.model_path).rglob("*.hef"), None)
        if hef_file is None:
            raise FileNotFoundError(f"No .hef file found in {self.model_path}")

        logger.debug("Loading %s with batch size %d", hef_file, self.batch_size)
        hef = HEF(str(hef_file))

        with ExitStack() as stack:
            device = stack.enter_context(VDevice())
            params = ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe
            )
            for name in params:
                params[name].batch_size = self.batch_size
            network_group = device.configure(hef, params)[0]
            stack.enter_context(network_group.activate(network_group.create_params()))
            self._pipeline = stack.enter_context(
                InferVStreams(
                    network_group,
                    InputVStreamParams.make(
                        network_group, format_type=FormatType.UINT8
                    ),
                    OutputVStreamParams.make(
                        network_group, format_type=FormatType.FLOAT32
                    ),
                )
            )
            self._stack = stack.pop_all()

        input_info = hef.get_input_vstream_infos()[0]
        self._input_name = input_info.name
        self._input_size = tuple(input_info.shape[:2])
        self._output_name = hef.get_output_vstream_infos()[0].name

        # A whole batch: a multi-context network only accepts multiples of
        # the batch size it was configured for.
        height, width = self._input_size
        self._infer([np.zeros((height, width, 3), dtype=np.uint8)] * self.batch_size)

        self.loads += 1
        self.load_seconds += time.perf_counter() - started
        return self

    def release(self) -> None:
        """Close the pipeline and give the device back."""
        if self._stack is not None:
            logger.debug("Releasing %s", self.model_path)
            self._stack.close()
        self._stack = None
        self._pipeline = None

    def best_box(
        self,
        image: np.ndarray,
        confidence: float | None = None,
        frame: int | None = None,
    ) -> tuple[tuple[float, float, float, float], float] | None:
        """The highest-confidence box as (xyxy, confidence), or None."""
        return self.best_boxes([image], confidence)[0]

    def best_boxes(
        self,
        images: list[np.ndarray],
        confidence: float | None = None,
        frames: list[int] | None = None,
    ) -> list[tuple[tuple[float, float, float, float], float] | None]:
        """
        best_box for several frames, sent to the chip `batch_size` at a time.

        A short last batch is filled up by repeating its last frame, whose
        answers are thrown away: the network was configured for whole batches.
        """
        self.load()
        threshold = self.confidence if confidence is None else confidence

        boxes = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            began = time.perf_counter()

            padded, placements = zip(
                *(letterbox(image, self._input_size) for image in chunk), strict=True
            )
            filler = [padded[-1]] * (self.batch_size - len(chunk))
            detections = self._infer([*padded, *filler])

            for image, placement, frame_detections in zip(
                chunk, placements, detections, strict=False
            ):
                boxes.append(
                    best_nms_box(
                        frame_detections,
                        self._input_size,
                        placement,
                        image.shape[:2],
                        threshold,
                    )
                )

            self.inferences += len(chunk)
            self.inference_seconds += time.perf_counter() - began

        return boxes

    def _infer(self, padded: list[np.ndarray]) -> list:
        """Run letterboxed RGB frames through the chip; per-frame NMS output."""
        batch = np.ascontiguousarray(np.stack(padded))
        return self._pipeline.infer({self._input_name: batch})[self._output_name]


def letterbox(
    image: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Fit a BGR frame into size (height, width) as Ultralytics does, as RGB.

    Scaled to fit, centred, and padded with LETTERBOX_FILL, so the network sees
    what it saw in training and calibration.

    :returns: the padded RGB frame, and (scale, left pad, top pad) to put boxes
        back into the original frame.
    """
    height, width = image.shape[:2]
    target_height, target_width = size

    scale = min(target_height / height, target_width / width)
    new_width, new_height = round(width * scale), round(height * scale)
    if (new_width, new_height) != (width, height):
        image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

    pad_x = (target_width - new_width) / 2
    pad_y = (target_height - new_height) / 2
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)

    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=LETTERBOX_FILL
    )
    return padded[..., ::-1], (scale, left, top)


def best_nms_box(
    detections: list,
    input_size: tuple[int, int],
    placement: tuple[float, float, float],
    image_size: tuple[int, int],
    confidence: float,
) -> tuple[tuple[float, float, float, float], float] | None:
    """
    The highest-scoring box in one frame of HailoRT's NMS-by-class output.

    :param detections: one array per class, each row (y1, x1, y2, x2, score)
        normalised to the network input.
    :param placement: (scale, left pad, top pad) from letterbox.
    :returns: (xyxy, confidence) in the original frame's pixels, or None.
    """
    best = None
    for class_detections in detections:
        rows = np.asarray(class_detections, dtype=float).reshape(-1, 5)
        if not len(rows):
            continue
        row = rows[int(np.argmax(rows[:, 4]))]
        if best is None or row[4] > best[4]:
            best = row

    if best is None or best[4] < confidence:
        return None

    input_height, input_width = input_size
    scale, left, top = placement
    image_height, image_width = image_size

    y1, x1, y2, x2, score = best
    x1, x2 = ((x * input_width - left) / scale for x in (x1, x2))
    y1, y2 = ((y * input_height - top) / scale for y in (y1, y2))

    return (
        float(np.clip(x1, 0, image_width)),
        float(np.clip(y1, 0, image_height)),
        float(np.clip(x2, 0, image_width)),
        float(np.clip(y2, 0, image_height)),
    ), float(score)


def _to_numpy(value):
    """Detach a torch tensor to numpy; numpy and None pass through unchanged."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value
