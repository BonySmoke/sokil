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
from pathlib import Path
from typing import Self

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
            self.predict(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))
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
        kwargs = {
            "conf": self.confidence if confidence is None else confidence,
            "imgsz": self.imgsz,
            "rect": False,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        results = self.model.predict(image, **kwargs)
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


def _to_numpy(value):
    """Detach a torch tensor to numpy; numpy and None pass through unchanged."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value
