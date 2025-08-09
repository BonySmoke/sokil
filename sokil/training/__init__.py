"""
The training pipeline: annotate in CVAT, export a YOLO dataset, train a model.

Kept in one subpackage because none of it runs during a review — only the CVAT
and training workflows import it, and only those need cvat-sdk.
"""

from .config import (
    MODEL_SPECS,
    TRAINING_ROOT,
    ModelKind,
    ModelSpec,
    TrainingConfigError,
    config_path,
    load_model_specs,
)
from .cvat import create_cvat_task, list_completed_task_ids, make_cvat_client
from .dataset import export_dataset
from .frames import extract_frames, load_undistorter
from .train import latest_checkpoint, train_model

__all__ = [
    "MODEL_SPECS",
    "TRAINING_ROOT",
    "ModelKind",
    "ModelSpec",
    "TrainingConfigError",
    "config_path",
    "create_cvat_task",
    "export_dataset",
    "extract_frames",
    "latest_checkpoint",
    "list_completed_task_ids",
    "load_model_specs",
    "load_undistorter",
    "make_cvat_client",
    "train_model",
]
