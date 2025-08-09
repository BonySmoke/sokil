"""
Running a training job and keeping its best weights.
"""

import logging
import shutil
from pathlib import Path

from .config import MODEL_SPECS, TRAINING_ROOT, ModelKind

logger = logging.getLogger(__name__)


def latest_checkpoint(kind: ModelKind) -> Path | None:
    """Most recently written last.pt across this model's training runs."""
    runs = TRAINING_ROOT / kind.value / "runs"
    checkpoints = sorted(
        runs.glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime
    )
    return checkpoints[-1] if checkpoints else None


def _train_output(results) -> tuple[Path, Path]:
    """Locate a finished run's directory and its best weights."""
    save_dir = Path(results.save_dir)
    return save_dir, save_dir / "weights" / "best.pt"


def train_model(
    kind: ModelKind,
    base_weights: str | None = None,
    epochs: int = 100,
    imgsz: int | None = None,
    device: str | None = None,
    batch: int | None = None,
    workers: int = 8,
    cache: bool | str = False,
    patience: int = 100,
    resume: bool = False,
    deterministic: bool | None = None,
) -> Path:
    """
    Train a YOLO model.
    """
    from ultralytics import YOLO

    spec = MODEL_SPECS[kind]
    data_yaml = TRAINING_ROOT / kind.value / "datasets" / "data.yaml"

    if not data_yaml.exists():
        raise ValueError(f"{data_yaml} not found — run export-dataset first")

    weights = base_weights or spec.weights
    # each family constrains imgsz differently — YOLO9 takes the dataset's own
    # 4:3 shape, while EC segmentation only accepts its native square size — so
    # the default belongs to the model, not to the CLI
    if imgsz is None:
        imgsz = spec.imgsz
    if batch is None:
        batch = spec.batch

    # Ultralytics defaults deterministic=True, which calls
    # torch.use_deterministic_algorithms(True, warn_only=True). Several backward
    # ops have no deterministic MPS kernel, so on Apple silicon that produces a
    # warning per step and then uses the non-deterministic kernel regardless —
    # the run is not reproducible either way. The other two knobs it sets
    # (cudnn.deterministic, CUBLAS_WORKSPACE_CONFIG) are CUDA-only, and seeding
    # happens outside the flag, so switching it off on MPS costs nothing.
    if deterministic is None:
        deterministic = "mps" not in str(device or "").lower()

    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
        "batch": batch,
        "workers": workers,
        "cache": cache,
        "patience": patience,
        "deterministic": deterministic,
        "project": str(TRAINING_ROOT / kind.value / "runs"),
    }

    if resume:
        checkpoint = Path(weights) if base_weights else latest_checkpoint(kind)
        if checkpoint is None or not checkpoint.exists():
            raise ValueError(
                f"No checkpoint to resume from under "
                f"{TRAINING_ROOT / kind.value / 'runs'} — train without --resume"
            )
        # the trainer restores the checkpoint's own settings (epochs, dataset,
        # augmentation); only device/batch/imgsz/workers/cache/patience are
        # taken from this call.
        logger.info("Resuming training from %s", checkpoint)
        weights = checkpoint
        train_args["resume"] = str(checkpoint)

    model = YOLO(str(weights))
    results = model.train(**train_args)

    save_dir, best = _train_output(results)

    if not best.is_file():
        raise ValueError(
            f"Training finished without a best checkpoint at {best}. The run "
            f"directory is {save_dir}"
        )

    models_dir = TRAINING_ROOT / kind.value / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    destination = models_dir / f"{save_dir.name}.pt"
    shutil.copy2(best, destination)

    logger.info("Best weights copied to %s", destination)
    return destination
