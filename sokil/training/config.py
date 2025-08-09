"""
Where the training defaults come from.

The model spec is read from a YAML file rather than hardcoded so the defaults
can be edited without touching code or repeating flags on every command line.
"""

import dataclasses
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Datasets are produced by export-cvat-dataset and belong to whoever runs it,
# not to the installed package, so they are addressed from the working
# directory. In a checkout run from the repository root this is ./training,
# which is where they already live.
TRAINING_ROOT = Path(os.environ.get("SOKIL_TRAINING_ROOT", "training"))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ModelKind(str, Enum):
    SHUTTLE = "shuttle"
    COURT = "court"


# Shipped with the package: these are the defaults the code was written
# against, so they have to be there whether the project was installed or is
# being run from a checkout. SOKIL_TRAINING_CONFIG points at your own copy.
DEFAULT_CONFIG_PATH = Path(__file__).parent / "training.yaml"


@dataclass(frozen=True)
class ModelSpec:
    """The per-model defaults read from the training config (see training.yaml)."""

    project: str  # CVAT project the annotation tasks live in
    label: str  # CVAT label, and the single class name in the exported data.yaml
    label_type: str  # CVAT shape type: rectangle, polygon, ...
    grayscale: bool  # convert frames on export, and feed the same at inference
    weights: str  # starting weights for a fresh run
    imgsz: int  # square training size
    batch: int


class TrainingConfigError(RuntimeError):
    """The training config is missing, malformed, or names something unknown."""


def config_path() -> Path:
    return Path(os.environ.get("SOKIL_TRAINING_CONFIG", DEFAULT_CONFIG_PATH))


def load_model_specs(path: Path | None = None) -> dict[ModelKind, ModelSpec]:
    """
    Read the per-model defaults, keyed by ModelKind.

    Every field is required and unknown keys are rejected: the file is meant to
    be edited by hand, and silently ignoring a typo would leave a run using a
    default the file appears to override.
    """
    import yaml

    path = path or config_path()

    try:
        document = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raise TrainingConfigError(
            f"No training config at {path}. Restore it, or point "
            f"SOKIL_TRAINING_CONFIG at another file."
        ) from None
    except yaml.YAMLError as error:
        raise TrainingConfigError(f"Could not parse {path}: {error}") from error

    models = document.get("models")
    if not isinstance(models, dict):
        raise TrainingConfigError(f"{path} has no 'models:' mapping")

    known = {kind.value for kind in ModelKind}
    unknown = sorted(set(models) - known)
    if unknown:
        raise TrainingConfigError(
            f"{path} configures unknown model(s) {unknown}; "
            f"known models: {sorted(known)}"
        )

    fields = {field.name for field in dataclasses.fields(ModelSpec)}
    specs = {}

    for kind in ModelKind:
        section = models.get(kind.value)
        if not isinstance(section, dict):
            raise TrainingConfigError(f"{path} is missing the '{kind.value}:' model")

        extra = sorted(set(section) - fields)
        if extra:
            raise TrainingConfigError(
                f"{path}: unknown key(s) {extra} under models.{kind.value}; "
                f"expected {sorted(fields)}"
            )
        missing = sorted(fields - set(section))
        if missing:
            raise TrainingConfigError(
                f"{path}: models.{kind.value} is missing {missing}"
            )

        values = dict(section)
        values["imgsz"] = _parse_imgsz(values["imgsz"], path, kind)
        specs[kind] = ModelSpec(**values)

    return specs


def _parse_imgsz(value, path: Path, kind: ModelKind) -> int:
    """Training runs square, so imgsz is a single positive number."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrainingConfigError(
            f"{path}: models.{kind.value}.imgsz must be a whole number, got {value!r}"
        )
    if value <= 0:
        raise TrainingConfigError(
            f"{path}: models.{kind.value}.imgsz must be positive, got {value!r}"
        )
    return value


MODEL_SPECS = load_model_specs()
