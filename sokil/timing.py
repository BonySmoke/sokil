"""
How long a run took: each stage, each model, and the whole.

A stage is timed end to end, so its frames per second is what the pipeline
sustains; a model's inference is timed around its predict call alone, so its
frames per second is what the model, or the accelerator it runs on, can do. The
gap between the two is the work around the model.
"""

import json
import platform
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import sokil


@dataclass
class StageStats:
    name: str
    seconds: float = 0.0
    # the frames the stage went through, where that is meaningful
    frames: int | None = None

    @property
    def fps(self) -> float | None:
        return self.frames / self.seconds if self.frames and self.seconds else None


@dataclass
class ModelStats:
    """What sokil.models.Model measured about itself."""

    name: str
    path: str
    loads: int
    load_seconds: float
    inferences: int
    inference_seconds: float

    @property
    def seconds_per_frame(self) -> float | None:
        return self.inference_seconds / self.inferences if self.inferences else None

    @property
    def fps(self) -> float | None:
        per_frame = self.seconds_per_frame
        return 1 / per_frame if per_frame else None


@dataclass
class RunStats:
    stages: list[StageStats] = field(default_factory=list)
    models: list[ModelStats] = field(default_factory=list)
    # from the sokil package being imported to the run starting
    startup_seconds: float | None = None

    @contextmanager
    def stage(self, name: str) -> Generator[StageStats]:
        """
        Time a block as one stage; set frames on what it yields.

        A stage run again, as a grid search reruns hit detection, adds to its
        earlier row rather than starting another.
        """
        timed = StageStats(name)
        started = time.perf_counter()
        try:
            yield timed
        finally:
            timed.seconds = time.perf_counter() - started
            earlier = next((s for s in self.stages if s.name == name), None)
            if earlier is None:
                self.stages.append(timed)
            else:
                earlier.seconds += timed.seconds
                if timed.frames is not None:
                    earlier.frames = (earlier.frames or 0) + timed.frames

    def start(self) -> None:
        """Mark the run as starting, which ends its startup."""
        self.startup_seconds = time.perf_counter() - sokil.STARTED

    def add_model(self, name: str, model) -> None:
        """Take a sokil.models.Model's own measurements, if it has any."""
        if getattr(model, "loads", 0) or getattr(model, "inferences", 0):
            self.models.append(
                ModelStats(
                    name=name,
                    path=model.model_path,
                    loads=model.loads,
                    load_seconds=model.load_seconds,
                    inferences=model.inferences,
                    inference_seconds=model.inference_seconds,
                )
            )

    @property
    def total_seconds(self) -> float:
        """From the sokil package being imported to now."""
        return time.perf_counter() - sokil.STARTED

    def to_dict(self) -> dict:
        return {
            "host": platform.node(),
            "machine": platform.machine(),
            "total_seconds": round(self.total_seconds, 3),
            "startup_seconds": _round(self.startup_seconds),
            "stages": [
                {
                    **asdict(stage),
                    "seconds": _round(stage.seconds),
                    "fps": _round(stage.fps),
                }
                for stage in self.stages
            ],
            "models": [
                {
                    **asdict(model),
                    "load_seconds": _round(model.load_seconds),
                    "inference_seconds": _round(model.inference_seconds),
                    "seconds_per_frame": _round(model.seconds_per_frame, 4),
                    "fps": _round(model.fps),
                }
                for model in self.models
            ],
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def table(self):
        """The stats as a rich Table, for the console."""
        from rich.table import Table

        table = Table(title=f"Run stats (total {self.total_seconds:.1f}s)")
        table.add_column("Stage")
        table.add_column("Time", justify="right")
        table.add_column("Frames", justify="right")
        table.add_column("fps", justify="right")

        if self.startup_seconds is not None:
            table.add_row("startup (imports)", f"{self.startup_seconds:.2f}s", "", "")
        for stage in self.stages:
            table.add_row(
                stage.name,
                f"{stage.seconds:.2f}s",
                "" if stage.frames is None else str(stage.frames),
                _format_fps(stage.fps),
            )

        if self.models:
            table.add_section()
        for model in self.models:
            per_frame = model.seconds_per_frame
            table.add_row(f"{model.name} load", f"{model.load_seconds:.2f}s", "", "")
            table.add_row(
                f"{model.name} inference",
                "" if per_frame is None else f"{per_frame * 1000:.1f} ms/frame",
                str(model.inferences),
                _format_fps(model.fps),
            )
        return table


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _format_fps(fps: float | None) -> str:
    return "" if fps is None else f"{fps:.1f}"
