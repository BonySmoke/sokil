"""
Recording an evaluation run so two runs can be compared.

A metric without the conditions that produced it cannot be compared with
anything later, so the record carries what was measured, what it was measured
with, and what it was measured on: the code, the thresholds, the dataset and
the machine. The hardware timings belong in the same record for the same
reason.
"""

import hashlib
import json
import logging
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def git_sha() -> str | None:
    """The commit the code is at, or None outside a checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout.strip() or None


def git_dirty() -> bool | None:
    """Whether the checkout has uncommitted changes, which makes the sha a lie."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(completed.stdout.strip())


def environment() -> dict:
    """
    Get details for the environment where the evaluation is performed.
    """
    import cv2
    import numpy as np

    return {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def dataset_state(dataset_path: Path, clips: list[str]) -> dict:
    """
    What the figures were measured against, closely enough to notice a change.

    A digest per table, because the tables are what the result rests on: a
    re-derived landing position moves every distance in the report, and without
    this the record would show the same path, the same clips, and no reason.
    """
    tables = {}
    for table in ("clips", "events", "court_points", "tracks"):
        path = dataset_path / f"{table}.csv"
        if not path.exists():
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        tables[table] = {
            "sha256": digest,
            "rows": max(len(path.read_text().splitlines()) - 1, 0),
        }

    return {"path": str(dataset_path), "clips": clips, "tables": tables}


def record(
    results: list,
    hit_config,
    dataset_path: Path,
    clips: list[str],
    totals: dict,
) -> dict:
    """The full record of one evaluation run."""
    return {
        "at": datetime.now(UTC).isoformat(),
        "git": {"sha": git_sha(), "dirty": git_dirty()},
        "environment": environment(),
        # The thresholds under test. A figure quoted without them says nothing
        # about which configuration produced it.
        "hit_config": asdict(hit_config),
        "dataset": dataset_state(Path(dataset_path), clips),
        "totals": totals,
        "clips": [asdict(result) for result in results],
    }


def write(record: dict, directory: Path) -> Path:
    """
    Write the record under a name that sorts by time and names the commit.

    One file per run rather than one appended log: runs are compared by reading
    two of them, and a file nobody rewrites cannot be corrupted by the next run.
    """
    directory.mkdir(parents=True, exist_ok=True)

    sha = (record["git"]["sha"] or "nogit")[:12]
    stamp = record["at"].replace(":", "").replace("-", "")[:15]
    path = directory / f"{stamp}-{sha}.json"
    path.write_text(json.dumps(record, indent=2, default=str))

    logger.info("Wrote the evaluation record -> %s", path)
    return path
