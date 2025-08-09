"""
Talking to CVAT: projects, annotation tasks, and optional pre-annotation.

Connection settings come from the environment (or a .env file): CVAT_URL,
CVAT_USERNAME, CVAT_PASSWORD.
"""

import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from .config import MODEL_SPECS, ModelKind
from .frames import extract_frames, load_undistorter

logger = logging.getLogger(__name__)

load_dotenv()


def make_cvat_client():
    from cvat_sdk import make_client

    url = os.environ.get("CVAT_URL", "http://localhost:8080")
    username = os.environ.get("CVAT_USERNAME")
    password = os.environ.get("CVAT_PASSWORD")

    if not username or not password:
        raise ValueError(
            "CVAT_USERNAME and CVAT_PASSWORD must be set (see .env.example)"
        )

    return make_client(host=url, credentials=(username, password))


def get_or_create_project(client, name: str, label_name: str, label_type: str):
    """Find a CVAT project by exact name, creating it if it doesn't exist."""
    from cvat_sdk import models

    for project in client.projects.list(name=name):
        if project.name == name:
            return project

    project = client.projects.create(
        models.ProjectWriteRequest(
            name=name,
            labels=[models.PatchedLabelRequest(name=label_name, type=label_type)],
        )
    )
    logger.info("Created CVAT project %d: %s", project.id, name)
    return project


def _enum_value(value) -> str:
    """CVAT enums come back as wrapper objects; unwrap them to plain strings."""
    return str(getattr(value, "value", value))


def find_project(client, name: str):
    for project in client.projects.list(name=name):
        if project.name == name:
            return project
    raise ValueError(f"CVAT project {name!r} not found")


def list_completed_task_ids(client, project_name: str) -> list[int]:
    """
    Task ids in the project whose annotation jobs are all marked completed.
    CVAT has no task-level "done" flag, so completion is derived from the jobs
    (ground truth jobs are ignored).
    """
    project = find_project(client, project_name)
    task_ids = []

    for task in client.tasks.list(project_id=project.id):
        jobs = [
            job for job in task.get_jobs() if _enum_value(job.type) != "ground_truth"
        ]
        if jobs and all(_enum_value(job.state) == "completed" for job in jobs):
            task_ids.append(task.id)
        else:
            logger.debug(
                "Skipping task %d (%s): jobs not completed", task.id, task.name
            )

    return sorted(task_ids)


def _simplify_polygon(points: np.ndarray, epsilon_ratio: float = 0.002) -> np.ndarray:
    """Reduce mask contours to editable polygons (masks have hundreds of points)."""
    contour = points.reshape(-1, 1, 2).astype(np.float32)
    epsilon = epsilon_ratio * cv2.arcLength(contour, closed=True)
    return cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)


def _pre_annotate_shapes(
    frame_paths: list[Path], model_path: str, kind: ModelKind, label_id: int
) -> list:
    """Run an existing model on the frames and build CVAT shapes."""
    from cvat_sdk import models

    from sokil.models import Segmenter, ShuttleDetector

    is_shuttle = kind == ModelKind.SHUTTLE
    model = (ShuttleDetector if is_shuttle else Segmenter)(model_path)
    shapes = []

    for frame_number, path in enumerate(frame_paths):
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("Could not read %s, skipping pre-annotation", path)
            continue

        if is_shuttle:
            detection = model.best_box(image)
            if detection is None:
                continue
            (x1, y1, x2, y2), _ = detection
            shapes.append(
                models.LabeledShapeRequest(
                    type=models.ShapeType("rectangle"),
                    frame=frame_number,
                    label_id=label_id,
                    points=[x1, y1, x2, y2],
                )
            )
        else:
            for polygon in model.polygons(image):
                polygon = _simplify_polygon(polygon)
                if len(polygon) < 3:
                    continue
                shapes.append(
                    models.LabeledShapeRequest(
                        type=models.ShapeType("polygon"),
                        frame=frame_number,
                        label_id=label_id,
                        points=[float(c) for point in polygon for c in point],
                    )
                )

    return shapes


def create_cvat_task(
    video_path: Path,
    kind: ModelKind,
    every_n_frames: int = 30,
    camera_intrinsics_path: str | None = None,
    camera_dist_path: str | None = None,
    pre_annotate_model_path: str | None = None,
    project_name: str | None = None,
) -> int:
    """
    Extract frames from the video and create a CVAT task for them inside the
    model's project (labels are defined on the project, not the task).
    """
    from cvat_sdk import models

    spec = MODEL_SPECS[kind]
    undistorter = load_undistorter(camera_intrinsics_path, camera_dist_path)

    with tempfile.TemporaryDirectory() as staging:
        frame_paths = extract_frames(
            video_path, Path(staging), every_n_frames, undistorter
        )

        if not frame_paths:
            raise ValueError(f"No frames extracted from {video_path}")

        with make_cvat_client() as client:
            project = get_or_create_project(
                client,
                project_name or spec.project,
                spec.label,
                spec.label_type,
            )

            task = client.tasks.create_from_data(
                spec=models.TaskWriteRequest(
                    name=f"{kind.value}-{video_path.stem}",
                    project_id=project.id,
                ),
                resources=frame_paths,
                data_params={
                    "image_quality": 100,
                    "sorting_method": "lexicographical",
                },
            )
            logger.info(
                "Created CVAT task %d: %s (project %r)",
                task.id,
                task.name,
                project.name,
            )

            if pre_annotate_model_path:
                labels = task.get_labels()
                label_id = next(
                    (label.id for label in labels if label.name == spec.label),
                    labels[0].id,
                )
                shapes = _pre_annotate_shapes(
                    frame_paths, pre_annotate_model_path, kind, label_id
                )
                if shapes:
                    task.update_annotations(
                        models.PatchedLabeledDataRequest(shapes=shapes)
                    )
                logger.info("Uploaded %d pre-annotation shapes", len(shapes))

            return task.id
