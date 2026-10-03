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
from .frames import extract_court_frame, extract_frames, load_undistorter

logger = logging.getLogger(__name__)

# Where the generated skeleton definition is written when a project needs
# creating by hand.
SKELETON_LABEL_PATH = Path("court-skeleton-label.json")

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


def court_skeleton_svg(sublabel_ids: dict[int, int] | None = None) -> str:
    """
    The drawing CVAT shows a skeleton's nodes on, laid out as the court is.

    :param sublabel_ids: vertex number -> the sublabel id CVAT assigned it.
        A node's data-label-id has to be that id, not the vertex number, or the
        editor cannot tell which node it is moving. The ids only exist once the
        CVAT never remaps these itself, so they have to be right in the
        definition it is given (see court_skeleton_label).
    """
    from ..court import CourtModel

    court = CourtModel()
    vertices = court.vertices
    sublabel_ids = sublabel_ids or {}

    # The drawing is only the picture an annotator picks nodes from, so it just
    # has to keep the court's proportions inside CVAT's 0-100 box. Y is flipped:
    # the court's origin is a baseline, and SVG's is the top.
    margin = 6.0
    span = 100.0 - 2 * margin
    scale = min(span / court.width, span / court.length)
    offset_x = (100.0 - court.width * scale) / 2
    offset_y = (100.0 - court.length * scale) / 2

    def place(vertex: tuple[float, float]) -> tuple[float, float]:
        x, y = vertex
        return offset_x + x * scale, 100.0 - (offset_y + y * scale)

    lines = []
    for start, end in court.edges:
        x1, y1 = place(vertices[start - 1])
        x2, y2 = place(vertices[end - 1])
        lines.append(
            f'<line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}" '
            f'data-type="edge" data-node-from="{start}" '
            f'data-node-to="{end}"></line>'
        )

    circles = []
    for number, vertex in enumerate(vertices, start=1):
        x, y = place(vertex)
        circles.append(
            f'<circle r="0.75" cx="{x:.4f}" cy="{y:.4f}" '
            f'data-type="element node" data-element-id="{number}" '
            f'data-node-id="{number}" '
            f'data-label-id="{sublabel_ids.get(number, number)}"></circle>'
        )

    return "\n".join(lines + circles)


def court_skeleton_label(label_name: str) -> list[dict]:
    """
    The skeleton label definition, in the shape CVAT's raw label editor takes.
    Generated from court model.
    """
    from ..court import CourtModel

    count = len(CourtModel().vertices)
    return [
        {
            "name": label_name,
            "type": "skeleton",
            "attributes": [],
            "sublabels": [
                {
                    "name": str(number),
                    "id": number,
                    "type": "points",
                    "attributes": [],
                }
                for number in range(1, count + 1)
            ],
            "svg": court_skeleton_svg({n: n for n in range(1, count + 1)}),
        }
    ]


def write_court_skeleton_label(path: Path, label_name: str) -> Path:
    """Write the skeleton definition for importing into CVAT by hand."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(court_skeleton_label(label_name), indent=2))
    logger.info("Wrote the court skeleton label -> %s", path)
    return path


def get_or_create_project(client, name: str, label_name: str, label_type: str):
    """Find a CVAT project by exact name, creating it if it doesn't exist."""
    from cvat_sdk import models

    for project in client.projects.list(name=name):
        if project.name == name:
            return project

    if label_type == "skeleton":
        # A skeleton cannot be built over the API. Therefore, we write the court SVG to a file
        path = write_court_skeleton_label(SKELETON_LABEL_PATH, label_name)
        raise ValueError(
            f"CVAT project {name!r} does not exist, and a skeleton project has "
            f"to be created by hand: CVAT only binds skeleton nodes correctly "
            f"through its own label editor. Create the project in CVAT, then "
            f"paste {path} into Labels > Raw."
        )

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
    median_frame: bool = True,
) -> int:
    """
    Extract frames from the video and create a CVAT task for them inside the
    model's project (labels are defined on the project, not the task).
    """
    from cvat_sdk import models

    spec = MODEL_SPECS[kind]
    undistorter = load_undistorter(camera_intrinsics_path, camera_dist_path)

    with tempfile.TemporaryDirectory() as staging:
        if kind == ModelKind.COURT_KEYPOINTS:
            frame_paths = [
                extract_court_frame(
                    video_path, Path(staging), undistorter, median=median_frame
                )
            ]
        else:
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

            if pre_annotate_model_path and kind == ModelKind.COURT_KEYPOINTS:
                logger.warning(
                    "Pre-annotating court keypoints is not supported yet; "
                    "the task was created empty"
                )
            elif pre_annotate_model_path:
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
