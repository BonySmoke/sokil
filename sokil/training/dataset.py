"""
Turning a CVAT export into a YOLO dataset on disk.
"""

import logging
import random
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2

from .config import IMAGE_SUFFIXES, MODEL_SPECS, TRAINING_ROOT, ModelKind
from .cvat import list_completed_task_ids, make_cvat_client

logger = logging.getLogger(__name__)


def _parse_cvat_images_xml(xml_path: Path, label_name: str) -> list[dict]:
    """
    Parse a 'CVAT for images 1.1' annotations.xml into
    [{"name", "width", "height", "objects": [...]}, ...] with only annotated
    images included.
    """
    root = ET.parse(xml_path).getroot()
    samples = []

    for image in root.findall("image"):
        width = int(image.get("width"))
        height = int(image.get("height"))
        objects = []

        for box in image.findall("box"):
            if box.get("label") != label_name:
                continue
            objects.append(
                {
                    "type": "box",
                    "points": (
                        float(box.get("xtl")),
                        float(box.get("ytl")),
                        float(box.get("xbr")),
                        float(box.get("ybr")),
                    ),
                }
            )

        for polygon in image.findall("polygon"):
            if polygon.get("label") != label_name:
                continue
            points = [
                tuple(map(float, p.split(",")))
                for p in polygon.get("points").split(";")
            ]
            objects.append({"type": "polygon", "points": points})

        if objects:
            samples.append(
                {
                    "name": Path(image.get("name")).name,
                    "width": width,
                    "height": height,
                    "objects": objects,
                }
            )

    return samples


def _to_yolo_label_lines(sample: dict) -> list[str]:
    width, height = sample["width"], sample["height"]
    lines = []

    for obj in sample["objects"]:
        if obj["type"] == "box":
            xtl, ytl, xbr, ybr = obj["points"]
            cx = ((xtl + xbr) / 2) / width
            cy = ((ytl + ybr) / 2) / height
            nw = (xbr - xtl) / width
            nh = (ybr - ytl) / height
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        else:
            coords = []
            for x, y in obj["points"]:
                coords.append(f"{x / width:.6f}")
                coords.append(f"{y / height:.6f}")
            lines.append("0 " + " ".join(coords))

    return lines


def _thin_by_frame_gap(samples: list[dict], min_frame_gap: int) -> list[dict]:
    """
    Keep frames at least min_frame_gap source frames apart, based on the frame
    index every exported name ends with. Tasks whose frames are already that far
    apart come back untouched, so only the densely annotated ones are thinned.
    """
    indexed = []
    for sample in samples:
        match = re.search(r"_(\d+)$", Path(sample["name"]).stem)
        if match is None:
            logger.warning(
                "Frame index missing from %r; skipping frame-gap thinning",
                sample["name"],
            )
            return samples
        indexed.append((int(match.group(1)), sample))

    kept = []
    last_index = None
    for index, sample in sorted(indexed, key=lambda pair: pair[0]):
        if last_index is None or index - last_index >= min_frame_gap:
            kept.append(sample)
            last_index = index

    return kept


def _partition_count(size: float, total: int, name: str) -> int:
    """
    Resolve a split size to a sample count.

    Values below 1 are a fraction of the dataset; 1 and above are an absolute
    count. Frames from a rally are near-identical, so a percentage of a growing
    export keeps inflating a validation set that stopped adding information
    long ago — an absolute count is usually what you want.
    """
    if size <= 0:
        return 0

    count = int(size) if size >= 1 else int(total * size)
    if count > total:
        logger.warning(
            "%s=%s exceeds the %d exported frames; using all of them", name, size, total
        )
    return min(count, total)


def _split(
    samples: list, seed: int, valid_size: float = 0.2, test_size: float = 0.2
) -> dict[str, list]:
    """
    Deterministic train/valid/test split; the remainder after valid and test
    goes to train (see _partition_count for how the sizes are read).
    """
    samples = sorted(samples, key=lambda s: s["name"])
    random.Random(seed).shuffle(samples)
    total = len(samples)

    n_test = _partition_count(test_size, total, "test_size")
    n_valid = _partition_count(valid_size, total, "valid_size")

    if n_test + n_valid >= total:
        raise ValueError(
            f"valid_size and test_size take {n_test + n_valid} of {total} "
            f"exported frames, leaving nothing to train on"
        )

    splits = {
        "test": samples[:n_test],
        "valid": samples[n_test : n_test + n_valid],
        "train": samples[n_test + n_valid :],
    }
    logger.info(
        "Split %d frames: train=%d valid=%d test=%d",
        total,
        len(splits["train"]),
        len(splits["valid"]),
        len(splits["test"]),
    )
    return splits


def export_dataset(
    kind: ModelKind,
    task_ids: list[int] | None = None,
    project_name: str | None = None,
    grayscale: bool | None = None,
    override: bool = False,
    seed: int = 1,
    min_frame_gap: int = 1,
    valid_size: float = 0.2,
    test_size: float = 0.2,
) -> Path:
    """
    Export CVAT tasks and build a YOLO dataset under
    training/<model>/datasets/. Returns the path to data.yaml.

    Without explicit task_ids, every task in the model's project whose jobs are
    all marked completed is exported.

    min_frame_gap thins each task down to frames at least that many source
    frames apart: tasks annotated on consecutive video frames contribute
    near-duplicates that cost training time without adding much signal, while
    tasks already extracted at a wider stride are left untouched.
    """
    spec = MODEL_SPECS[kind]
    if grayscale is None:
        grayscale = spec.grayscale

    dataset_dir = TRAINING_ROOT / kind.value / "datasets"
    if dataset_dir.exists() and override:
        shutil.rmtree(dataset_dir)

    with tempfile.TemporaryDirectory() as tmp, make_cvat_client() as client:
        tmp = Path(tmp)
        samples = []

        if not task_ids:
            project = project_name or spec.project
            task_ids = list_completed_task_ids(client, project)
            if not task_ids:
                raise ValueError(f"No tasks with completed jobs in project {project!r}")
            logger.info(
                "Exporting %d completed task(s) from %r: %s",
                len(task_ids),
                project,
                task_ids,
            )

        for task_id in task_ids:
            task = client.tasks.retrieve(task_id)
            zip_path = tmp / f"task_{task_id}.zip"
            task.export_dataset("CVAT for images 1.1", zip_path, include_images=True)

            extract_dir = tmp / f"task_{task_id}"
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            task_samples = _parse_cvat_images_xml(
                extract_dir / "annotations.xml", spec.label
            )

            annotated = len(task_samples)
            if min_frame_gap > 1:
                task_samples = _thin_by_frame_gap(task_samples, min_frame_gap)

            # Match on the stem: tasks created from mounted frame folders list
            # extension-less names in the XML while the images export as .PNG.
            images_by_stem = {
                p.stem: p
                for p in extract_dir.rglob("*")
                if p.suffix.lower() in IMAGE_SUFFIXES
            }
            safe_task_name = re.sub(r"[^\w.-]+", "_", task.name)
            for sample in task_samples:
                stem = Path(sample["name"]).stem
                sample["image_path"] = images_by_stem[stem]
                # Tasks created directly from videos or mounted frame folders
                # (before frames were uploaded individually) export generic
                # frame_<idx> names; prefix them with the task name so frames
                # from different tasks don't collide and map back to their source.
                match = re.fullmatch(r"frame_(\d+)", stem)
                if match:
                    stem = f"{safe_task_name}_{match.group(1)}"
                sample["name"] = f"{stem}.jpg"

            logger.info(
                "Task %d: %d annotated frames, %d kept",
                task_id,
                annotated,
                len(task_samples),
            )
            samples.extend(task_samples)

        if not samples:
            raise ValueError("No annotated frames found in the given tasks")

        splits = _split(samples, seed, valid_size=valid_size, test_size=test_size)
        for partition, partition_samples in splits.items():
            images_dir = dataset_dir / "images" / partition
            labels_dir = dataset_dir / "labels" / partition
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            for sample in partition_samples:
                img = cv2.imread(str(sample["image_path"]))
                if grayscale:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(images_dir / sample["name"]), img)

                label_path = labels_dir / (Path(sample["name"]).stem + ".txt")
                label_path.write_text("\n".join(_to_yolo_label_lines(sample)) + "\n")

            logger.info("%s: %d frames", partition, len(partition_samples))

    data_yaml = dataset_dir / "data.yaml"
    # nc is written explicitly even though it is derivable from names: only some
    # model families infer it, and one that does not keeps its pretrained class
    # count, then fails when the checkpoint's names no longer match it
    data_yaml.write_text(
        f"path: {dataset_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        f"  0: {spec.label}\n"
    )
    return data_yaml
