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
from collections import defaultdict, deque
from pathlib import Path

import cv2

from .config import IMAGE_SUFFIXES, MODEL_SPECS, TRAINING_ROOT, ModelKind
from .cvat import list_completed_task_ids, make_cvat_client

logger = logging.getLogger(__name__)


def _frame_index(name: str) -> int | None:
    """
    The frame number is the end of the exporter file name: <clip_id>_<frame>.jpg
    """
    match = re.search(r"_(\d+)$", Path(name).stem)
    return int(match.group(1)) if match else None


def _clip_id(name: str, fallback: str) -> str:
    """
    Which clip an exported frame came from.
    Clip naming convention is: <clip_id>_<frame>.jpg
    """
    parent = Path(name).parent
    if parent != Path("."):
        return parent.parts[0]

    stem = Path(name).stem
    if _frame_index(stem) is not None:
        prefix = stem.rsplit("_", 1)[0]
        # "frame" is what a bare frame_000123.jpg leaves behind, and it names
        # no clip: every task would claim the same one. Those are the names
        # the fallback exists for.
        if prefix and prefix != "frame":
            return prefix

    return fallback


def _find_image(name: str, by_path: dict[str, Path], by_stem: dict[str, Path]) -> Path:
    """
    The exported image an annotation refers to, by relative path where the
    export kept one and by bare stem otherwise.
    """
    key = str(Path(name).with_suffix(""))
    if key in by_path:
        return by_path[key]

    stem = Path(name).stem
    if stem in by_stem:
        return by_stem[stem]

    raise ValueError(f"No exported image for annotated frame {name!r}")


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
                    "name": image.get("name"),
                    "width": width,
                    "height": height,
                    "objects": objects,
                }
            )

    return samples


def _identify(
    samples: list[dict],
    task_name: str,
    task_id: int,
    images_by_path: dict[str, Path],
    images_by_stem: dict[str, Path],
) -> list[dict]:
    """
    Give every sample its metadata such as clip ID, task ID, etc.
    """
    for sample in samples:
        name = sample["name"]
        stem = Path(name).stem
        clip_id = _clip_id(name, task_name)

        sample["clip_id"] = clip_id
        sample["task_id"] = task_id
        sample["image_path"] = _find_image(name, images_by_path, images_by_stem)

        # Handle an edge case when the video was not uploaded to CVAT but mounted from the local filesystem
        match = re.fullmatch(r"frame_(\d+)", stem)
        if match:
            stem = f"{clip_id}_{match.group(1)}"
        sample["name"] = f"{stem}.jpg"

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

    Gaps are measured within a clip: frame numbers restart with every clip.
    """
    by_clip = defaultdict(list)
    kept = []

    for sample in samples:
        index = _frame_index(sample["name"])
        if index is None:
            logger.warning(
                "Frame index missing from %r; skipping frame-gap thinning",
                sample["name"],
            )
            return samples

        by_clip[sample["clip_id"]].append((index, sample))

    for clip_id in sorted(by_clip):
        last_index = None

        for index, sample in sorted(by_clip[clip_id], key=lambda pair: pair[0]):
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


def _cap_by_spread(samples: list, size: float, name: str) -> list:
    """
    Thin a partition to `size` frames, spread as widely across each clip as
    they go.

    Validating on three thousand frames of one rally is not three thousand
    pieces of evidence: consecutive frames are near-identical, so the figure
    looks precise while resting on very little. Frames are therefore kept at an
    even stride rather than taken from the front, and each clip keeps a share
    proportional to its length so one long clip cannot crowd the others out.

    Sizes read as in _partition_count, but of this partition rather than of the
    whole export: the subsets have already decided what is in it.
    """
    if size <= 0 or not samples:
        return samples

    target = _partition_count(size, len(samples), name)
    if target >= len(samples):
        return samples

    by_clip = defaultdict(list)
    for sample in samples:
        by_clip[sample["clip_id"]].append(sample)

    kept = []
    for clip_id in sorted(by_clip):
        clip_samples = sorted(
            by_clip[clip_id], key=lambda sample: _frame_index(sample["name"]) or 0
        )
        quota = max(1, round(target * len(clip_samples) / len(samples)))
        quota = min(quota, len(clip_samples))

        stride = len(clip_samples) / quota
        kept += [clip_samples[int(index * stride)] for index in range(quota)]

    logger.info(
        "%s: thinned %d frames to %d, spread across %d clip(s)",
        name,
        len(samples),
        len(kept),
        len(by_clip),
    )
    return kept


def _split_by_subset(
    samples: list,
    subsets: dict[str, str],
    valid_size: float = 0.0,
    test_size: float = 0.0,
) -> dict[str, list]:
    """
    Assign whole clips by the subset their CVAT task declares.

    Where the split is stated in CVAT it is stated for a reason — which angles
    a model is tested on is a judgement about coverage, and a shuffle cannot
    make it. A clip whose task carries no subset is an unanswered question, so
    it is refused rather than defaulted.
    """
    from ..dataset.splits import SUBSET_SPLITS, SplitError

    by_clip = defaultdict(list)
    for sample in samples:
        by_clip[sample["clip_id"]].append(sample)

    splits: dict[str, list] = {name: [] for name in SUBSET_SPLITS.values()}
    unassigned, unknown = [], {}

    for clip_id in sorted(by_clip):
        key = (subsets.get(clip_id) or "").strip().lower()
        if not key:
            unassigned.append(clip_id)
            continue

        split = SUBSET_SPLITS.get(key)
        if split is None:
            unknown[clip_id] = subsets[clip_id]
            continue

        splits[split].extend(by_clip[clip_id])

    if unassigned or unknown:
        detail = []
        if unassigned:
            detail.append(f"no subset in CVAT: {', '.join(unassigned)}")
        if unknown:
            detail.append(
                "unrecognised subset(s): "
                + ", ".join(f"{c}={v!r}" for c, v in unknown.items())
            )
        raise SplitError(
            "Cannot split by CVAT subsets — " + "; ".join(detail) + ". Set each "
            "task's Subset in CVAT, or pass --seed-split to shuffle instead."
        )

    if not splits["train"]:
        raise SplitError(
            "No clip is in the Train subset, so there is nothing to train on"
        )

    # The subsets say which clips; these say how much of them to measure on.
    splits["valid"] = _cap_by_spread(splits["valid"], valid_size, "valid")
    splits["test"] = _cap_by_spread(splits["test"], test_size, "test")

    for name, partition in splits.items():
        clips = sorted({sample["clip_id"] for sample in partition})
        logger.info(
            "%s: %d clip(s), %d frames %s", name, len(clips), len(partition), clips
        )

    return splits


def _split(
    samples: list, seed: int, valid_size: float = 0.2, test_size: float = 0.2
) -> dict[str, list]:
    """
    Deterministic train/valid/test split, drawn on whole clips.

    Consecutive frames of one rally are near-identical, so splitting frames
    individually puts a frame's twin in both train and validation: the score
    that comes back measures recall of a frame the model has already seen, and
    it stops responding honestly to anything you change. Keeping a clip whole
    costs some control over the exact partition sizes, which is the cheaper
    thing to give up.

    Sizes are targets rather than exact counts (see _partition_count for how
    they are read): clips are taken until a partition reaches its target, so it
    lands on or just above it.
    """
    by_clip = defaultdict(list)
    for sample in samples:
        by_clip[sample["clip_id"]].append(sample)

    clip_ids = sorted(by_clip)
    random.Random(seed).shuffle(clip_ids)

    total = len(samples)
    targets = {
        "test": _partition_count(test_size, total, "test_size"),
        "valid": _partition_count(valid_size, total, "valid_size"),
    }

    remaining = deque(clip_ids)
    splits = {}

    for partition, target in targets.items():
        chosen = []
        assigned = 0
        # One clip is always held back: a partition that swallowed the last one
        # would leave nothing to train on.
        while assigned < target and len(remaining) > 1:
            clip_id = remaining.popleft()
            chosen.append(clip_id)
            assigned += len(by_clip[clip_id])

        if target > 0 and not chosen:
            raise ValueError(
                f"{partition}_size asks for {target} of {total} frames, but "
                f"only {len(clip_ids)} clip(s) were exported and a clip cannot "
                f"be split across partitions. Annotate more clips, or set "
                f"--{partition}-size 0."
            )

        splits[partition] = chosen

    splits["train"] = list(remaining)

    for partition, clips in splits.items():
        logger.info(
            "%s: %d clip(s), %d frames %s",
            partition,
            len(clips),
            sum(len(by_clip[clip_id]) for clip_id in clips),
            sorted(clips),
        )

    return {
        partition: [sample for clip_id in clips for sample in by_clip[clip_id]]
        for partition, clips in splits.items()
    }


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
    seed_split: bool = False,
) -> Path:
    """
    Export CVAT tasks and build a YOLO dataset.

    Without explicit task_ids, every task in the model's project whose jobs are
    all marked completed is exported.

    min_frame_gap thins each task down to frames at least that many source frames apart.

    The split is drawn on whole clips.

    By default each clip goes where its CVAT task's Subset says, e.g. Train, Test, Validation.
    Pass `seed_split` to shuffle clips by seed instead.

    :return: the path to YOLO data.yaml
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

            # Match on the stem: tasks created from mounted frame folders list
            # extension-less names in the XML while the images export as .PNG.
            #
            # Keyed by the path below the export too, because two clips in one
            # task can each hold a frame_000123 and a bare stem cannot tell
            # them apart.
            images = [
                p for p in extract_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
            ]
            images_by_stem = {p.stem: p for p in images}
            images_by_path = {
                str(p.relative_to(extract_dir).with_suffix("")): p for p in images
            }

            _identify(
                task_samples,
                re.sub(r"[^\w.-]+", "_", task.name),
                task_id,
                images_by_path,
                images_by_stem,
            )

            # Thinning groups by clip, so it has to follow identification: a
            # sample does not know which clip it belongs to until then.
            if min_frame_gap > 1:
                task_samples = _thin_by_frame_gap(task_samples, min_frame_gap)

            logger.info(
                "Task %d: %d annotated frames, %d kept",
                task_id,
                annotated,
                len(task_samples),
            )
            samples.extend(task_samples)

        if not samples:
            raise ValueError("No annotated frames found in the given tasks")

        if seed_split:
            splits = _split(samples, seed, valid_size=valid_size, test_size=test_size)
        else:
            subsets = {}
            for task_id in task_ids:
                task = client.tasks.retrieve(task_id)
                for clip_id in {
                    sample["clip_id"]
                    for sample in samples
                    if sample["task_id"] == task_id
                }:
                    subsets[clip_id] = getattr(task, "subset", "") or ""
            splits = _split_by_subset(
                samples, subsets, valid_size=valid_size, test_size=test_size
            )

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
