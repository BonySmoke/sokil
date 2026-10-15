"""
Compile YOLO weights to a Hailo HEF, calibrating on a dataset sokil exported.

Runs inside the image built from this directory, not in the project's
environment: the Hailo suite brings its own, older Python, so sokil itself is
not installed there and this script depends on ultralytics alone.

The result is the directory ultralytics writes, <stem>-<arch>_hailo_model/,
holding the .hef and the metadata that lets YOLO() load it on the device.
"""

# The suite's Python version is not ours to choose; keep the annotations from
# being evaluated in case it predates builtin generics.
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS

HAILO_ARCHS = ("hailo8", "hailo8l", "hailo10h", "hailo15h", "hailo15l")


def relocate_dataset(data_yaml: Path, workdir: Path) -> tuple[Path, dict]:
    """
    A copy of data.yaml whose path points where the dataset is mounted.

    export-cvat-dataset writes the absolute path of the machine it ran on, which
    does not exist in the container. The dataset always sits beside its
    data.yaml, so that directory is the path wherever it is mounted.

    :return: the rewritten data.yaml and its contents
    """
    config = yaml.safe_load(data_yaml.read_text())
    config["path"] = str(data_yaml.parent.resolve())
    relocated = workdir / data_yaml.name
    relocated.write_text(yaml.safe_dump(config, sort_keys=False))
    return relocated, config


def count_images(config: dict, split: str) -> int:
    """How many images a split holds, for turning a target count into a fraction."""
    if split not in config:
        raise SystemExit(f"data.yaml has no {split!r} split")
    split_dir = Path(config["path"]) / config[split]
    if not split_dir.is_dir():
        raise SystemExit(f"{split_dir} is not a directory of images")
    return sum(
        1 for f in split_dir.rglob("*") if f.suffix.lower().lstrip(".") in IMG_FORMATS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--weights", type=Path, required=True, help="The .pt weights to compile"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="data.yaml of the dataset to calibrate on",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/out"), help="Where to write the result"
    )
    parser.add_argument(
        "--arch",
        choices=HAILO_ARCHS,
        default="hailo8",
        help="The accelerator the HEF will run on",
    )
    # Must match expects.imgsz in sokil/models.yaml: the HEF's input size is fixed.
    parser.add_argument("--imgsz", type=int, default=640)
    # The exporter calibrates on val by default, which is a few hundred frames
    # at most here; train is where the volume and the variety are.
    parser.add_argument(
        "--split", default="train", help="Dataset split to calibrate on"
    )
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=1024,
        help="How many images of the split to calibrate on; Hailo recommends at least 1024. "
        "The exporter fine-tunes on every one, so more costs time without much gain",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"No such file: {args.weights}")
    if not args.data.is_file():
        raise SystemExit(f"No such file: {args.data}")

    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as workdir:
        data_yaml, config = relocate_dataset(args.data, Path(workdir))

        available = count_images(config, args.split)
        fraction = min(1.0, args.calibration_images / available)
        print(
            f"Calibrating on {min(args.calibration_images, available)} of {available} "
            f"{args.split} images (fraction={fraction:.4f})",
            file=sys.stderr,
        )

        # The exporter writes next to the weights, and the repo is mounted
        # read-only, so compile a copy in the output. Naming it after the arch
        # keeps a hailo8 and a hailo8l build of the same weights apart.
        weights = args.output / f"{args.weights.stem}-{args.arch}.pt"
        shutil.copy2(args.weights, weights)
        try:
            exported = YOLO(str(weights)).export(
                format="hailo",
                name=args.arch,
                imgsz=args.imgsz,
                data=str(data_yaml),
                split=args.split,
                fraction=fraction,
            )
        finally:
            weights.unlink(missing_ok=True)

    print(exported)


if __name__ == "__main__":
    main()
