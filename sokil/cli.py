import logging
from typing import Annotated

import typer
from rich.logging import RichHandler

from sokil.court import CourtCameraMode, CourtConfig, CourtModelType

app = typer.Typer()

models_app = typer.Typer(help="Manage the trained weights")
app.add_typer(models_app, name="models")

calibrate_app = typer.Typer(help="Measure the camera lens, once per camera")
app.add_typer(calibrate_app, name="calibrate")


# third-party libraries stay at WARNING so our own output remains readable
NOISY_LOGGERS = ("ultralytics", "matplotlib", "torch", "urllib3", "PIL", "requests")


def configure_logging(verbose: bool, keep: tuple[str, ...] = ()):
    """Configure logging; loggers named in `keep` are left at the chosen level."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=verbose)],
    )
    for noisy in NOISY_LOGGERS:
        if noisy not in keep:
            logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def review(
    video_path: Annotated[str, typer.Option(help="The path to the video to review")],
    output_path: Annotated[
        str, typer.Option(help="The output path of the review video")
    ],
    shuttle_detector_model_path: Annotated[
        str | None,
        typer.Option(help="The YOLO model for shuttlecock detection"),
    ] = None,
    court_segmentation_model_path: Annotated[
        str | None,
        typer.Option(help="The YOLO model for court segmentation"),
    ] = None,
    court_keypoint_model_path: Annotated[
        str | None, typer.Option(help="The YOLO model for court keypoint detection")
    ] = None,
    court_camera_mode: Annotated[
        CourtCameraMode,
        typer.Option(help="The side of the camera the camera is facing"),
    ] = CourtCameraMode.FULL,
    court_model_type: Annotated[
        CourtModelType,
        typer.Option(
            help="The court type we are reviewing. Depends on the discripline, either singles or doubles"
        ),
    ] = CourtModelType.DOUBLES,
    camera_intrinsics_path: Annotated[
        str | None,
        typer.Option(
            help="The path to the camera intrinsics matrix to undistort frames"
        ),
    ] = None,
    camera_dist_path: Annotated[
        str | None,
        typer.Option(
            help="The path to the camera distortion matrix to undistort frames"
        ),
    ] = None,
    court_homography_path: Annotated[
        str | None,
        typer.Option(
            help="Use this image->court homography instead of solving the court "
            "from the video. Takes whatever matrix the file holds, as-is"
        ),
    ] = None,
    save_court_homography_path: Annotated[
        str | None,
        typer.Option(
            help="After solving, write the court homography here so later runs "
            "can reuse it with --court-homography-path"
        ),
    ] = None,
    output_fps: Annotated[
        float | None,
        typer.Option(
            help="Frame rate of the review video (defaults to the source rate). "
            "A lower rate replays every frame more slowly: 30 on 120 fps "
            "footage gives 4x slow motion"
        ),
    ] = None,
    court_sample_seconds: Annotated[
        float,
        typer.Option(
            help="Spacing of the frames the court lines are detected on. The "
            "solver keeps a floor on the number of frames regardless, so lower "
            "this only to sample a long clip more densely"
        ),
    ] = 1.0,
    explain: Annotated[
        bool,
        typer.Option(help="Explain the stages of the review visually"),
    ] = False,
    verbose: Annotated[
        bool, typer.Option(help="Enable debug logging (per-hit verdicts, etc.)")
    ] = False,
):
    from sokil.court import CourtSolverConfig, load_homography, save_homography
    from sokil.models import PoseEstimator, Segmenter
    from sokil.render import RenderConfig
    from sokil.review import Review
    from sokil.settings import ModelName, ensure_model_path, model_spec

    configure_logging(verbose)

    if shuttle_detector_model_path is None:
        shuttle_detector_model_path = str(ensure_model_path(ModelName.SHUTTLE))
    if court_segmentation_model_path is None:
        court_segmentation_model_path = str(ensure_model_path(ModelName.COURT))

    court_config = CourtConfig(
        court_camera_mode=court_camera_mode,
        court_model_type=court_model_type,
    )

    if output_fps is not None and output_fps <= 0:
        raise typer.BadParameter("--output-fps must be positive")

    if court_sample_seconds <= 0:
        raise typer.BadParameter("--court-sample-seconds must be positive")

    court_segmentation_model = Segmenter(court_segmentation_model_path)

    # The weights declare the frames they were trained on, so the pipeline
    # honours that instead of assuming. A pair that disagrees can only be fed
    # one way, so the shuttle detector wins: it is the one that runs per frame.
    shuttle_expects = model_spec(ModelName.SHUTTLE).expects
    court_expects = model_spec(ModelName.COURT).expects
    if shuttle_expects.get("grayscale") != court_expects.get("grayscale"):
        logging.getLogger(__name__).warning(
            "The models disagree on grayscale input; using the shuttle "
            "detector's expectation (%s)",
            shuttle_expects.get("grayscale", True),
        )

    court_keypoint_model = None
    if court_keypoint_model_path:
        court_keypoint_model = PoseEstimator(court_keypoint_model_path)

    review = Review(
        video_path=video_path,
        court_detector_kwargs={
            "segmentation_model": court_segmentation_model,
            "keypoint_model": court_keypoint_model,
            "court_config": court_config,
        },
        grayscale=shuttle_expects.get("grayscale", True),
        shuttle_detector_kwargs={
            "model": shuttle_detector_model_path,
        },
        camera_intrinsics_path=camera_intrinsics_path,
        camera_dist_path=camera_dist_path,
        court_homography=(
            load_homography(court_homography_path) if court_homography_path else None
        ),
        render_config=RenderConfig(output_fps=output_fps, explain=explain),
        court_solver_config=CourtSolverConfig(sample_seconds=court_sample_seconds),
    )

    review.preprocess()

    if save_court_homography_path:
        if review.court is None:
            raise typer.BadParameter(
                "No court was established, so there is no homography to save"
            )
        save_homography(review.court, save_court_homography_path)

    review.set_frame_hit_candidates()
    review.set_hit_positions()
    review.set_track_points()

    review.video_to_file(output_path)


@models_app.command("download")
def models_download(
    force: Annotated[
        bool, typer.Option(help="Re-download even if the weights are already cached")
    ] = False,
    verbose: Annotated[bool, typer.Option(help="Enable debug logging")] = False,
):
    """Download the published weights into the per-user cache."""
    from sokil.settings import download_models

    configure_logging(verbose)

    for name, path in download_models(force=force).items():
        typer.echo(f"{name}: {path}")


@models_app.command("path")
def models_path(
    verbose: Annotated[bool, typer.Option(help="Enable debug logging")] = False,
):
    """Show where each model is resolved from, and where it would be looked for."""
    from sokil.settings import (
        ModelResolutionError,
        model_names,
        model_spec,
        resolve_model_path,
        search_paths,
    )

    configure_logging(verbose)

    missing = False
    for name in model_names():
        spec = model_spec(name)
        try:
            typer.echo(f"{spec}: {resolve_model_path(name)}")
        except ModelResolutionError:
            missing = True
            looked_in = ", ".join(str(path) for path in search_paths(name))
            typer.echo(f"{spec}: not found (looked in: {looked_in})")

    if missing:
        raise typer.Exit(code=1)


def _format_expects(expects: dict, indent: str) -> list[str]:
    """Render the expects block back out in the manifest's own style."""
    lines = []
    for key, value in expects.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            rendered = "[" + ", ".join(str(item) for item in value) + "]"
        else:
            rendered = str(value)
        lines.append(f"{indent}{key}: {rendered}")
    return lines


@models_app.command("pin")
def models_pin(
    tag: Annotated[
        str, typer.Option(help="The release tag the assets are published under")
    ],
    weights: Annotated[
        list[str] | None,
        typer.Option(
            help="model=path of the weights to hash, repeatable, e.g. "
            "--weights shuttle=runs/detect/train8/weights/best.pt"
        ),
    ] = None,
    from_release: Annotated[
        bool,
        typer.Option(
            help="Read the checksums from the published release instead of "
            "hashing local files"
        ),
    ] = False,
    version: Annotated[
        str | None,
        typer.Option(
            help="Version label (defaults to the tag without a leading 'models-')"
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Enable debug logging")] = False,
):
    """
    Print the manifest entries for a new set of weights.

    Paste the output into sokil/models.yaml — keeping its comments — and commit
    it in the same change that publishes the release. Hashing the local files is
    preferred over --from-release: it needs no network, and it pins the bytes
    you actually published rather than what a server reports about them.
    """
    from pathlib import Path

    from sokil.settings import (
        model_names,
        model_spec,
        release_asset_digests,
        sha256,
    )

    configure_logging(verbose)

    label = version or tag.removeprefix("models-")

    if from_release:
        if weights:
            raise typer.BadParameter("--weights and --from-release are exclusive")
        digests = release_asset_digests(tag)
    else:
        if not weights:
            raise typer.BadParameter(
                "Pass --weights model=path for every model, or --from-release"
            )
        digests = {}
        for pair in weights:
            name, separator, raw_path = pair.partition("=")
            if not separator:
                raise typer.BadParameter(f"Expected model=path, got {pair!r}")
            path = Path(raw_path)
            if not path.is_file():
                raise typer.BadParameter(f"No such file: {path}")
            digests[model_spec(name.strip()).asset] = sha256(path)

    typer.echo("models:")
    exit_code = 0
    for name in model_names():
        spec = model_spec(name)
        digest = digests.get(spec.asset)
        if digest is None:
            typer.echo(f"  # {name}: no checksum for {spec.asset}", err=True)
            exit_code = 1
            continue

        typer.echo(f"  {name}:")
        # quoted: an unquoted date is YAML for datetime.date, not a string
        typer.echo(f'    version: "{label}"')
        typer.echo(f"    release_tag: {tag}")
        typer.echo(f"    asset: {spec.asset}")
        typer.echo(f'    sha256: "{digest}"')
        if spec.expects:
            typer.echo("    expects:")
            for line in _format_expects(spec.expects, "      "):
                typer.echo(line)

    if exit_code:
        raise typer.Exit(code=exit_code)


@models_app.command("clear-cache")
def models_clear_cache(
    verbose: Annotated[bool, typer.Option(help="Enable debug logging")] = False,
):
    """Delete the downloaded weights cache."""
    from sokil.settings import clear_cache

    configure_logging(verbose)
    typer.echo(f"Cleared {clear_cache()}")


@calibrate_app.command("frames")
def calibrate_frames(
    video_path: Annotated[
        str, typer.Option(help="Video of the checkerboard held at various angles")
    ],
    output_path: Annotated[
        str, typer.Option(help="Directory to write the selected frames into")
    ] = "calibration_images",
    interactive: Annotated[
        bool,
        typer.Option(
            help="Step through the video and pick frames by hand instead of "
            "automatically. Needs a display"
        ),
    ] = False,
    count: Annotated[
        int, typer.Option(help="How many distinct views to keep (automatic only)")
    ] = 20,
    min_difference: Annotated[
        float,
        typer.Option(
            help="How far apart two views must be to both be kept, as a "
            "fraction of the image diagonal (automatic only). Lower it if too "
            "few views are found"
        ),
    ] = 0.06,
    stride: Annotated[
        int, typer.Option(help="Examine every Nth frame (automatic only)")
    ] = 5,
    board_cols: Annotated[
        int, typer.Option(help="Inner corners across the board's width")
    ] = 8,
    board_rows: Annotated[
        int, typer.Option(help="Inner corners down the board's height")
    ] = 5,
    verbose: Annotated[bool, typer.Option(help="Enable debug logging")] = False,
):
    """
    Collect checkerboard frames from a video, ready to calibrate from.

    Automatic by default: a frame is kept only when the board's pose differs
    enough from every frame already kept, which is what makes the set
    constrain the lens model rather than merely being large.
    """
    from sokil.calibration import (
        CalibrationError,
        CheckerBoard,
        select_frames,
        select_frames_interactively,
    )

    configure_logging(verbose)

    board = CheckerBoard(cols=board_cols, rows=board_rows)

    try:
        if interactive:
            written = select_frames_interactively(video_path, output_path, board)
        else:
            written = select_frames(
                video_path,
                output_path,
                board,
                target=count,
                min_difference=min_difference,
                stride=stride,
            )
    except CalibrationError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"{len(written)} frames in {output_path}")
    typer.echo(f"Next: sokil calibrate compute --images-path {output_path}")


@calibrate_app.command("compute")
def calibrate_compute(
    images_path: Annotated[
        str, typer.Option(help="Directory of checkerboard images")
    ] = "calibration_images",
    output_path: Annotated[
        str, typer.Option(help="Directory to write the calibration into")
    ] = "artifacts",
    board_cols: Annotated[
        int, typer.Option(help="Inner corners across the board's width")
    ] = 8,
    board_rows: Annotated[
        int, typer.Option(help="Inner corners down the board's height")
    ] = 5,
    square: Annotated[
        float, typer.Option(help="Side of one board square, in cm")
    ] = 2.5,
    minimum_views: Annotated[
        int, typer.Option(help="Refuse to solve from fewer usable views than this")
    ] = 10,
    preview: Annotated[
        bool,
        typer.Option(help="Show each detection as it is found. Needs a display"),
    ] = False,
    write_preview: Annotated[
        bool,
        typer.Option(help="Also write a picture of the distortion this lens applies"),
    ] = False,
    verbose: Annotated[bool, typer.Option(help="Enable debug logging")] = False,
):
    """
    Solve the lens model from checkerboard images.

    Writes camera_K.npy and camera_dist.npy, which `sokil review` takes via
    --camera-intrinsics-path and --camera-dist-path.
    """
    from pathlib import Path

    import cv2

    from sokil.calibration import (
        CalibrationError,
        CheckerBoard,
        calibrate,
        distortion_preview,
    )

    configure_logging(verbose)

    board = CheckerBoard(cols=board_cols, rows=board_rows, square=square)

    try:
        result = calibrate(
            images_path,
            board,
            minimum_views=minimum_views,
            preview=preview,
        )
    except CalibrationError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    intrinsics_path, distortion_path = result.save(output_path)

    if write_preview:
        preview_path = Path(output_path) / "distortion_preview.jpg"
        cv2.imwrite(str(preview_path), distortion_preview(result))
        typer.echo(f"Distortion preview: {preview_path}")

    typer.echo(
        f"Calibrated from {len(result.used)} views at "
        f"{result.image_size[0]}x{result.image_size[1]}: "
        f"reprojection error {result.rms:.4f} px ({result.quality})"
    )
    typer.echo(f"  {intrinsics_path}")
    typer.echo(f"  {distortion_path}")
    typer.echo(
        "\nUse them with:\n"
        f"  sokil review --video-path CLIP --output-path REVIEW \\\n"
        f"    --camera-intrinsics-path {intrinsics_path} \\\n"
        f"    --camera-dist-path {distortion_path}"
    )


@app.command()
def create_cvat_task(
    video_path: Annotated[str, typer.Option(help="The path to the video to annotate")],
    model: Annotated[
        str, typer.Option(help="Which model the task is for: shuttle or court")
    ],
    every_n_frames: Annotated[
        int, typer.Option(help="Extract every Nth frame from the video. If 0 ")
    ] = 30,
    camera_intrinsics_path: Annotated[
        str | None,
        typer.Option(
            help="The path to the camera intrinsics matrix to undistort frames"
        ),
    ] = None,
    camera_dist_path: Annotated[
        str | None,
        typer.Option(
            help="The path to the camera distortion matrix to undistort frames"
        ),
    ] = None,
    pre_annotate_model_path: Annotated[
        str | None,
        typer.Option(
            help="An existing YOLO model used to pre-annotate the extracted frames"
        ),
    ] = None,
    project_name: Annotated[
        str | None,
        typer.Option(
            help="CVAT project to create the task in (defaults per model; created if missing)"
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Enable debug logging")] = False,
):
    """Extract frames from a video and create a CVAT annotation task."""
    from pathlib import Path

    from sokil.training import ModelKind, create_cvat_task

    configure_logging(verbose)

    task_id = create_cvat_task(
        video_path=Path(video_path),
        kind=ModelKind(model),
        every_n_frames=every_n_frames,
        camera_intrinsics_path=camera_intrinsics_path,
        camera_dist_path=camera_dist_path,
        pre_annotate_model_path=pre_annotate_model_path,
        project_name=project_name,
    )
    typer.echo(f"Created CVAT task {task_id}")


@app.command()
def export_cvat_dataset(
    model: Annotated[
        str, typer.Option(help="Which model the dataset is for: shuttle or court")
    ],
    task_id: Annotated[
        list[int] | None,
        typer.Option(
            help="CVAT task id(s) to export (repeatable). "
            "If omitted, all tasks with completed jobs in the project are exported"
        ),
    ] = None,
    project_name: Annotated[
        str | None,
        typer.Option(
            help="CVAT project to pull completed tasks from (defaults per model)"
        ),
    ] = None,
    grayscale: Annotated[
        bool | None,
        typer.Option(
            help="Convert images to grayscale (defaults to on for court, off for shuttle)"
        ),
    ] = None,
    override: Annotated[
        bool,
        typer.Option(
            help="Whether to override the existing dataset for the mode. Important: all data will be lost"
        ),
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed for the dataset split")] = 1,
    min_frame_gap: Annotated[
        int,
        typer.Option(
            help="Keep frames at least N source frames apart — thins tasks "
            "annotated on consecutive frames, leaves sparser tasks untouched"
        ),
    ] = 1,
    valid_size: Annotated[
        float,
        typer.Option(
            help="Validation split size: below 1 is a fraction of the export, "
            "1 or above is an absolute frame count (e.g. 800)"
        ),
    ] = 0.2,
    test_size: Annotated[
        float,
        typer.Option(help="Test split size, read the same way as --valid-size"),
    ] = 0.2,
    verbose: Annotated[bool, typer.Option(help="Enable debug logging")] = False,
):
    """Export CVAT tasks and prepare a YOLO dataset under training/<model>/datasets.

    With no --task-id, every task in the project whose jobs are all marked
    completed in CVAT is exported.
    """
    from sokil.training import ModelKind, export_dataset

    configure_logging(verbose)

    data_yaml = export_dataset(
        kind=ModelKind(model),
        task_ids=task_id,
        project_name=project_name,
        grayscale=grayscale,
        seed=seed,
        override=override,
        min_frame_gap=min_frame_gap,
        valid_size=valid_size,
        test_size=test_size,
    )
    typer.echo(f"Dataset ready: {data_yaml}")


@app.command()
def train_model(
    model: Annotated[str, typer.Option(help="Which model to train: shuttle or court")],
    base_weights: Annotated[
        str | None,
        typer.Option(
            help="Base weights to fine-tune (defaults to a per-model yolo11n)"
        ),
    ] = None,
    epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 100,
    imgsz: Annotated[
        int | None,
        typer.Option(
            help="Training image size. Defaults to the per-model size in training.yaml"
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            help="Torch device to train on: cpu, mps, cuda, 0, 0,1 ... "
            "(defaults to the trainer's pick: CUDA if available, else CPU)"
        ),
    ] = None,
    batch: Annotated[
        int | None,
        typer.Option(
            help="Batch size. Defaults to the per-model size in training.yaml — "
            "segmentation needs a much smaller batch than detection"
        ),
    ] = None,
    workers: Annotated[int, typer.Option(help="Dataloader workers")] = 8,
    cache: Annotated[
        str,
        typer.Option(
            help="Cache images to skip decoding every epoch: ram, disk or none"
        ),
    ] = "none",
    patience: Annotated[
        int, typer.Option(help="Stop early after N epochs without improvement")
    ] = 100,
    resume: Annotated[
        bool,
        typer.Option(
            help="Continue the newest interrupted run (or --base-weights if it "
            "points at a checkpoint) instead of starting fresh. The checkpoint's "
            "own epochs and dataset are reused, so don't resume onto a rebuilt "
            "dataset"
        ),
    ] = False,
    deterministic: Annotated[
        bool | None,
        typer.Option(
            help="Force deterministic ops. Defaults on, except on --device mps, "
            "where several backward ops have no deterministic kernel and the "
            "flag only warns once per step without making the run reproducible"
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Enable debug logging")] = False,
):
    """Train a YOLO model on the prepared dataset in training/<model>/datasets."""
    from sokil.training import ModelKind, train_model

    # keep ultralytics' own output: it is the only progress signal during training
    configure_logging(verbose, keep=("ultralytics",))

    weights = train_model(
        kind=ModelKind(model),
        base_weights=base_weights,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=workers,
        cache=False if cache == "none" else cache,
        patience=patience,
        resume=resume,
        deterministic=deterministic,
    )
    typer.echo(f"Trained model saved to {weights}")


if __name__ == "__main__":
    app()
