import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

from sokil.court import CourtCameraMode, CourtConfig, CourtModelType

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}

app = typer.Typer()

models_app = typer.Typer(help="Manage the trained weights")
app.add_typer(models_app, name="models")

calibrate_app = typer.Typer(help="Measure the camera lens, once per camera")
app.add_typer(calibrate_app, name="calibrate")

dataset_app = typer.Typer(help="Work with the annotation dataset")
app.add_typer(dataset_app, name="dataset")


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

    # The weights declare the frames they were trained on, so the pipeline
    # honours that instead of assuming. A pair that disagrees can only be fed
    # one way, so the shuttle detector wins: it is the one that runs per frame.
    shuttle_expects = model_spec(ModelName.SHUTTLE).expects
    court_expects = model_spec(ModelName.COURT).expects

    court_segmentation_model = Segmenter(
        court_segmentation_model_path, imgsz=court_expects.get("imgsz", 640)
    )

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
            "imgsz": shuttle_expects.get("imgsz", 640),
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


@models_app.command("export-onnx")
def models_export_onnx(
    model: Annotated[str, typer.Option(help="The model the weights belong to")],
    weights: Annotated[
        str,
        typer.Option(help="The .pt weights to export, e.g. models/shuttle-detect.pt"),
    ],
    verbose: Annotated[bool, typer.Option(help="Enable debug logging")] = False,
):
    """
    Export PyTorch weights to ONNX.

    The image size comes from the model's expects block in the manifest
    """
    from pathlib import Path

    from ultralytics import YOLO

    from sokil.settings import ModelResolutionError, model_spec

    configure_logging(verbose, keep=("ultralytics",))

    path = Path(weights)
    if not path.is_file():
        raise typer.BadParameter(f"No such file: {path}")

    try:
        spec = model_spec(model)
    except ModelResolutionError as error:
        raise typer.BadParameter(str(error))

    imgsz = spec.expects.get("imgsz")
    if imgsz is None:
        raise typer.BadParameter(f"The manifest gives no expects.imgsz for {spec}")

    # simplify needs onnxslim, which ultralytics would pip-install on the spot
    exported = YOLO(str(path)).export(format="onnx", imgsz=imgsz, simplify=False)
    typer.echo(exported)


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
        str,
        typer.Option(
            help="Which model the task is for: shuttle, court or court_keypoints"
        ),
    ],
    every_n_frames: Annotated[
        int, typer.Option(help="Extract every Nth frame from the video. If 0 ")
    ] = 30,
    median_frame: Annotated[
        bool,
        typer.Option(
            help="court_keypoints only: annotate the temporal median of the clip "
            "with the players averaged away leaving just the court"
        ),
    ] = True,
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
    """
    Extract frames from a video and create a CVAT annotation task.
    """
    from pathlib import Path

    from sokil.training import ModelKind, create_cvat_task

    configure_logging(verbose)

    task_id = create_cvat_task(
        video_path=Path(video_path),
        kind=ModelKind(model),
        every_n_frames=every_n_frames,
        median_frame=median_frame,
        camera_intrinsics_path=camera_intrinsics_path,
        camera_dist_path=camera_dist_path,
        pre_annotate_model_path=pre_annotate_model_path,
        project_name=project_name,
    )
    typer.echo(f"Created CVAT task {task_id}")


@dataset_app.command("init-clips")
def dataset_init_clips(
    videos_dir: Annotated[
        str, typer.Option(help="Folder holding the source recordings")
    ] = "videos",
    path: Annotated[
        str, typer.Option(help="Directory holding the dataset tables")
    ] = "dataset",
    clip_id: Annotated[
        list[str] | None,
        typer.Option(
            help="Only these clips (repeatable); default is every clip already annotated"
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging")] = False,
):
    """
    Fill clips.csv from the recordings.
    """
    import pandas as pd

    from sokil.dataset import Dataset
    from sokil.dataset.clips import blank_columns, clip_rows
    from sokil.dataset.schema import columns

    configure_logging(verbose)

    dataset = Dataset(Path(path))

    def read(table):
        return dataset.read(table) if dataset.path(table).exists() else None

    events = read("events")
    court_points = read("court_points")
    existing = read("clips")

    wanted = set(clip_id or [])
    if not wanted:
        for table in (events, court_points, existing):
            if table is not None:
                wanted |= set(table["clip_id"])

    if not wanted:
        raise typer.BadParameter(
            "Nothing is annotated yet and no --clip-id was given, so there is "
            "no clip to describe"
        )

    videos = {}
    for clip in sorted(wanted):
        matches = sorted(Path(videos_dir).glob(f"{clip}.*"))

        matches = [m for m in matches if m.suffix.lower() in VIDEO_EXTENSIONS]
        if not matches:
            typer.echo(f"No recording for {clip!r} in {videos_dir}", err=True)
            continue

        videos[clip] = matches[0]

    if not videos:
        raise typer.Exit(code=1)

    rows = clip_rows(videos, events, court_points, existing)
    written = dataset.write("clips", pd.DataFrame(rows, columns=columns("clips")))
    typer.echo(f"Described {len(rows)} clip(s) -> {written}")

    missing = blank_columns(rows)
    if missing:
        typer.echo("\nStill to fill in by hand:", err=True)
        for clip, empty in sorted(missing.items()):
            typer.echo(f"  {clip}: {', '.join(empty)}", err=True)


@dataset_app.command("label-hits")
def dataset_label_hits(
    video_path: Annotated[str, typer.Option(help="The clip to label")],
    path: Annotated[
        str, typer.Option(help="Directory holding the dataset tables")
    ] = "dataset",
    clip_id: Annotated[
        str | None,
        typer.Option(
            help="The clip being labelled; taken from the video filename if omitted"
        ),
    ] = None,
    annotator: Annotated[
        str | None, typer.Option(help="Who is making the calls; defaults to $USER")
    ] = None,
    court_type: Annotated[
        str | None,
        typer.Option(help="singles or doubles; taken from clips.csv if omitted"),
    ] = None,
    play_fps: Annotated[float, typer.Option(help="Playback rate while playing")] = 30.0,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging")] = False,
):
    """
    Label where the shuttle landed, and whether it was in/out.
    The position in metres is derived from court keypoints.
    """
    import pandas as pd

    from sokil.dataset import Dataset
    from sokil.dataset.cvat_import import default_clip_id
    from sokil.dataset.labeler import (
        event_rows,
        label_clip,
        load_session,
        merge_events,
    )
    from sokil.dataset.landing import image_to_court
    from sokil.dataset.schema import columns

    configure_logging(verbose)

    video = Path(video_path)
    if not video.is_file():
        raise typer.BadParameter(f"No clip at {video}")

    clip = clip_id or default_clip_id(video.name)

    dataset = Dataset(Path(path))
    events = (
        dataset.read("events")
        if dataset.path("events").exists()
        else pd.DataFrame(columns=columns("events"))
    )
    session = load_session(events, clip, annotator)

    court_points = None
    if dataset.path("court_points").exists():
        court_points = dataset.read("court_points")
    else:
        typer.echo(
            "No court_points.csv: landings will have no position in metres",
            err=True,
        )

    if court_type:
        session.court_type = court_type
    elif dataset.path("clips").exists():
        clips = dataset.read("clips")
        known = clips[clips["clip_id"] == clip]
        if not known.empty:
            session.court_type = known.iloc[0]["court_type"]
        else:
            typer.echo(f"{clip!r} is not in clips.csv", err=True)

    def save(current):
        to_court = (
            image_to_court(court_points, clip, 0) if court_points is not None else None
        )
        merged = merge_events(events, clip, event_rows(current, to_court))
        written = dataset.write("events", merged)
        typer.echo(f"Saved {len(current.landings)} landing(s) -> {written}")

    label_clip(video, session, court_points, play_fps=play_fps, on_save=save)


@dataset_app.command("tracks")
def dataset_tracks(
    path: Annotated[
        str, typer.Option(help="Directory holding the dataset tables")
    ] = "dataset",
    task_id: Annotated[
        list[int] | None,
        typer.Option(
            help="CVAT task id(s) to read (repeatable); default is every completed task"
        ),
    ] = None,
    project_name: Annotated[
        str, typer.Option(help="CVAT project the shuttle annotations live in")
    ] = "Badminton Shuttle Detection",
    clip_id: Annotated[
        list[str] | None,
        typer.Option(help="Map a task to a clip id as TASK_ID=CLIP_ID (repeatable)"),
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging")] = False,
):
    """
    Rebuild tracks.csv: the ground-truth shuttle box on every annotated frame.
    """
    from sokil.dataset import Dataset
    from sokil.dataset.track_import import import_tracks, merge_tracks

    configure_logging(verbose)

    overrides = {}
    for mapping in clip_id or []:
        task, _, clip = mapping.partition("=")
        if not clip:
            raise typer.BadParameter(f"{mapping!r} is not TASK_ID=CLIP_ID")
        overrides[int(task)] = clip

    dataset = Dataset(Path(path))
    known = (
        set(dataset.read("clips")["clip_id"])
        if dataset.path("clips").exists()
        else None
    )

    rows = import_tracks(
        task_ids=task_id or None,
        project_name=project_name,
        clip_ids=overrides,
        known_clips=known,
    )

    if known is not None:
        skipped = {row["clip_id"] for row in rows} - known
        if skipped:
            typer.echo(
                f"Skipping {len(skipped)} clip(s) not in clips.csv: "
                f"{', '.join(sorted(skipped))}",
                err=True,
            )
        rows = [row for row in rows if row["clip_id"] in known]

    if not rows:
        typer.echo("No shuttle annotations found", err=True)
        raise typer.Exit(code=1)

    existing = dataset.read("tracks") if dataset.path("tracks").exists() else None
    merged = merge_tracks(existing, rows)

    written = dataset.write("tracks", merged)
    typer.echo(
        f"Wrote {len(rows)} row(s) for {len({r['clip_id'] for r in rows})} clip(s); "
        f"{len(merged)} rows for {merged['clip_id'].nunique()} clip(s) total "
        f"-> {written}"
    )


@dataset_app.command("court-points")
def dataset_court_points(
    path: Annotated[
        str, typer.Option(help="Directory holding the dataset tables")
    ] = "dataset",
    task_id: Annotated[
        list[int] | None,
        typer.Option(
            help="CVAT task id(s) to read (repeatable). "
            "If omitted, every task in the project is read"
        ),
    ] = None,
    project_name: Annotated[
        str, typer.Option(help="CVAT project the keypoint tasks live in")
    ] = "Badminton Court Keypoints",
    clip_id: Annotated[
        list[str] | None,
        typer.Option(
            help="Map a task to a clip id as TASK_ID=CLIP_ID (repeatable). "
            "Needed when the recording was renamed after it was uploaded"
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging")] = False,
):
    """
    Rebuild court_points.csv from the CVAT keypoint annotations.
    """
    from sokil.dataset import Dataset
    from sokil.dataset.cvat_import import import_court_points, merge_court_points

    configure_logging(verbose)

    overrides = {}
    for mapping in clip_id or []:
        task, _, clip = mapping.partition("=")
        if not clip:
            raise typer.BadParameter(f"{mapping!r} is not TASK_ID=CLIP_ID")
        overrides[int(task)] = clip

    dataset = Dataset(Path(path))
    known = (
        set(dataset.read("clips")["clip_id"])
        if dataset.path("clips").exists()
        else None
    )

    rows = import_court_points(
        task_ids=task_id or None,
        project_name=project_name,
        clip_ids=overrides,
        known_clips=known,
    )
    if not rows:
        typer.echo("No court keypoints found", err=True)
        raise typer.Exit(code=1)

    # clips.csv is the dataset's manifest, so a task annotating something that
    # is not in it is not part of the dataset — an old task, or one belonging
    # to another project's work. Before the manifest exists there is nothing to
    # filter against and everything is taken.
    if known is not None:
        skipped = {row["clip_id"] for row in rows} - known
        if skipped:
            typer.echo(
                f"Skipping {len(skipped)} clip(s) not in clips.csv: "
                f"{', '.join(sorted(skipped))}",
                err=True,
            )
        rows = [row for row in rows if row["clip_id"] in known]

        if not rows:
            typer.echo("Nothing left to import", err=True)
            raise typer.Exit(code=1)

    existing = (
        dataset.read("court_points") if dataset.path("court_points").exists() else None
    )
    merged = merge_court_points(existing, rows)

    written = dataset.write("court_points", merged)
    typer.echo(
        f"Wrote {len(rows)} row(s) for {len({r['clip_id'] for r in rows})} clip(s); "
        f"{len(merged)} rows for {merged['clip_id'].nunique()} clip(s) total "
        f"-> {written}"
    )


@dataset_app.command("split")
def dataset_split(
    path: Annotated[
        str, typer.Option(help="Directory holding the dataset tables")
    ] = "dataset",
    project_name: Annotated[
        str,
        typer.Option(
            help="CVAT project whose task subsets state each clip's partition"
        ),
    ] = "Badminton Court Keypoints",
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging")] = False,
):
    """
    Rebuild dataset/splits/ from the CVAT task subsets for hit detection.
    Model YOLO training datasets are generated by export_cvat_dataset.
    """
    from sokil.dataset import Dataset
    from sokil.dataset.cvat_import import read_subsets
    from sokil.dataset.splits import SplitError, mode_coverage, split_from_subsets

    configure_logging(verbose)

    dataset = Dataset(Path(path))
    clips = dataset.read("clips")
    known = set(clips["clip_id"])

    subsets = read_subsets(project_name, known_clips=known)
    missing = sorted(known - set(subsets))
    if missing:
        typer.echo(
            f"{len(missing)} clip(s) in clips.csv have no task in "
            f"{project_name!r}: {', '.join(missing)}",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        splits = split_from_subsets({c: subsets[c] for c in sorted(known)})
    except SplitError as error:
        typer.echo(str(error), err=True)
        typer.echo(
            "\nSet each task's Subset in CVAT (Train / Validation / Test), "
            "then run this again.",
            err=True,
        )
        raise typer.Exit(code=1) from None

    dataset.write_splits(splits)
    for name, clip_ids in splits.items():
        typer.echo(f"{name}: {len(clip_ids)} clip(s)")
        for clip_id in clip_ids:
            typer.echo(f"  {clip_id}")

    camera_modes = dict(zip(clips["clip_id"], clips["camera_mode"]))
    for warning in mode_coverage(splits, camera_modes):
        typer.echo(f"\nwarning: {warning}", err=True)


@dataset_app.command("validate")
def dataset_validate(
    path: Annotated[str, typer.Option(help="Directory holding dataset")] = "dataset",
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging")] = False,
):
    """
    Check the annotation dataset inconsistencies.
    """
    from sokil.dataset import Dataset, DatasetError

    configure_logging(verbose)

    try:
        problems = Dataset(Path(path)).validate()
    except DatasetError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None

    for problem in problems:
        typer.echo(problem, err=True)

    if problems:
        typer.echo(f"\n{len(problems)} problem(s) found", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"{path}: no problems found")


@app.command()
def evaluate(
    path: Annotated[
        str, typer.Option(help="Directory holding the dataset tables")
    ] = "dataset",
    videos_dir: Annotated[
        str, typer.Option(help="Folder holding the source recordings")
    ] = "videos",
    configuration: Annotated[
        list[str] | None,
        typer.Option(
            help="Which configuration(s) to measure (repeatable): "
            "'algorithm' substitutes ground truth for both models, 'detector' "
            "only for the court, 'system' for neither. Default: algorithm"
        ),
    ] = None,
    split: Annotated[
        str,
        typer.Option(
            help="Which clips to measure: all, train, valid or test. Tuning "
            "belongs on train; a figure you intend to publish belongs on test"
        ),
    ] = "all",
    shuttle_detector_model_path: Annotated[
        str | None,
        typer.Option(help="Shuttle weights, for the detector and system runs"),
    ] = None,
    court_segmentation_model_path: Annotated[
        str | None,
        typer.Option(help="Court weights, for the system run only"),
    ] = None,
    results_dir: Annotated[
        str, typer.Option(help="Where the JSON record is written")
    ] = "evaluations",
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging")] = False,
):
    """
    Measure whether the hit detection is right, against the dataset.

    Three configurations over one ground truth, so a change can be credited to
    whatever caused it:

    \b
      algorithm  annotated boxes, annotated court — the logic alone
      detector   model boxes, annotated court     — plus the shuttle detector
      system     model boxes, solved court        — as it ships

    The algorithm configuration is the one to tune against: HitDetectionConfig
    carries seven hand-set thresholds meant to be swept, and a figure that also
    moves when a model is retrained cannot tell you whether a sweep helped.

    Every run writes a JSON record carrying the commit, the thresholds, a
    digest of each dataset table and the versions that could change a figure.
    How fast the pipeline runs is a separate measurement — benchmarking — and
    belongs in the same record, not under this name.
    """
    from sokil.dataset import Dataset
    from sokil.evaluation import report
    from sokil.evaluation.run import (
        CONFIGURATIONS,
        run_algorithm,
        run_detector,
        run_system,
        summarise,
    )
    from sokil.hits import HitDetectionConfig

    configure_logging(verbose)

    wanted_configurations = configuration or ["algorithm"]
    unknown = [name for name in wanted_configurations if name not in CONFIGURATIONS]
    if unknown:
        raise typer.BadParameter(
            f"{unknown} is not a configuration; expected {sorted(CONFIGURATIONS)}"
        )

    dataset = Dataset(Path(path))
    clips = dataset.read("clips").set_index("clip_id")
    events = dataset.read("events")
    court_points = dataset.read("court_points")
    tracks = dataset.read("tracks")

    wanted_clips = _clips_to_evaluate(dataset, events, tracks, split)
    hit_config = HitDetectionConfig()

    # The models declare the frames they were trained on, so that is what they
    # are fed — the same reasoning as the review command.
    needs_models = {"detector", "system"} & set(wanted_configurations)
    grayscale, shuttle_kwargs, court_kwargs = True, None, None
    if needs_models:
        grayscale, shuttle_kwargs, court_kwargs = _evaluation_models(
            shuttle_detector_model_path,
            court_segmentation_model_path,
            need_court="system" in wanted_configurations,
        )

    results = []
    for name in wanted_configurations:
        typer.echo(f"\n{name}: {CONFIGURATIONS[name]}")

        for clip_id in wanted_clips:
            video = Path(videos_dir) / f"{clip_id}.mp4"
            if not video.is_file():
                typer.echo(f"No recording for {clip_id!r} in {videos_dir}", err=True)
                continue

            common = {
                "clip": clips.loc[clip_id],
                "events": events,
                "court_points": court_points,
                "hit_config": hit_config,
                "grayscale": grayscale,
            }

            if name == "algorithm":
                result = run_algorithm(clip_id, video, tracks=tracks, **common)
            elif name == "detector":
                result = run_detector(
                    clip_id, video, shuttle_detector_kwargs=shuttle_kwargs, **common
                )
            else:
                result = run_system(
                    clip_id,
                    video,
                    shuttle_detector_kwargs=shuttle_kwargs,
                    court_detector_kwargs=court_kwargs,
                    **common,
                )

            results.append(result)

        _echo_evaluation_table(
            [r for r in results if r.configuration == name],
            summarise([r for r in results if r.configuration == name]),
        )

    if not results:
        raise typer.Exit(code=1)

    totals = {
        name: summarise([r for r in results if r.configuration == name])
        for name in wanted_configurations
    }
    written = report.write(
        report.record(results, hit_config, Path(path), wanted_clips, totals),
        Path(results_dir),
    )
    typer.echo(f"\nrecord: {written}")


def _clips_to_evaluate(dataset, events, tracks, split: str) -> list[str]:
    """The clips a configuration can be measured on, honouring the split."""
    wanted = sorted(set(events["clip_id"]) & set(tracks["clip_id"]))

    if split != "all":
        splits = dataset.read_splits()
        if split not in splits:
            raise typer.BadParameter(f"{split!r} is not a split")
        if not splits[split]:
            typer.echo(
                f"The {split} split is empty — run 'sokil dataset split' first",
                err=True,
            )
            raise typer.Exit(code=1)
        wanted = [clip for clip in wanted if clip in set(splits[split])]

    if not wanted:
        typer.echo("No clip has both landings and a ground-truth track", err=True)
        raise typer.Exit(code=1)

    return wanted


def _evaluation_models(
    shuttle_path: str | None, court_path: str | None, need_court: bool
):
    """
    The weights the model-backed configurations run, and what they expect.

    Taken from the published manifest rather than assumed: the weights declare
    the frames they were trained on, and feeding them anything else measures
    the mismatch instead of the model.
    """
    from sokil.models import Segmenter
    from sokil.settings import ModelName, ensure_model_path, model_spec

    shuttle_expects = model_spec(ModelName.SHUTTLE).expects
    shuttle_kwargs = {
        "model": shuttle_path or str(ensure_model_path(ModelName.SHUTTLE)),
        "imgsz": shuttle_expects.get("imgsz", 640),
    }

    court_kwargs = None
    if need_court:
        court_expects = model_spec(ModelName.COURT).expects
        court_kwargs = {
            "segmentation_model": Segmenter(
                court_path or str(ensure_model_path(ModelName.COURT)),
                imgsz=court_expects.get("imgsz", 640),
            ),
            "keypoint_model": None,
        }

    return shuttle_expects.get("grayscale", True), shuttle_kwargs, court_kwargs


def _echo_evaluation_table(results, totals):
    """The per-clip figures, and the pooled ones beneath them."""
    from sokil.evaluation.metrics import MATCH_WINDOW_FRAMES

    header = (
        f"{'clip':<26}{'landings':>9}{'matched':>9}{'missed':>8}{'spurious':>10}"
        f"{'precision':>11}{'recall':>9}{'position':>10}{'verdict':>9}"
        f"{'court':>9}"
    )
    typer.echo("\n" + header)
    typer.echo("-" * len(header))

    for result in results:
        detection = result.detection
        typer.echo(
            f"{_short_clip(result.clip_id):<26}"
            f"{detection['labelled']:>9}{detection['matched']:>9}"
            f"{detection['missed']:>8}{detection['spurious']:>10}"
            f"{_percentage(detection['precision']):>11}"
            f"{_percentage(detection['recall']):>9}"
            f"{_centimetres(result.position['median_cm']):>10}"
            f"{_percentage(result.verdict['accuracy']):>9}"
            f"{_court_offset(result):>9}"
        )

    detection = totals["detection"]
    typer.echo("-" * len(header))
    typer.echo(
        f"{'pooled':<26}"
        f"{detection['labelled']:>9}{detection['matched']:>9}"
        f"{detection['missed']:>8}{detection['spurious']:>10}"
        f"{_percentage(detection['precision']):>11}"
        f"{_percentage(detection['recall']):>9}"
        f"{_centimetres(totals['position']['median_cm']):>10}"
        f"{_percentage(totals['verdict']['accuracy']):>9}"
        f"{'':>9}"
    )

    typer.echo(
        "\nlandings: labelled in the dataset.  matched: found within "
        f"{MATCH_WINDOW_FRAMES} frames of a labelled landing."
        "\nmissed: labelled but not found.  spurious: found but accounting "
        "for no labelled landing."
        "\nposition: median distance from the labelled landing, over the "
        "matched ones.  verdict: of those, how many calls were right."
        "\ncourt: how far the court this run used sits from the annotated "
        "one, which is nothing unless it was solved.  'failed': no court was "
        "solved, so nothing below it could be judged.  'FLIPPED': solved the "
        "right shape end-for-end, so verdicts survive but coordinates mirror."
    )

    verdict = totals["verdict"]
    if verdict["called_in_was_out"] or verdict["called_out_was_in"]:
        typer.echo(
            f"\nwrong calls: {verdict['called_in_was_out']} called in when out, "
            f"{verdict['called_out_was_in']} called out when in"
        )


def _court_offset(result) -> str:
    """
    How far this run's court sits from the annotated one.

    Blank where the court was given rather than solved: zero by construction
    says nothing, and printing it would invite reading it as a measurement.
    """
    if not result.court_solved:
        return "failed"

    if result.court.get("flipped_end_for_end"):
        return "FLIPPED"

    solved = result.court.get("solved_cm")
    if solved is None or solved["median"] < 1.0:
        return ""

    return f"{solved['median']:.0f}cm"


def _short_clip(clip_id: str, width: int = 25) -> str:
    """
    A clip id short enough for a table, keeping the part that identifies it.

    The tail is the cutting tool's doing — "-merged-cut-merged-1784750377220" —
    and is the same shape on every clip, so truncating from the right leaves
    every row looking alike.
    """
    import re

    trimmed = re.sub(r"(-merged)*-cut-merged-\d+$", "", clip_id)
    trimmed = trimmed.removeprefix("volan_leva_")
    return trimmed[:width]


def _percentage(value) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _centimetres(value) -> str:
    return "-" if value is None else f"{value:.1f}cm"


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
    seed_split: Annotated[
        bool,
        typer.Option(
            help="Shuffle clips by seed instead of using each CVAT task's Subset"
        ),
    ] = False,
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
        seed_split=seed_split,
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
