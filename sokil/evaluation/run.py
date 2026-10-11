"""
Running the pipeline over a clip and scoring what comes out.

The stages are composed here rather than through Review, because what the
evaluation varies is precisely which stage gets ground truth instead of a
model.
Naming the substitution in one place keeps that visible instead of hidden
behind a flag.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..court import CourtCameraMode, CourtConfig, CourtModelType, CourtSolver
from ..dataset.boxes import ShuttleBoxSource
from ..dataset.court import (
    court_coordinates,
    court_from_points,
    reprojection_residuals,
)
from ..dataset.landing import CM_PER_M
from ..hits import HitDetectionConfig, HitDetector, HitJudge
from ..models import ShuttleDetector
from ..tracker import ShuttleTracker, TrackerConfig
from ..video import VideoSource
from .metrics import (
    MATCH_WINDOW_FRAMES,
    detection_counts,
    match_landings,
    position_errors,
    verdict_counts,
)

logger = logging.getLogger(__name__)

# What the configurations are called, and what each substitutes.
CONFIGURATIONS = {
    "algorithm": "ground-truth boxes, ground-truth court",
    "detector": "model boxes, ground-truth court",
    "system": "model boxes, solved court",
}


@dataclass
class ClipResult:
    """One clip, one configuration."""

    clip_id: str
    configuration: str
    detection: dict
    position_error: dict
    verdict_correctness: dict
    court_error: dict = field(default_factory=dict)
    frames_tracked: int = 0
    # False only where the court was solved from the video and the solve
    # failed, which makes every figure below it meaningless rather than bad.
    court_solved: bool = True


def _landings(tracks) -> dict[int, object]:
    """Frame -> the hit detected on it."""
    return {track.number: track for track in tracks if track.hit is not None}


def ground_truth_court(clip_id: str, clip: pd.Series, court_points: pd.DataFrame):
    """The court the annotator placed, for the configurations that are given it."""
    return court_from_points(
        court_points,
        clip_id,
        court_config=_court_config(clip),
    )


def _court_config(clip: pd.Series) -> CourtConfig:
    return CourtConfig(
        court_camera_mode=CourtCameraMode(clip["camera_mode"]),
        court_model_type=CourtModelType(clip["court_type"]),
    )


def run_clip(
    clip_id: str,
    video_path,
    clip: pd.Series,
    events: pd.DataFrame,
    court_points: pd.DataFrame,
    configuration: str,
    detector,
    court=None,
    court_solver=None,
    grayscale: bool = True,
    hit_config: HitDetectionConfig | None = None,
    tracker_config: TrackerConfig | None = None,
    window: int = MATCH_WINDOW_FRAMES,
) -> ClipResult:
    """
    Track, detect and judge one clip, then score it.

    :param detector: where the shuttle's box comes from — the trained model, or
        a ShuttleBoxSource reading the annotations.
    :param court: the court to adjudicate against, when it is given rather than
        solved.
    :param court_solver: used only when no court is given, which is the one
        configuration that measures the solver too.

    The stages are composed here rather than through Review because what the
    configurations vary is precisely which of them gets ground truth, and a
    substitution hidden behind a flag is a substitution nobody checks.
    """
    source = VideoSource(str(video_path), grayscale=grayscale)
    shuttle_tracks = ShuttleTracker(detector=detector, config=tracker_config).track(
        source
    )

    if court is None:
        if court_solver is None:
            raise ValueError("Give either a court or a solver to find one with")
        court = court_solver.solve(source)

    if court is None:
        logger.warning("%s: no court was solved; nothing can be judged", clip_id)

    HitDetector(hit_config or HitDetectionConfig()).detect(
        shuttle_tracks,
        court,
        float(clip["fps"]),
        frame_width=int(clip["width"]),
    )
    HitJudge().judge(shuttle_tracks, court)

    return _score(
        clip_id,
        configuration,
        shuttle_tracks,
        events,
        court_points,
        window,
        court,
        frames_tracked=len(shuttle_tracks),
    )


def run_algorithm(
    clip_id: str,
    video_path,
    clip: pd.Series,
    events: pd.DataFrame,
    court_points: pd.DataFrame,
    tracks: pd.DataFrame,
    **kwargs,
) -> ClipResult:
    """
    The algorithm alone: the annotator's boxes and the annotator's court.

    Whatever the hit detection gets wrong here is its own, which is what makes
    this the figure to tune against — it does not move when a model is
    retrained.
    """
    return run_clip(
        clip_id,
        video_path,
        clip,
        events,
        court_points,
        configuration="algorithm",
        detector=ShuttleBoxSource(tracks, clip_id),
        court=ground_truth_court(clip_id, clip, court_points),
        **kwargs,
    )


def run_detector(
    clip_id: str,
    video_path,
    clip: pd.Series,
    events: pd.DataFrame,
    court_points: pd.DataFrame,
    shuttle_detector_kwargs: dict,
    **kwargs,
) -> ClipResult:
    """
    The algorithm and the shuttle detector, on the annotator's court.

    The difference from the algorithm configuration is the detector and nothing
    else, so whatever moves between them is the detector's doing.
    """
    return run_clip(
        clip_id,
        video_path,
        clip,
        events,
        court_points,
        configuration="detector",
        detector=ShuttleDetector(
            shuttle_detector_kwargs["model"],
            confidence=(kwargs.get("tracker_config") or TrackerConfig()).confidence,
            device=(kwargs.get("tracker_config") or TrackerConfig()).device,
            imgsz=shuttle_detector_kwargs.get("imgsz", 640),
        ),
        court=ground_truth_court(clip_id, clip, court_points),
        **kwargs,
    )


def run_system(
    clip_id: str,
    video_path,
    clip: pd.Series,
    events: pd.DataFrame,
    court_points: pd.DataFrame,
    shuttle_detector_kwargs: dict,
    court_detector_kwargs: dict,
    court_solver_config=None,
    **kwargs,
) -> ClipResult:
    """
    The system as it ships: both models, and the court solved from the video.

    This is the figure a user would get, and the only one that moves when
    either model changes. On clips the models trained on it is optimistic,
    which belongs in the datasheet rather than in a caveat nobody reads.
    """
    return run_clip(
        clip_id,
        video_path,
        clip,
        events,
        court_points,
        configuration="system",
        detector=ShuttleDetector(
            shuttle_detector_kwargs["model"],
            confidence=(kwargs.get("tracker_config") or TrackerConfig()).confidence,
            device=(kwargs.get("tracker_config") or TrackerConfig()).device,
            imgsz=shuttle_detector_kwargs.get("imgsz", 640),
        ),
        court_solver=CourtSolver(
            {**court_detector_kwargs, "court_config": _court_config(clip)},
            court_solver_config,
        ),
        **kwargs,
    )


def _score(
    clip_id: str,
    configuration: str,
    shuttle_tracks,
    events: pd.DataFrame,
    court_points: pd.DataFrame,
    window: int,
    court,
    frames_tracked: int,
) -> ClipResult:
    """
    Compare what the pipeline produced against what was labelled.

    :param court: the court this run used, and so the one its landings are read
        back through. Using the annotated court here instead would quietly
        excuse a solve that was wrong — or, where the solve failed outright,
        report distances for a court the run never had.
    """
    labelled = events[events["clip_id"] == clip_id].set_index("frame")
    found = _landings(shuttle_tracks)

    matches, missed, false_positives = match_landings(
        sorted(found), sorted(labelled.index), window
    )
    counts = detection_counts(sorted(found), sorted(labelled.index), window)

    # Read back through the court this configuration ran with, so the figure
    # is the claim the configuration actually makes.
    to_court = _court_reader(court)

    positions, verdicts = [], []
    for match in matches:
        track = found[match.detected_frame]
        truth = labelled.loc[match.labelled_frame]

        if to_court is not None and track.hit.position is not None:
            placed = to_court(*track.hit.position)
            if placed is not None and not pd.isna(truth["x_m"]):
                positions.append((placed, (truth["x_m"], truth["y_m"])))

        if track.hit.is_in is not None:
            verdicts.append(("in" if track.hit.is_in else "out", truth["call"]))

    return ClipResult(
        clip_id=clip_id,
        configuration=configuration,
        detection={
            "labelled": counts.labelled,
            "detected": counts.detected,
            "matched": counts.matched,
            "missed": counts.missed,
            "false_positives": counts.false_positives,
            "precision": None
            if counts.precision is None
            else round(counts.precision, 4),
            "recall": None if counts.recall is None else round(counts.recall, 4),
            "f1": None if counts.f1 is None else round(counts.f1, 4),
            "missed_frames": missed,
            "false_positive_frames": false_positives,
            "frame_error": [m.frame_error for m in matches],
        },
        position_error=position_errors(positions),
        verdict_correctness=verdict_counts(verdicts),
        court_error=_court_error(court_points, clip_id, court),
        frames_tracked=frames_tracked,
        court_solved=court is not None,
    )


def _court_reader(court):
    """
    A function putting an image pixel into court metres, through one court.

    None where there is no court, so a run whose solve failed reports no
    landing positions rather than positions measured against a court it never
    had.
    """
    if court is None:
        return None

    transformer = court.get_view_transformer()
    if transformer is None:
        return None

    def to_court(x_px: float, y_px: float) -> tuple[float, float] | None:
        placed = transformer.transform_points(np.array([[x_px, y_px]]))
        if placed is None or not len(placed):
            return None
        return float(placed[0][0]) / CM_PER_M, float(placed[0][1]) / CM_PER_M

    return to_court


def _court_error(court_points: pd.DataFrame, clip_id: str, court) -> dict:
    """
    Two different things about the court, kept apart.

    `annotation_px` is how well the annotated keypoints fit the court model —
    the same in every configuration, because it is a property of the
    annotation. It bounds how well any landing can be placed, whoever placed
    the court.

    `vertex_cm` is how far this run's court places the annotated vertices from
    where they truly sit on a court, in centimetres. It is not zero even when
    the court *is* the annotated one: it then measures how precisely the
    vertices were clicked, which is the floor under every landing position.
    Where the court was solved it carries the solver's error on top of that
    floor, so the two configurations read together say how much the solver
    added.
    """
    clip_points = court_points[
        (court_points["clip_id"] == clip_id) & court_points["visible"]
    ]
    if clip_points.empty:
        return {}

    vertices = list(clip_points["vertex"])
    pixels = clip_points[["x_px", "y_px"]].to_numpy()

    error = {}
    residuals = reprojection_residuals(vertices, pixels)
    if residuals is not None:
        error["vertices"] = len(residuals)
        error["annotation_px"] = {
            "median": round(float(np.median(residuals)), 2),
            "max": round(float(residuals.max()), 2),
        }

    to_court = _court_reader(court)
    if to_court is not None:
        truth = court_coordinates(vertices) / CM_PER_M
        placed = [to_court(x_px, y_px) for x_px, y_px in pixels]

        if all(point is not None for point in placed):
            placed = np.array(placed)
            direct = np.linalg.norm(placed - truth, axis=1) * CM_PER_M

            # A badminton court is almost symmetric end to end, so a solver
            # matching lines by mask area cannot tell which baseline is which,
            # and may place a court of the right shape the wrong way round.
            # In/out survives that because the boundary is symmetric; a
            # reported landing coordinate does not. Named as a flip rather
            # than left as a distance, because a court's own length is not an
            # error anybody can act on.
            mirrored_truth = truth.copy()
            mirrored_truth[:, 1] = _court_length_m() - mirrored_truth[:, 1]
            mirrored = np.linalg.norm(placed - mirrored_truth, axis=1) * CM_PER_M

            flipped = bool(np.median(mirrored) < np.median(direct))
            distances = mirrored if flipped else direct

            error["vertex_cm"] = {
                "median": round(float(np.median(distances)), 1),
                "max": round(float(distances.max()), 1),
            }
            error["flipped_end_for_end"] = flipped

    return error


def _court_length_m() -> float:
    """The court's length in metres, the axis a flip mirrors about."""
    from ..court import CourtModel

    return CourtModel().length / CM_PER_M


def summarise(results: list[ClipResult]) -> dict:
    """
    One set of figures over every clip.

    Counts are pooled rather than averaged over clips: a clip with 39 landings
    and one with 3 are not equal evidence, and averaging their rates would
    pretend they were.
    """
    matched = sum(result.detection["matched"] for result in results)
    missed = sum(result.detection["missed"] for result in results)
    false_positives = sum(result.detection["false_positives"] for result in results)

    labelled = matched + missed
    detected = matched + false_positives
    precision = matched / detected if detected else None
    recall = matched / labelled if labelled else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None

    verdicts = sum(result.verdict_correctness["count"] for result in results)
    right = sum(
        round(
            (result.verdict_correctness["accuracy"] or 0)
            * result.verdict_correctness["count"]
        )
        for result in results
    )

    return {
        "clips": len(results),
        "detection": {
            "labelled": labelled,
            "detected": detected,
            "matched": matched,
            "missed": missed,
            "false_positives": false_positives,
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "f1": None if f1 is None else round(f1, 4),
        },
        "verdict_correctness": {
            "count": verdicts,
            "accuracy": None if not verdicts else round(right / verdicts, 4),
            "called_in_was_out": sum(
                result.verdict_correctness["called_in_was_out"] for result in results
            ),
            "called_out_was_in": sum(
                result.verdict_correctness["called_out_was_in"] for result in results
            ),
        },
        "position_error": {
            "count": sum(result.position_error["count"] for result in results),
            "median_cm": _pooled_median(results),
        },
    }


def _pooled_median(results: list[ClipResult]) -> float | None:
    """
    The median of the per-clip medians, since the errors themselves are not kept.

    An approximation, and labelled as one: a true pooled median would need
    every error retained, which the record does not carry.
    """
    medians = [
        result.position_error["median_cm"]
        for result in results
        if result.position_error["median_cm"] is not None
    ]
    return None if not medians else round(float(np.median(medians)), 1)
