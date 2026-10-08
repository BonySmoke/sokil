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

from ..court import CourtCameraMode, CourtConfig, CourtModelType
from ..dataset.boxes import ShuttleBoxSource
from ..dataset.court import court_from_points, reprojection_residuals
from ..dataset.landing import image_to_court
from ..hits import HitDetectionConfig, HitDetector, HitJudge
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
    position: dict
    verdict: dict
    court: dict = field(default_factory=dict)
    frames_tracked: int = 0


def _landings(tracks) -> dict[int, object]:
    """Frame -> the hit detected on it."""
    return {track.number: track for track in tracks if track.hit is not None}


def run_algorithm(
    clip_id: str,
    video_path,
    clip: pd.Series,
    events: pd.DataFrame,
    court_points: pd.DataFrame,
    tracks: pd.DataFrame,
    hit_config: HitDetectionConfig | None = None,
    tracker_config: TrackerConfig | None = None,
    window: int = MATCH_WINDOW_FRAMES,
) -> ClipResult:
    """
    Score the algorithm with both learned stages replaced by ground truth.

    The boxes are the annotator's and the court is the annotator's, so whatever
    the hit detection gets wrong here is its own. That is what makes this the
    figure to tune against: it does not move when a model is retrained.
    """
    court = court_from_points(
        court_points,
        clip_id,
        court_config=CourtConfig(
            court_camera_mode=CourtCameraMode(clip["camera_mode"]),
            court_model_type=CourtModelType(clip["court_type"]),
        ),
    )

    source = VideoSource(str(video_path), grayscale=bool(clip.get("grayscale", True)))
    shuttle_tracks = ShuttleTracker(
        detector=ShuttleBoxSource(tracks, clip_id), config=tracker_config
    ).track(source)

    detector = HitDetector(hit_config or HitDetectionConfig())
    detector.detect(
        shuttle_tracks,
        court,
        float(clip["fps"]),
        frame_width=int(clip["width"]),
    )
    HitJudge().judge(shuttle_tracks, court)

    return _score(
        clip_id,
        "algorithm",
        shuttle_tracks,
        events,
        court_points,
        window,
        frames_tracked=len(shuttle_tracks),
    )


def _score(
    clip_id: str,
    configuration: str,
    shuttle_tracks,
    events: pd.DataFrame,
    court_points: pd.DataFrame,
    window: int,
    frames_tracked: int,
) -> ClipResult:
    """Compare what the pipeline produced against what was labelled."""
    labelled = events[events["clip_id"] == clip_id].set_index("frame")
    found = _landings(shuttle_tracks)

    matches, missed, spurious = match_landings(
        sorted(found), sorted(labelled.index), window
    )
    counts = detection_counts(sorted(found), sorted(labelled.index), window)

    # The landing position is read back through the same court the labelling
    # used, so the two sides are comparing the same geometry.
    to_court = image_to_court(court_points, clip_id, 0)

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
            "spurious": counts.spurious,
            "precision": None
            if counts.precision is None
            else round(counts.precision, 4),
            "recall": None if counts.recall is None else round(counts.recall, 4),
            "f1": None if counts.f1 is None else round(counts.f1, 4),
            "missed_frames": missed,
            "spurious_frames": spurious,
            "frame_error": [m.frame_error for m in matches],
        },
        position=position_errors(positions),
        verdict=verdict_counts(verdicts),
        court=_court_error(court_points, clip_id),
        frames_tracked=frames_tracked,
    )


def _court_error(court_points: pd.DataFrame, clip_id: str) -> dict:
    """
    How well the annotated court fits itself, in pixels.

    Zero would mean the keypoints are perfectly consistent with the court
    model. It is reported for every configuration because it bounds how well
    any landing position can be placed, whoever placed the court.
    """
    clip_points = court_points[
        (court_points["clip_id"] == clip_id) & court_points["visible"]
    ]
    if clip_points.empty:
        return {}

    residuals = reprojection_residuals(
        list(clip_points["vertex"]), clip_points[["x_px", "y_px"]].to_numpy()
    )
    if residuals is None:
        return {}

    return {
        "vertices": len(residuals),
        "median_px": round(float(np.median(residuals)), 2),
        "max_px": round(float(residuals.max()), 2),
    }


def summarise(results: list[ClipResult]) -> dict:
    """
    One set of figures over every clip.

    Counts are pooled rather than averaged over clips: a clip with 39 landings
    and one with 3 are not equal evidence, and averaging their rates would
    pretend they were.
    """
    matched = sum(result.detection["matched"] for result in results)
    missed = sum(result.detection["missed"] for result in results)
    spurious = sum(result.detection["spurious"] for result in results)

    labelled = matched + missed
    detected = matched + spurious
    precision = matched / detected if detected else None
    recall = matched / labelled if labelled else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None

    verdicts = sum(result.verdict["count"] for result in results)
    right = sum(
        round((result.verdict["accuracy"] or 0) * result.verdict["count"])
        for result in results
    )

    return {
        "clips": len(results),
        "detection": {
            "labelled": labelled,
            "detected": detected,
            "matched": matched,
            "missed": missed,
            "spurious": spurious,
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "f1": None if f1 is None else round(f1, 4),
        },
        "verdict": {
            "count": verdicts,
            "accuracy": None if not verdicts else round(right / verdicts, 4),
            "called_in_was_out": sum(
                result.verdict["called_in_was_out"] for result in results
            ),
            "called_out_was_in": sum(
                result.verdict["called_out_was_in"] for result in results
            ),
        },
        "position": {
            "count": sum(result.position["count"] for result in results),
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
        result.position["median_cm"]
        for result in results
        if result.position["median_cm"] is not None
    ]
    return None if not medians else round(float(np.median(medians)), 1)
