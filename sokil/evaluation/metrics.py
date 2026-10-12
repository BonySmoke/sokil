"""
What the evaluation counts, as plain functions over plain values.

Kept apart from running the pipeline so the arithmetic can be read and tested
without a video: a metric nobody can check by hand is a metric nobody should
quote.
"""

import logging
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

logger = logging.getLogger(__name__)

# How far a detected landing may sit from a labelled one and still be the same
# event. At 120 fps this is about 17 ms: tight enough that a match is a real
# match, which is what makes a recall figure worth anything.
MATCH_WINDOW_FRAMES = 2

# The court's length is divided here when errors are reported by distance.
# Perspective stretches a pixel with distance, so a single average over the
# whole court blends two regimes that differ by more than tenfold.
DISTANCE_BANDS_M = (3.0, 6.0, 9.0)


@dataclass(frozen=True)
class MatchedLanding:
    """A detected landing paired with the labelled one it accounts for."""

    detected_frame: int
    labelled_frame: int

    @property
    def frame_error(self) -> int:
        return self.detected_frame - self.labelled_frame


@dataclass(frozen=True)
class DetectionCounts:
    """How many landings were found, missed and invented."""

    matched: int
    missed: int
    false_positives: int

    @property
    def labelled(self) -> int:
        return self.matched + self.missed

    @property
    def detected(self) -> int:
        return self.matched + self.false_positives

    @property
    def recall(self) -> float | None:
        return None if not self.labelled else self.matched / self.labelled

    @property
    def precision(self) -> float | None:
        return None if not self.detected else self.matched / self.detected

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if not precision or not recall:
            return None
        return 2 * precision * recall / (precision + recall)


def match_landings(
    detected: list[int],
    labelled: list[int],
    window: int = MATCH_WINDOW_FRAMES,
) -> tuple[list[MatchedLanding], list[int], list[int]]:
    """
    Pair detected landings with labelled ones, one for one.

    Closest pairs are taken first, and each landing is used once: two
    detections near one labelled landing means one match and one false
    positive, not two matches. Otherwise a detector that fires repeatedly
    around every landing would score perfectly.

    :returns: the matches, the labelled frames nothing found, and the detected
        frames that account for nothing.
    """
    candidates = sorted(
        (abs(detected_frame - labelled_frame), detected_frame, labelled_frame)
        for detected_frame in detected
        for labelled_frame in labelled
        if abs(detected_frame - labelled_frame) <= window
    )

    matches, used_detected, used_labelled = [], set(), set()
    for _, detected_frame, labelled_frame in candidates:
        if detected_frame in used_detected or labelled_frame in used_labelled:
            continue
        matches.append(
            MatchedLanding(detected_frame=detected_frame, labelled_frame=labelled_frame)
        )
        used_detected.add(detected_frame)
        used_labelled.add(labelled_frame)

    missed = sorted(set(labelled) - used_labelled)
    false_positives = sorted(set(detected) - used_detected)

    return sorted(matches, key=lambda m: m.labelled_frame), missed, false_positives


def detection_counts(
    detected: list[int],
    labelled: list[int],
    window: int = MATCH_WINDOW_FRAMES,
) -> DetectionCounts:
    """Precision, recall and F1 over one clip's landings."""
    matches, missed, false_positives = match_landings(detected, labelled, window)
    return DetectionCounts(
        matched=len(matches), missed=len(missed), false_positives=len(false_positives)
    )


def position_errors(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> dict:
    """
    Landing position error in centimetres, overall and by distance.

    :param pairs: (detected, labelled) positions in court metres.

    Reported over found landings only. A missed landing has no position to be
    wrong about, so it belongs in recall rather than hidden inside an average
    here — which is also why this figure must never be quoted without it.
    """
    if not pairs:
        return {"count": 0, "median_cm": None, "p90_cm": None, "by_distance": {}}

    detected = np.array([p[0] for p in pairs], dtype=float)
    labelled = np.array([p[1] for p in pairs], dtype=float)
    errors = np.linalg.norm(detected - labelled, axis=1) * 100

    bands = {}
    edges = (0.0, *DISTANCE_BANDS_M, float("inf"))
    for low, high in pairwise(edges):
        in_band = (labelled[:, 1] >= low) & (labelled[:, 1] < high)
        if not in_band.any():
            continue
        label = f"{low:.0f}-{high:.0f} m" if high != float("inf") else f"{low:.0f}+ m"
        bands[label] = {
            "count": int(in_band.sum()),
            "median_cm": round(float(np.median(errors[in_band])), 1),
            "max_cm": round(float(errors[in_band].max()), 1),
        }

    return {
        "count": len(errors),
        "median_cm": round(float(np.median(errors)), 1),
        "p90_cm": round(float(np.percentile(errors, 90)), 1),
        "by_distance": bands,
    }


def verdict_counts(pairs: list[tuple[str, str]]) -> dict:
    """
    Verdict accuracy, with the two mistakes kept apart.

    :param pairs: (called, labelled) verdicts, each "in" or "out".

    Calling a shuttle in when it was out hands the point to the wrong player
    and is the error an umpire would hear about, so averaging the two together
    would hide the one that matters.
    """
    counted = [
        (called, labelled)
        for called, labelled in pairs
        if called in ("in", "out") and labelled in ("in", "out")
    ]
    if not counted:
        return {
            "count": 0,
            "accuracy": None,
            "called_in_was_out": 0,
            "called_out_was_in": 0,
        }

    right = sum(1 for called, labelled in counted if called == labelled)
    return {
        "count": len(counted),
        "accuracy": round(right / len(counted), 4),
        "called_in_was_out": sum(
            1 for called, labelled in counted if called == "in" and labelled == "out"
        ),
        "called_out_was_in": sum(
            1 for called, labelled in counted if called == "out" and labelled == "in"
        ),
    }
