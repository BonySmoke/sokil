"""
What the evaluation counts, as plain functions over plain values.

Kept apart from running the pipeline so the arithmetic can be read and tested
without a video: a metric nobody can check by hand is a metric nobody should
quote.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# How far a detected landing may sit from a labelled one and still be the same
# event. At 120 fps this is about 17 ms: tight enough that a match is a real
# match, which is what makes a recall figure worth anything.
MATCH_WINDOW_FRAMES = 2

# How many strips the court's length is cut into when landings are reported by
# how far they fell from the camera. Perspective stretches a pixel with
# distance, so a single average over the whole court blends regimes that
# differ severalfold; more strips than this leave too few landings in each.
PROXIMITY_BINS = 4


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
    Landing position error in centimetres.

    :param pairs: (detected, labelled) positions in court metres.

    Reported over found landings only. A missed landing has no position to be
    wrong about, so it belongs in recall rather than hidden inside an average
    here — which is also why this figure must never be quoted without it.
    """
    if not pairs:
        return {"count": 0, "median_cm": None, "p90_cm": None}

    errors = [position_error_cm(detected, labelled) for detected, labelled in pairs]

    return {
        "count": len(errors),
        "median_cm": round(float(np.median(errors)), 1),
        "p90_cm": round(float(np.percentile(errors, 90)), 1),
    }


def position_error_cm(
    detected: tuple[float, float], labelled: tuple[float, float]
) -> float:
    """How far one landing was placed from where it was labelled, in cm."""
    return float(np.linalg.norm(np.subtract(detected, labelled))) * 100


def strip_of(y_m: float, length_m: float, bins: int = PROXIMITY_BINS) -> int:
    """
    Which strip across the court a landing at y_m falls in.

    A landing past either baseline belongs to the end strip it overshot: an
    out call lands just beyond the line and is as far from the camera as the
    in calls beside it.
    """
    strip = int(y_m / length_m * bins)
    return min(max(strip, 0), bins - 1)


def proximity_ranks(rows: list[float]) -> list[int]:
    """
    Each strip's rank by closeness to the camera, 0 being the nearest.

    :param rows: the image row each strip's centre projects to. On a flat floor
        seen from above, nearer ground sits lower in the frame, so the lowest
        strip is the nearest whatever end of the court the camera stands at.
    """
    order = sorted(range(len(rows)), key=lambda strip: rows[strip], reverse=True)
    ranks = [0] * len(rows)
    for rank, strip in enumerate(order):
        ranks[strip] = rank
    return ranks


def by_proximity(landings: list[dict], bins: int = PROXIMITY_BINS) -> list[dict]:
    """
    Recall and position error per strip, nearest to the camera first.

    :param landings: one per labelled landing, carrying its `proximity` rank
        (None where its clip's court could not be ranked), whether it was
        `matched`, and its `error_cm` where it was placed.

    Pooled from every landing rather than from per-clip figures: each strip
    holds a few landings per clip, so only the pool has enough to read.
    """
    report = []
    for rank in range(bins):
        here = [landing for landing in landings if landing["proximity"] == rank]
        matched = sum(1 for landing in here if landing["matched"])
        errors = [
            landing["error_cm"] for landing in here if landing["error_cm"] is not None
        ]
        report.append(
            {
                "proximity": rank,
                "labelled": len(here),
                "matched": matched,
                "recall": round(matched / len(here), 4) if here else None,
                "positions": len(errors),
                "median_cm": round(float(np.median(errors)), 1) if errors else None,
                "p90_cm": (
                    round(float(np.percentile(errors, 90)), 1) if errors else None
                ),
            }
        )
    return report


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
