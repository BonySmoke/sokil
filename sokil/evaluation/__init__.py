"""
Measuring whether the system is right, against the annotation dataset.

This is evaluation: correctness, as precision, recall, landing error and
verdict accuracy. How fast it runs and what it costs in memory is a separate
question with a separate word — benchmarking — and belongs beside these figures
in the same record rather than under the same name.

Three configurations over one ground truth, so that a change can be credited to
whatever caused it:

    ground-truth boxes, ground-truth court   the algorithm alone
    model boxes,        ground-truth court   the algorithm and the detector
    model boxes,        solved court         the system as it ships

The first is the one to tune against: it moves when a threshold moves and not
when a model is retrained.
"""

from .metrics import (
    DetectionCounts,
    MatchedLanding,
    detection_counts,
    match_landings,
    position_errors,
    verdict_counts,
)

__all__ = [
    "DetectionCounts",
    "MatchedLanding",
    "detection_counts",
    "match_landings",
    "position_errors",
    "verdict_counts",
]
