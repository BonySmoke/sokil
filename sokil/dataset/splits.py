"""
The evaluation split: which clips the hit detection may be tuned on.
"""

import logging

logger = logging.getLogger(__name__)

# TODO: check if this belongs in sokil/training
# What CVAT's subset field is called, and what this calls it. CVAT lets a
# subset be any string, so anything else is a typo or a scheme nobody has
# explained, and is refused rather than guessed at.
SUBSET_SPLITS = {"train": "train", "validation": "valid", "test": "test"}


class SplitError(RuntimeError):
    """The split cannot be derived from what CVAT currently says."""


def split_from_subsets(subsets: dict[str, str]) -> dict[str, list[str]]:
    """
    Clip ids per split, from each clip's CVAT subset.

    A clip with no subset is an unanswered question, not a default: assuming
    train would quietly train on a clip somebody meant to hold out, and the
    only cost of refusing is a trip to the CVAT UI.
    """
    splits: dict[str, list[str]] = {name: [] for name in SUBSET_SPLITS.values()}
    unassigned, unknown = [], {}

    for clip_id, subset in sorted(subsets.items()):
        key = (subset or "").strip().lower()

        if not key:
            unassigned.append(clip_id)
            continue

        split = SUBSET_SPLITS.get(key)
        if split is None:
            unknown[clip_id] = subset
            continue

        splits[split].append(clip_id)

    problems = []
    if unassigned:
        problems.append(
            f"{len(unassigned)} clip(s) have no subset in CVAT: "
            + ", ".join(unassigned)
        )
    if unknown:
        problems.append(
            "subset(s) CVAT holds that this does not recognise: "
            + ", ".join(f"{clip}={subset!r}" for clip, subset in unknown.items())
            + f" (expected one of {sorted(SUBSET_SPLITS)})"
        )
    if problems:
        raise SplitError("; ".join(problems))

    if not splits["train"]:
        raise SplitError(
            "no clip is in train, so there is nothing to tune the hit detection on"
        )

    return splits


def mode_coverage(
    splits: dict[str, list[str]], camera_modes: dict[str, str]
) -> list[str]:
    """
    Warnings about what the split covers, by camera mode.

    A mode that appears in test but not in train is being extrapolated to, and
    the figure that comes back measures the gap rather than the system. A mode
    that appears only in train is simply untested, which is a coverage gap
    worth stating out loud rather than discovering later.
    """
    by_split = {
        name: {camera_modes[clip] for clip in clips if clip in camera_modes}
        for name, clips in splits.items()
    }

    warnings = []

    for extrapolated in sorted(
        by_split.get("test", set()) - by_split.get("train", set())
    ):
        warnings.append(
            f"camera mode {extrapolated!r} is in test but not in train, so the "
            f"result measures generalising to an unseen angle rather than "
            f"performance; a mode with only one clip belongs in train"
        )

    untested = by_split.get("train", set()) - by_split.get("test", set())
    for gap in sorted(untested):
        warnings.append(f"camera mode {gap!r} is trained on but never tested")

    return warnings
