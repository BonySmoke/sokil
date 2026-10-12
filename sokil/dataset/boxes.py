"""
Answering the detector's question from the annotations instead of a model.

ShuttleBoxSource stands in for ShuttleDetector in the tracking pass: asked for
the shuttle on a frame, it looks the frame up in tracks.csv rather than
predicting on the pixels. Everything the tracker derives afterwards — the
Kalman-filtered speed, the smoothed direction, the angle change, the cork
searched for inside the box, the blur measured through it — is read off the
decoded frame exactly as in production. Only the four numbers come from a human.

That is what makes a hit-detection figure stand on its own: measured this way it
moves when a threshold moves and not when a model is retrained, so tuning can be
credited to the thing being tuned.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# What an annotated box is reported as. A person drawing a box around the
# shuttle is not expressing a probability, and the tracker does not read this
# field — but the detector's answer carries one, so this one does too.
GROUND_TRUTH_CONFIDENCE = 1.0


class ShuttleBoxSource:
    """
    The annotated shuttle box per frame of one clip.

    A frame with no row returns nothing, which is what a detector finding
    nothing returns: between rallies the shuttle is out of play or out of
    frame, and the tracker meets that gap either way.
    """

    def __init__(self, tracks: pd.DataFrame, clip_id: str):
        clip_tracks = tracks[tracks["clip_id"] == clip_id]
        if clip_tracks.empty:
            raise ValueError(f"No ground-truth track for clip {clip_id!r}")

        self.clip_id = clip_id
        self.boxes: dict[int, tuple[float, float, float, float]] = {
            int(row.frame): (
                float(row.x1),
                float(row.y1),
                float(row.x2),
                float(row.y2),
            )
            for row in clip_tracks.itertuples()
        }
        logger.info("Ground truth for %s on %d frames", clip_id, len(self.boxes))

    def best_box(
        self,
        image: np.ndarray | None = None,
        confidence: float | None = None,
        frame: int | None = None,
    ) -> tuple[tuple[float, float, float, float], float] | None:
        """
        The annotated box on that frame, as (xyxy, confidence), or None.

        The image is accepted and ignored: the whole point is that the pixels
        are not consulted. Without a frame there is nothing to look up, which
        is a caller that has not passed the metadata rather than a frame with
        no shuttle on it, so it is an error rather than a None.
        """
        if frame is None:
            raise ValueError(
                "ShuttleBoxSource needs the frame number; it looks a box up "
                "rather than predicting one"
            )

        box = self.boxes.get(frame)
        return None if box is None else (box, GROUND_TRUTH_CONFIDENCE)

    def __len__(self) -> int:
        return len(self.boxes)
