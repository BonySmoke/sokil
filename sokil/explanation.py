"""
Explain each stage of the review
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .court import Court
from .frame import ShuttleTrack
from .visualization import (
    draw_caption,
    stage_court_edges,
    stage_detection,
    stage_hit_point,
    stage_homography,
    stage_line_clusters,
    stage_observable_area,
    stage_raw_frame,
    stage_top_view,
)

logger = logging.getLogger(__name__)


@dataclass
class Stage:
    """One stage of the walkthrough: a frame-sized image and what it shows."""

    key: str
    title: str
    caption: str
    image: np.ndarray  # BGR


# The walkthrough, in order: what each stage is called and what it says. How
# each is drawn is decided in build_stages, which skips any stage it has no
# material for — the court lines cannot be shown without a segmentation.
STAGES = [
    (
        "raw",
        "The Clip",
        "A rally as the camera sees it.",
    ),
    (
        "detection",
        "Detection",
        (
            "A segmentation model outlines the court and a detection model finds "
            "the shuttle on every frame"
        ),
    ),
    (
        "edges",
        "Isolating Court Lines",
        "Inside the court mask, find painted court lines",
    ),
    (
        "clusters",
        "Line Clusters",
        "Collapse line edges into clusters.",
    ),
    (
        "homography",
        "Court Homography",
        (
            "The clustered lines are matched to the known court dimensions. "
            "The full court model is projected back onto the frame."
        ),
    ),
    (
        "observable",
        "Observable area",
        (
            "The area of the image used for analysis. "
            "Shuttle frames outside of this area is not adjudicated."
        ),
    ),
    (
        "hit",
        "Shuttle-to-Court Contact Point",
        "The contact point of the shuttle with the court",
    ),
    (
        "top_view",
        "The call",
        "The contact point mapped through the homography onto the real court. ",
    ),
]


def wipe(
    before: np.ndarray,
    after: np.ndarray,
    progress: float,
    divider_width: int = 4,
    divider_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    The slider transition: `after` revealed left to right over `before`.

    :param progress: 0 shows `before` untouched, 1 shows `after` untouched.
    """
    width = before.shape[1]
    split = round(width * float(np.clip(progress, 0.0, 1.0)))

    out = before.copy()
    if split > 0:
        out[:, :split] = after[:, :split]

    if 0 < split < width and divider_width > 0:
        left = max(0, split - divider_width // 2)
        out[:, left : left + divider_width] = divider_color

    return out


def build_stages(
    frame: np.ndarray,
    track: ShuttleTrack,
    session_court: Court,
    frame_court: Court | None = None,
    consensus_verticals=None,
    consensus_horizontals=None,
    observable_padding: float = 50,
) -> list[Stage]:
    """
    Draw every stage of the walkthrough on the contact frame.

    :param frame: the frame every stage is drawn on — the contact frame.
    :param track: the hit measured on that frame.
    :param session_court: the solved court, which supplies the homography.
    :param frame_court: a Court segmenting `frame` itself, for the stages that
        need this frame's own mask and lines. Without it those are skipped.
    :param consensus_verticals: the solver's surviving lines, if it ran. Without
        them the clusters stage shows this frame's own clusters alone.
    :returns: the stages that could be drawn, in order, captioned.
    """
    height, width = frame.shape[:2]

    drawn = {
        "raw": lambda: stage_raw_frame(frame),
        "detection": lambda: stage_detection(frame, frame_court, track),
        "edges": lambda: (
            stage_court_edges(frame_court) if frame_court is not None else None
        ),
        "clusters": lambda: (
            stage_line_clusters(
                frame, frame_court, consensus_verticals, consensus_horizontals
            )
            if frame_court is not None
            else None
        ),
        "homography": lambda: stage_homography(frame, session_court),
        "observable": lambda: stage_observable_area(
            frame, session_court, track, padding=observable_padding
        ),
        "hit": lambda: stage_hit_point(frame, session_court, track),
        "top_view": lambda: stage_top_view(session_court, track, width, height),
    }

    stages = []
    for key, title, caption in STAGES:
        image = drawn[key]()
        if image is None:
            logger.warning("Stage %r could not be drawn, skipping it", key)
            continue
        stages.append(Stage(key, title, caption, draw_caption(image, title, caption)))

    return stages


class Explainer:
    """
    Turns one landing into the frames that explain it.

    Held by the review renderer, which asks it for frames when it reaches a hit
    and writes them into the review video in place of the zoom and close-up
    beats. It carries what the walkthrough needs but a single frame does not
    have: the detector to segment the contact frame with, and the consensus
    lines the session court was solved from.
    """

    def __init__(
        self,
        court_detector_kwargs: dict,
        config,
        consensus_verticals=None,
        consensus_horizontals=None,
    ):
        self.court_detector_kwargs = court_detector_kwargs
        self.config = config
        self.consensus_verticals = consensus_verticals
        self.consensus_horizontals = consensus_horizontals

    def frames(
        self,
        frame: np.ndarray,
        track: ShuttleTrack,
        session_court: Court,
        fps: float,
    ) -> Iterator[np.ndarray]:
        """
        The walkthrough of one landing, frame by frame.

        :param frame: the undrawn contact frame. The stages draw their own
            overlays, so the review's are not wanted here.
        :param fps: the rate the frames will be written at, which is what turns
            the configured seconds into a number of frames.
        """
        # Each landing is segmented on its own contact frame rather than reusing
        # the session court's: that court's mask and lines belong to the frame
        # the homography was solved on, so borrowing them would cut to a
        # different picture halfway through the walkthrough.
        frame_court = Court(**self.court_detector_kwargs, image=frame)

        stages = build_stages(
            frame=frame,
            track=track,
            session_court=session_court,
            frame_court=frame_court,
            consensus_verticals=self.consensus_verticals,
            consensus_horizontals=self.consensus_horizontals,
            observable_padding=self.config.explain_observable_padding,
        )
        logger.debug(
            "Explaining frame %d: %s", track.number, ", ".join(s.key for s in stages)
        )

        transition_frames = round(fps * self.config.explain_transition_seconds)

        for index, stage in enumerate(stages):
            if index and transition_frames:
                previous = stages[index - 1].image
                for step in range(transition_frames):
                    yield wipe(previous, stage.image, (step + 1) / transition_frames)

            # The rally has just played into this frame, so the first stage is
            # already familiar and is held only long enough to register, not for
            # a full stage beat like the ones the viewer has not seen.
            seconds = (
                self.config.explain_stage_seconds
                if index
                else self.config.explain_freeze_seconds
            )
            for _ in range(round(fps * seconds)):
                yield stage.image
