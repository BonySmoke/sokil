"""
Shared fixtures for the unit tests.

Nothing here touches a trained checkpoint or a real recording: the models are
replaced by fakes that return whatever a test wants to feed the pipeline, and
the only video is a handful of synthetic frames written to a temporary file.
"""

import matplotlib
import numpy as np
import pytest

from sokil.court import Court
from sokil.frame import ShuttleTrack

# Every plotting helper under test builds a figure. Agg renders one without a
# display, which neither CI nor a detached terminal has. Selected here rather
# than through MPLBACKEND so running a single test file by hand behaves the
# same as a full run.
matplotlib.use("Agg")

# The court model is measured in centimetres and the homography fixtures below
# are the identity, so an image pixel IS a centimetre in these tests. That keeps
# every expected value readable against CourtModel's own dimensions.
IDENTITY_HOMOGRAPHY = np.eye(3, dtype=np.float64)


@pytest.fixture
def identity_court() -> Court:
    """
    A court whose image->court homography is the identity.

    Point (x, y) in "image pixels" is therefore (x, y) cm on the court, so a
    test can say "600 cm down the court" and write 600.
    """
    return Court.from_homography(IDENTITY_HOMOGRAPHY)


@pytest.fixture
def make_track():
    """
    Build a ShuttleTrack with only the fields a test cares about.

    Every field has a plausible default so a test that is about, say, the cork
    position does not have to invent a timestamp and a blur value.
    """

    def build(
        number: int = 0,
        xyxy: tuple[int, int, int, int] = (100, 100, 110, 110),
        speed: float = 10.0,
        angle_change: float = 0.0,
        velocity: tuple[float, float] = (10.0, 10.0),
        timestamp: float | None = None,
        blur: float = 50.0,
        **kwargs,
    ) -> ShuttleTrack:
        return ShuttleTrack(
            number=number,
            xyxy=xyxy,
            speed=speed,
            angle_change=angle_change,
            velocity=velocity,
            timestamp=number / 30.0 if timestamp is None else timestamp,
            blur=blur,
            **kwargs,
        )

    return build


@pytest.fixture
def frame() -> np.ndarray:
    """A small BGR frame with some structure, so filters have something to bite."""
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :] = (30, 30, 30)
    image[40:44, :] = (255, 255, 255)  # a horizontal white "line"
    image[:, 60:64] = (255, 255, 255)  # a vertical one
    return image
