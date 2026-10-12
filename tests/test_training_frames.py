"""
Choosing the single frame a static clip's court is annotated on.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from sokil.training.frames import extract_court_frame

WIDTH, HEIGHT, FRAMES = 160, 120, 30


@pytest.fixture
def clip(tmp_path) -> Path:
    """An empty court with one moving object crossing it."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (WIDTH, HEIGHT)
    )
    court = np.full((HEIGHT, WIDTH, 3), 30, np.uint8)
    cv2.rectangle(court, (20, 20), (140, 100), (255, 255, 255), 1)

    for index in range(FRAMES):
        frame = court.copy()
        cv2.circle(frame, (10 + index * 5, 60), 8, (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def red_pixels(image) -> int:
    return int(((image[:, :, 2] > 150) & (image[:, :, 0] < 100)).sum())


class TestCourtFrame:
    def test_is_named_as_the_clip_s_first_frame(self, clip, tmp_path):
        path = extract_court_frame(clip, tmp_path / "out")

        assert path.name == "clip_000000.jpg"

    def test_the_median_leaves_an_empty_court(self, clip, tmp_path):
        # The point of the median: whoever was standing on the line is gone,
        # so the intersection under them can be clicked.
        path = extract_court_frame(clip, tmp_path / "out")

        assert red_pixels(cv2.imread(str(path))) == 0

    def test_the_first_frame_keeps_whoever_was_on_court(self, clip, tmp_path):
        path = extract_court_frame(clip, tmp_path / "out", median=False)

        assert red_pixels(cv2.imread(str(path))) > 0

    def test_the_court_itself_survives_the_median(self, clip, tmp_path):
        path = extract_court_frame(clip, tmp_path / "out")

        image = cv2.imread(str(path))
        assert int((image > 200).sum()) > 100

    def test_an_unreadable_video_says_so(self, tmp_path):
        (tmp_path / "broken.mp4").write_bytes(b"not a video")

        with pytest.raises(ValueError, match="Could not open video"):
            extract_court_frame(tmp_path / "broken.mp4", tmp_path / "out")
