"""
Reading a clip's frames by decode order.

Every frame in the fixture is a flat colour carrying its own index, so a test
can say which frame came back. That is the whole point of the class: frame N
has to be frame N whether it was reached by playing forward, jumping back, or
asking for it twice.
"""

import cv2
import numpy as np
import pytest

from sokil.dataset.frames import VideoFrames

COUNT = 40
SIZE = (64, 48)


@pytest.fixture
def clip(tmp_path):
    """A clip whose Nth frame is filled with the value N."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, SIZE)
    for index in range(COUNT):
        writer.write(np.full((SIZE[1], SIZE[0], 3), index * 5, np.uint8))
    writer.release()
    return path


def index_of(frame) -> int:
    """
    Which frame this is, read back off its own pixels.

    The green channel only: chroma subsampling in the container moves blue and
    red by a few counts, which is enough to read the wrong index off a median
    taken over all three.
    """
    return round(float(np.median(frame[:, :, 1])) / 5)


class TestReading:
    def test_reports_the_clip_s_properties(self, clip):
        with VideoFrames(clip) as frames:
            assert len(frames) == COUNT
            assert (frames.width, frames.height) == SIZE

    def test_reads_frames_in_order(self, clip):
        with VideoFrames(clip) as frames:
            assert [index_of(frames[i]) for i in range(5)] == [0, 1, 2, 3, 4]

    def test_a_jump_forward_lands_on_the_frame_asked_for(self, clip):
        with VideoFrames(clip) as frames:
            assert index_of(frames[25]) == 25

    def test_a_jump_back_lands_on_the_frame_asked_for(self, clip):
        with VideoFrames(clip) as frames:
            frames[30]
            assert index_of(frames[4]) == 4

    def test_the_same_frame_twice_is_the_same_frame(self, clip):
        with VideoFrames(clip) as frames:
            first = index_of(frames[17])
            frames[35]
            assert index_of(frames[17]) == first

    def test_scrubbing_back_and_forth_stays_honest(self, clip):
        # The pattern a labeller actually produces: hunting around a landing.
        with VideoFrames(clip) as frames:
            for wanted in (20, 21, 20, 19, 22, 3, 39, 0, 20):
                assert index_of(frames[wanted]) == wanted

    def test_past_the_end_reads_as_nothing(self, clip):
        with VideoFrames(clip) as frames:
            assert frames[COUNT + 10] is None

    def test_before_the_start_is_a_mistake(self, clip):
        with VideoFrames(clip) as frames, pytest.raises(IndexError):
            frames[-1]

    def test_a_missing_clip_says_so(self, tmp_path):
        with pytest.raises(ValueError, match="Could not open video"):
            VideoFrames(tmp_path / "nothing.mp4")


class TestCache:
    def test_a_small_cache_still_returns_the_right_frames(self, clip):
        # Forced to rewind and re-decode on every backward jump, which must
        # change how long it takes and nothing else.
        with VideoFrames(clip, cache_mb=0.001) as frames:
            assert index_of(frames[30]) == 30
            assert index_of(frames[2]) == 2
            assert index_of(frames[30]) == 30

    def test_the_cache_does_not_grow_past_its_budget(self, clip):
        with VideoFrames(clip, cache_mb=0.02) as frames:
            for index in range(COUNT):
                frames[index]

            assert len(frames._cache) <= frames.cache_size < COUNT

    def test_a_cached_frame_is_the_frame_that_was_decoded(self, clip):
        # Bit-exact, not merely similar: a frame read twice has to be the same
        # frame, and the same one the pipeline decodes.
        with VideoFrames(clip) as frames:
            first = frames[12].copy()
            frames[30]

            assert np.array_equal(frames[12], first)
