"""
VideoSource: a re-readable video that every pass decodes identically.

The clip is generated here rather than committed, so the tests carry no binary
fixture and still exercise real decoding.
"""

import cv2
import numpy as np
import pytest

from sokil.video import DecodedFrame, VideoSource

FRAME_COUNT = 12
WIDTH, HEIGHT = 64, 48
FPS = 10.0


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> str:
    """A short synthetic clip: each frame is a different flat colour."""
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    for number in range(FRAME_COUNT):
        frame = np.full((HEIGHT, WIDTH, 3), 10 + number * 15, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("this OpenCV build cannot write an mp4")

    return str(path)


class TestOpening:
    def test_reads_the_clips_properties(self, clip):
        source = VideoSource(clip)

        assert (source.width, source.height) == (WIDTH, HEIGHT)
        assert source.fps == pytest.approx(FPS)
        assert source.frame_count == FRAME_COUNT

    def test_a_missing_file_is_reported_when_it_is_opened(self, tmp_path):
        with pytest.raises(ValueError, match="Could not open video"):
            VideoSource(tmp_path / "nope.mp4")

    def test_describe_names_the_file_and_its_shape(self, clip):
        described = VideoSource(clip).describe()

        assert "clip.mp4" in described
        assert f"{WIDTH}x{HEIGHT}" in described


class TestIteration:
    def test_yields_every_frame_in_decode_order(self, clip):
        numbers = [decoded.number for decoded in VideoSource(clip)]

        assert numbers == list(range(FRAME_COUNT))

    def test_the_source_can_be_iterated_more_than_once(self, clip):
        source = VideoSource(clip)

        first = [decoded.number for decoded in source]
        second = [decoded.number for decoded in source]

        assert first == second

    def test_each_frame_carries_its_pixels_and_a_timestamp(self, clip):
        decoded = next(iter(VideoSource(clip)))

        assert isinstance(decoded, DecodedFrame)
        assert decoded.image.shape == (HEIGHT, WIDTH, 3)
        assert decoded.timestamp == pytest.approx(0.0)

    def test_timestamps_advance_with_the_frame_rate(self, clip):
        timestamps = [decoded.timestamp for decoded in VideoSource(clip)]

        assert timestamps == sorted(timestamps)
        assert timestamps[-1] == pytest.approx((FRAME_COUNT - 1) / FPS, abs=0.05)


class TestSampling:
    def test_a_stride_keeps_the_original_frame_numbering(self, clip):
        numbers = [decoded.number for decoded in VideoSource(clip).sample(4)]

        assert numbers == [0, 4, 8]

    def test_a_stride_of_one_is_every_frame(self, clip):
        assert len(list(VideoSource(clip).sample(1))) == FRAME_COUNT

    def test_a_stride_longer_than_the_clip_yields_the_first_frame_only(self, clip):
        numbers = [decoded.number for decoded in VideoSource(clip).sample(100)]

        assert numbers == [0]


class TestDecodeTransforms:
    def test_by_default_the_model_sees_the_same_pixels_as_the_viewer(self, clip):
        decoded = next(iter(VideoSource(clip)))

        assert decoded.model_input is decoded.image

    def test_in_grayscale_mode_the_model_input_is_desaturated(self, clip):
        decoded = next(iter(VideoSource(clip, grayscale=True)))

        # still three channels, so the detector's input shape does not change
        assert decoded.model_input.shape == decoded.image.shape
        channels = cv2.split(decoded.model_input)
        assert np.array_equal(channels[0], channels[1])
        assert np.array_equal(channels[1], channels[2])

    def test_the_displayed_image_is_left_in_colour(self, clip):
        source = VideoSource(clip, grayscale=True)

        decoded = next(iter(source))

        assert decoded.image.shape == (HEIGHT, WIDTH, 3)

    def test_the_undistorter_is_applied_to_every_frame(self, clip):
        # a stand-in for Undistorter: what matters here is that the source
        # routes every decoded frame through it before anything else sees it
        seen = []

        def undistorter(image):
            seen.append(image)
            return np.zeros_like(image)

        images = [decoded.image for decoded in VideoSource(clip, undistorter)]

        assert len(seen) == FRAME_COUNT
        assert all(not image.any() for image in images)

    def test_the_detector_sees_the_undistorted_frame_too(self, clip):
        marker = np.full((HEIGHT, WIDTH, 3), 7, dtype=np.uint8)
        source = VideoSource(clip, undistorter=lambda image: marker, grayscale=True)

        decoded = next(iter(source))

        assert np.array_equal(decoded.image, marker)
        assert decoded.model_input.max() == 7
