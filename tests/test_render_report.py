"""
Turning a finished review into output: the drawn trails, the hit close-up, the
per-frame CSV and the diagnostic plots.
"""

import csv

import cv2
import numpy as np
import pytest
from matplotlib import pyplot as plt

from sokil.frame import Hit
from sokil.render import (
    FFmpegWriter,
    RenderConfig,
    ReviewRenderer,
    figure_to_frame,
    hit_closeup,
    open_video_writer,
)
from sokil.report import Analytics, StatsExporter


@pytest.fixture
def tracks(make_track):
    return [
        make_track(number=i, xyxy=(i * 10, i * 5, i * 10 + 10, i * 5 + 10))
        for i in range(8)
    ]


class TestAssignTrackPoints:
    def test_every_track_of_a_trajectory_gets_the_same_colour(self, tracks):
        ReviewRenderer().assign_track_points(tracks, [[0, 1, 2, 3]])

        colours = {tracks[i].track_color for i in range(4)}

        assert len(colours) == 1
        assert colours != {None}

    def test_the_trail_grows_as_the_trajectory_plays(self, tracks):
        ReviewRenderer().assign_track_points(tracks, [[0, 1, 2, 3]])

        assert [len(tracks[i].track_points) for i in range(4)] == [1, 2, 3, 4]

    def test_each_track_keeps_its_own_copy_of_the_trail(self, tracks):
        ReviewRenderer().assign_track_points(tracks, [[0, 1, 2]])

        assert tracks[0].track_points is not tracks[1].track_points

    def test_the_trail_follows_the_box_centres(self, tracks):
        ReviewRenderer().assign_track_points(tracks, [[0, 1]])

        assert tracks[1].track_points == [tracks[0].bbox_center, tracks[1].bbox_center]

    def test_tracks_outside_any_trajectory_are_left_undrawn(self, tracks):
        ReviewRenderer().assign_track_points(tracks, [[0, 1]])

        assert tracks[5].track_points is None
        assert tracks[5].track_color is None

    def test_two_trajectories_get_their_own_trails(self, tracks):
        ReviewRenderer().assign_track_points(tracks, [[0, 1], [4, 5]])

        assert len(tracks[5].track_points) == 2
        assert tracks[5].track_points[0] == tracks[4].bbox_center

    def test_no_trajectories_at_all_changes_nothing(self, tracks):
        ReviewRenderer().assign_track_points(tracks, [])

        assert all(track.track_points is None for track in tracks)


class TestRenderConfig:
    def test_the_defaults_render_a_plain_review(self):
        config = RenderConfig()

        assert config.explain is False
        assert config.output_fps is None

    def test_the_renderer_keeps_the_config_it_was_given(self):
        config = RenderConfig(show_speed=True)

        assert ReviewRenderer(config).config is config

    def test_a_renderer_without_a_config_gets_the_defaults(self):
        assert ReviewRenderer().config == RenderConfig()


class TestFigureToFrame:
    def figure(self, width_inches=4.0, height_inches=1.0):
        fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=50)
        ax.plot([0, 1], [0, 1])
        return fig

    def test_the_result_is_exactly_the_frame_size(self):
        frame = figure_to_frame(self.figure(), width=320, height=240)

        assert frame.shape == (240, 320, 3)

    def test_a_figure_of_another_aspect_is_letterboxed_not_stretched(self):
        background = (34, 139, 34)

        # a 4:1 figure inside a 4:3 frame leaves bands above and below
        frame = figure_to_frame(
            self.figure(4.0, 1.0), width=320, height=240, background=background
        )

        assert tuple(frame[0, 160]) == background
        assert tuple(frame[239, 160]) == background

    def test_the_figure_itself_lands_in_the_middle(self):
        frame = figure_to_frame(self.figure(4.0, 1.0), width=320, height=240)

        assert tuple(frame[120, 160]) != (34, 139, 34)


class TestHitCloseup:
    def test_draws_the_landing_on_a_frame_sized_canvas(
        self, identity_court, make_track
    ):
        track = make_track(number=4, measured_cork=(300, 600), velocity=(10.0, 10.0))
        track.hit = Hit(position=(300, 600), is_in=True)

        closeup = hit_closeup(identity_court, track, width=320, height=240, dpi=50)

        assert closeup.shape == (240, 320, 3)

    def test_an_in_call_and_an_out_call_look_different(
        self, identity_court, make_track
    ):
        inside = make_track(number=4, measured_cork=(300, 600), velocity=(10.0, 10.0))
        inside.hit = Hit(position=(300, 600), is_in=True)
        outside = make_track(number=4, measured_cork=(300, 600), velocity=(10.0, 10.0))
        outside.hit = Hit(position=(300, 600), is_in=False)

        assert not np.array_equal(
            hit_closeup(identity_court, inside, 320, 240, dpi=50),
            hit_closeup(identity_court, outside, 320, 240, dpi=50),
        )

    def test_there_is_no_closeup_without_a_homography(self, make_track, frame):
        from sokil.court import Court

        class NoMask:
            def polygons(self, image, confidence=None):
                return []

        track = make_track(number=4, measured_cork=(300, 600), velocity=(10.0, 10.0))
        court = Court(image=frame, segmentation_model=NoMask())

        assert hit_closeup(court, track, width=320, height=240, dpi=50) is None


class TestStatsExporter:
    def test_one_row_per_track(self, tracks):
        rows = StatsExporter().export(tracks, video="clip.mp4")

        assert len(rows) == len(tracks)

    def test_a_row_describes_the_frame_it_came_from(self, make_track):
        track = make_track(number=7, speed=12.345, angle_change=-3.2109, timestamp=0.5)
        track.hit = Hit(position=(50, 60), is_in=False)

        (row,) = StatsExporter().export([track], video="clip.mp4")

        assert row["video"] == "clip.mp4"
        assert row["frame"] == 7
        assert row["speed"] == 12.345
        assert row["angle_change"] == -3.211
        assert row["is_hit"] is True
        assert row["is_inside_court"] is False
        assert row["timestamp"] == 0.5

    def test_a_frame_without_a_hit_has_no_verdict(self, make_track):
        (row,) = StatsExporter().export([make_track()], video="clip.mp4")

        assert row["is_hit"] is False
        assert row["is_inside_court"] is None

    def test_the_position_is_the_cork_not_the_box_centre(self, make_track):
        track = make_track(xyxy=(0, 0, 20, 20), velocity=(10.0, 0.0))

        (row,) = StatsExporter().export([track], video="clip.mp4")

        assert (row["cx"], row["cy"]) == (20.0, 10.0)

    def test_writing_a_csv_produces_a_header_and_the_rows(self, tracks, tmp_path):
        path = tmp_path / "stats.csv"

        StatsExporter().export(tracks, video="clip.mp4", output_path=str(path))

        written = list(csv.DictReader(path.open()))
        assert len(written) == len(tracks)
        assert written[0]["video"] == "clip.mp4"

    def test_appending_a_second_video_does_not_repeat_the_header(
        self, tracks, tmp_path
    ):
        path = tmp_path / "stats.csv"
        exporter = StatsExporter()

        exporter.export(tracks, video="a.mp4", output_path=str(path))
        exporter.export(tracks, video="b.mp4", output_path=str(path), append=True)

        written = list(csv.DictReader(path.open()))
        assert len(written) == 2 * len(tracks)
        assert {row["video"] for row in written} == {"a.mp4", "b.mp4"}

    def test_appending_to_a_new_file_still_writes_the_header(self, tracks, tmp_path):
        path = tmp_path / "stats.csv"

        StatsExporter().export(
            tracks, video="a.mp4", output_path=str(path), append=True
        )

        assert next(iter(csv.DictReader(path.open())))["video"] == "a.mp4"

    def test_a_second_export_without_append_overwrites(self, tracks, tmp_path):
        path = tmp_path / "stats.csv"
        exporter = StatsExporter()

        exporter.export(tracks, video="a.mp4", output_path=str(path))
        exporter.export(tracks[:2], video="b.mp4", output_path=str(path))

        assert len(list(csv.DictReader(path.open()))) == 2

    def test_nothing_to_export_writes_no_file(self, tmp_path):
        path = tmp_path / "stats.csv"

        assert StatsExporter().export([], video="clip.mp4", output_path=str(path)) == []
        assert not path.exists()


class TestAnalytics:
    def test_plots_a_panel_per_measurement(self, tracks):
        fig, axs = Analytics().plot(tracks)

        assert len(axs) == 5
        plt.close(fig)

    def test_the_hits_are_marked_on_the_speed_panel(self, tracks):
        plain_fig, plain_axs = Analytics().plot(tracks)
        marks_before = len(plain_axs[1].lines)
        plt.close(plain_fig)

        tracks[3].hit = Hit(position=(30, 15), is_in=True)
        tracks[6].hit = Hit(position=(60, 30), is_in=False)
        fig, axs = Analytics().plot(tracks)

        assert len(axs[1].lines) == marks_before + 2
        plt.close(fig)

    def test_a_run_with_no_hits_marks_nothing(self, tracks):
        fig, axs = Analytics().plot(tracks)

        # only the speed series itself
        assert len(axs[1].lines) == 1
        plt.close(fig)


class RenderSource:
    """A decodable video for the renderer, with no file behind it."""

    WIDTH, HEIGHT, FPS = 160, 120, 30.0

    def __init__(self, count: int):
        self.width = self.WIDTH
        self.height = self.HEIGHT
        self.fps = self.FPS
        self.frame_count = count
        self._count = count

    def __iter__(self):
        from sokil.video import DecodedFrame

        for number in range(self._count):
            image = np.full((self.HEIGHT, self.WIDTH, 3), 40, dtype=np.uint8)
            image[50:60, 20 + number : 30 + number] = 220
            yield DecodedFrame(
                number=number,
                image=image,
                model_input=image,
                timestamp=number / self.FPS,
            )


class CountingExplainer:
    """Stands in for the Explainer: yields a fixed number of frames per landing."""

    def __init__(self, count: int = 3):
        self.count = count
        self.calls = []

    def frames(self, image, track, court, fps):
        self.calls.append(track.number)
        for _ in range(self.count):
            yield np.zeros_like(image)


def frames_in(path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


class TestRender:
    def tracked(self, make_track, numbers):
        return [
            make_track(number=n, xyxy=(20 + n, 50, 30 + n, 60), velocity=(4.0, 0.0))
            for n in numbers
        ]

    def test_writes_a_playable_video(self, tmp_path, make_track):
        output = tmp_path / "review.mp4"

        ReviewRenderer().render(
            RenderSource(6), self.tracked(make_track, range(6)), None, str(output)
        )

        assert output.exists()
        assert frames_in(output) == 6

    def test_frames_with_no_detection_are_still_written(self, tmp_path, make_track):
        output = tmp_path / "review.mp4"

        ReviewRenderer().render(
            RenderSource(6), self.tracked(make_track, [1, 3]), None, str(output)
        )

        assert frames_in(output) == 6

    def test_a_landing_is_held_on_screen(self, tmp_path, make_track):
        tracks = self.tracked(make_track, range(6))
        tracks[2].hit = Hit(position=(24, 55), is_in=True)
        config = RenderConfig(hit_zoom_seconds=0.5, hit_closeup_seconds=0.0)
        output = tmp_path / "review.mp4"

        ReviewRenderer(config).render(RenderSource(6), tracks, None, str(output))

        # the six source frames plus half a second of magnified contact frame
        assert frames_in(output) == 6 + round(RenderSource.FPS * 0.5)

    def test_the_verdict_closeup_needs_a_court(
        self, tmp_path, make_track, identity_court
    ):
        tracks = self.tracked(make_track, range(6))
        tracks[2].hit = Hit(position=(24, 55), is_in=True)
        tracks[2].measured_cork = (24, 55)
        config = RenderConfig(hit_zoom_seconds=0.0, hit_closeup_seconds=0.1)
        without = tmp_path / "without.mp4"
        with_court = tmp_path / "with.mp4"

        ReviewRenderer(config).render(RenderSource(6), tracks, None, str(without))
        ReviewRenderer(config).render(
            RenderSource(6), tracks, identity_court, str(with_court)
        )

        assert frames_in(with_court) > frames_in(without)

    def test_an_explainer_replaces_the_zoom_and_closeup_beats(
        self, tmp_path, make_track, identity_court
    ):
        tracks = self.tracked(make_track, range(6))
        tracks[2].hit = Hit(position=(24, 55), is_in=True)
        tracks[2].measured_cork = (24, 55)
        explainer = CountingExplainer(count=4)
        output = tmp_path / "review.mp4"

        ReviewRenderer().render(
            RenderSource(6), tracks, identity_court, str(output), explainer
        )

        assert explainer.calls == [2]
        assert frames_in(output) == 6 + 4

    def test_the_output_frame_rate_can_be_slowed_down(self, tmp_path, make_track):
        output = tmp_path / "review.mp4"

        ReviewRenderer(RenderConfig(output_fps=10)).render(
            RenderSource(6), self.tracked(make_track, range(6)), None, str(output)
        )

        capture = cv2.VideoCapture(str(output))
        try:
            assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0)
        finally:
            capture.release()

    def test_a_clip_with_no_tracks_at_all_still_renders(self, tmp_path):
        output = tmp_path / "review.mp4"

        ReviewRenderer().render(RenderSource(4), [], None, str(output))

        assert frames_in(output) == 4


class TestVideoWriterCodec:
    """
    How the review video is encoded. OpenCV's writer is limited to the codecs
    its wheel was built with — the Linux build cannot encode H.264 at all — and
    gives no quality control, so ffmpeg is preferred and OpenCV is the fallback.
    """

    SIZE = (64, 48)  # width, height

    def test_the_defaults_ask_for_h264(self):
        config = RenderConfig()

        assert config.video_codec == "avc1"
        assert config.fallback_video_codec == "mp4v"
        assert 0 < config.video_crf < 52

    def test_ffmpeg_is_preferred(self, tmp_path):
        writer = open_video_writer(str(tmp_path / "out.mp4"), 30, self.SIZE)

        assert isinstance(writer, FFmpegWriter)
        assert writer.isOpened()
        writer.release()

    def test_it_writes_h264_in_a_playable_pixel_format(self, tmp_path):
        path = tmp_path / "out.mp4"
        writer = open_video_writer(str(path), 30, self.SIZE)
        for _ in range(10):
            writer.write(np.zeros((self.SIZE[1], self.SIZE[0], 3), dtype=np.uint8))
        writer.release()

        stream = cv2.VideoCapture(str(path))
        try:
            assert int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
        finally:
            stream.release()
        # h264 in an mp4 carries the avc1 tag; mp4v would be MPEG-4 Part 2
        assert b"avc1" in path.read_bytes()[:2048]

    def test_an_odd_frame_size_is_padded_rather_than_rejected(self, tmp_path):
        # yuv420p has no valid chroma plane for an odd dimension
        path = tmp_path / "odd.mp4"
        writer = open_video_writer(str(path), 30, (65, 49))
        for _ in range(5):
            writer.write(np.zeros((49, 65, 3), dtype=np.uint8))
        writer.release()

        assert path.stat().st_size > 0

    def test_releasing_twice_is_harmless(self, tmp_path):
        writer = open_video_writer(str(tmp_path / "out.mp4"), 30, self.SIZE)
        writer.write(np.zeros((self.SIZE[1], self.SIZE[0], 3), dtype=np.uint8))

        writer.release()
        writer.release()

        assert not writer.isOpened()

    def test_writing_after_release_is_an_error_not_a_silent_drop(self, tmp_path):
        writer = open_video_writer(str(tmp_path / "out.mp4"), 30, self.SIZE)
        writer.write(np.zeros((self.SIZE[1], self.SIZE[0], 3), dtype=np.uint8))
        writer.release()

        with pytest.raises(RuntimeError, match="already closed"):
            writer.write(np.zeros((self.SIZE[1], self.SIZE[0], 3), dtype=np.uint8))

    def test_a_lower_quality_setting_produces_a_smaller_file(self, tmp_path):
        sizes = {}
        for crf in (20, 35):
            path = tmp_path / f"crf{crf}.mp4"
            writer = open_video_writer(str(path), 30, (160, 120), crf=crf)
            for i in range(40):
                frame = np.full((120, 160, 3), 40, dtype=np.uint8)
                frame[40:80, i * 2 : i * 2 + 30] = 220
                writer.write(frame)
            writer.release()
            sizes[crf] = path.stat().st_size

        assert sizes[35] < sizes[20]

    def test_the_opencv_fallback_still_works(self, tmp_path):
        writer = open_video_writer(
            str(tmp_path / "out.mp4"), 30, self.SIZE, prefer_ffmpeg=False
        )

        assert not isinstance(writer, FFmpegWriter)
        assert writer.isOpened()
        writer.release()

    def test_the_fallback_tries_the_second_fourcc_when_the_first_is_unusable(
        self, tmp_path, caplog
    ):
        writer = open_video_writer(
            str(tmp_path / "out.mp4"),
            30,
            self.SIZE,
            codec="zzzz",
            fallback="mp4v",
            prefer_ffmpeg=False,
        )

        assert writer.isOpened()
        assert "cannot encode zzzz" in caplog.text
        writer.release()

    def test_no_usable_codec_at_all_is_an_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="Could not open"):
            open_video_writer(
                str(tmp_path / "out.mp4"),
                30,
                self.SIZE,
                codec="zzzz",
                fallback="yyyy",
                prefer_ffmpeg=False,
            )

    def test_an_unavailable_ffmpeg_falls_back_with_a_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        import sokil.render as render_module

        def no_ffmpeg(*args, **kwargs):
            raise OSError("ffmpeg not found")

        monkeypatch.setattr(render_module.FFmpegWriter, "__init__", no_ffmpeg)

        writer = open_video_writer(str(tmp_path / "out.mp4"), 30, self.SIZE)

        assert not isinstance(writer, FFmpegWriter)
        assert "Could not start ffmpeg" in caplog.text
        writer.release()

    def test_the_renderer_produces_a_smaller_file_than_the_opencv_default(
        self, tmp_path, make_track
    ):
        # the whole point of the change; a regression means review videos
        # silently get several times bigger again
        tracks = [
            make_track(number=n, xyxy=(20 + n, 50, 30 + n, 60), velocity=(4.0, 0.0))
            for n in range(60)
        ]
        sizes = {}
        for name, config in (
            ("ffmpeg", RenderConfig()),
            ("opencv", RenderConfig(prefer_ffmpeg=False, video_codec="mp4v")),
        ):
            path = tmp_path / f"{name}.mp4"
            ReviewRenderer(config).render(RenderSource(60), tracks, None, str(path))
            sizes[name] = path.stat().st_size

        assert sizes["ffmpeg"] < sizes["opencv"]

    def test_a_wrong_sized_frame_is_refused_rather_than_scrambling_the_stream(
        self, tmp_path
    ):
        # rawvideo has no frame boundaries: a short frame would shift every
        # one after it and ffmpeg would report nothing wrong
        writer = open_video_writer(str(tmp_path / "out.mp4"), 30, self.SIZE)
        try:
            with pytest.raises(ValueError, match="expected .* got .* bytes"):
                writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
        finally:
            writer.release()
