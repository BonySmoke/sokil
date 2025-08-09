"""
Rendering the review video.
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from rich.progress import Progress

from .court import Court
from .frame import Frame, ShuttleFrame, ShuttleTrack, tracks_by_number
from .util import fig_to_numpy, zoom_inset
from .video import VideoSource

if TYPE_CHECKING:
    # only for the annotation: visualization imports from this module, so
    # importing the explanation at runtime would close the circle
    from .explanation import Explainer

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    """Knobs for the review video (see ReviewRenderer)."""

    show_frame_number: bool = True
    show_speed: bool = False
    show_angle_change: bool = False
    show_track: bool = True
    show_shuttle_contour: bool = True
    show_direction: bool = False

    # frame rate written into the review video. None keeps the source rate.
    output_fps: float | None = None

    # How long to show the frame where the shuttle contacts the court
    hit_zoom_seconds: float = 0.5
    # how long to show the top-down view onto the area where the shuttle hit the ground
    hit_closeup_seconds: float = 2.0
    closeup_dpi: int = 100
    # the close-up is rendered at this multiple of the frame size and downscaled
    # back, so the centered fit anti-aliases its lines and text
    closeup_supersample: int = 2
    closeup_background: tuple[int, int, int] = (34, 139, 34)  # forestgreen (BGR)

    # How the review video is encoded. ffmpeg encodes H.264 and is preferred;
    # OpenCV's own writer is the fallback and costs roughly 4-8x the bitrate for
    # the same picture — its Linux wheel cannot encode H.264 at all.
    prefer_ffmpeg: bool = True  # set False to force OpenCV's own writer
    video_crf: int = 23  # x264 quality, lower is better; 18 is ~visually lossless
    video_preset: str = "medium"  # x264 speed/size trade-off
    # fourccs for the OpenCV fallback, tried in this order
    video_codec: str = "avc1"
    fallback_video_codec: str = "mp4v"

    explain: bool = False
    # how long the contact frame is held before the first wipe
    explain_freeze_seconds: float = 1.0
    # how long each stage after it is held
    explain_stage_seconds: float = 3.0
    # how long the slider takes to wipe from one stage to the next
    explain_transition_seconds: float = 0.7
    explain_observable_padding: float = 50.0


class ReviewRenderer:
    """
    Writes the review video: one decode pass, drawing each frame's track on it.

    Frames are re-decoded here rather than kept from the tracking pass, so the
    renderer pairs each image with the ShuttleTrack measured on that frame
    number. Hits additionally get a full-frame court close-up held for a beat.
    """

    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()

    def assign_track_points(
        self, tracks: list[ShuttleTrack], trajectories: list[list[int]]
    ):
        """
        Give every trajectory a color and the growing polyline drawn for it.

        Each track in a trajectory keeps the points up to and including itself,
        so the trail draws itself out as the video plays.
        """
        for trajectory in trajectories:
            color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
            trajectory_points = []
            for index in trajectory:
                shuttle_track = tracks[index]
                trajectory_points.append(shuttle_track.bbox_center)
                shuttle_track.track_color = color
                shuttle_track.track_points = trajectory_points.copy()

    def render(
        self,
        source: VideoSource,
        tracks: list[ShuttleTrack],
        court: Court | None,
        output_path: str,
        explainer: "Explainer | None" = None,
    ):
        """
        Render the whole video to output_path.

        :param explainer: when given (and a court was solved), each landing is
            explained stage by stage.
        """
        logger.info("Rendering review video -> %s", output_path)

        track_index = tracks_by_number(tracks)
        court_contour = (
            court.get_court_projection_points() if court is not None else None
        )

        output_fps = self.config.output_fps or round(source.fps)
        if source.fps and output_fps < source.fps:
            logger.info(
                "Rendering at %g fps from %g fps source (%.1fx slow motion)",
                output_fps,
                source.fps,
                source.fps / output_fps,
            )

        output = open_video_writer(
            output_path,
            output_fps,
            (source.width, source.height),
            crf=self.config.video_crf,
            preset=self.config.video_preset,
            codec=self.config.video_codec,
            fallback=self.config.fallback_video_codec,
            prefer_ffmpeg=self.config.prefer_ffmpeg,
        )

        progress = Progress()
        progress_task = progress.add_task(
            "Rendering review", total=source.frame_count or None
        )
        progress.start()

        try:
            for decoded in source:
                progress.advance(progress_task)
                shuttle_track = track_index.get(decoded.number)

                if shuttle_track is None:
                    output.write(
                        Frame(frame=decoded.image, number=decoded.number).draw(
                            show_frame_number=self.config.show_frame_number
                        )
                    )
                    continue

                shuttle_frame = ShuttleFrame(
                    frame=decoded.image,
                    number=decoded.number,
                    track=shuttle_track,
                    court=court,
                    # the court outline is drawn on the hit frames, where the
                    # in/out verdict is on screen
                    court_contour=court_contour if shuttle_track.hit else None,
                )
                drawn = shuttle_frame.draw(
                    show_angle_change=self.config.show_angle_change,
                    show_speed=self.config.show_speed,
                    show_frame_number=self.config.show_frame_number,
                    show_track=self.config.show_track,
                    show_shuttle_contour=self.config.show_shuttle_contour,
                    show_direction=self.config.show_direction,
                )
                output.write(drawn)

                if not shuttle_track.hit:
                    continue

                if explainer is not None and court is not None:
                    for image in explainer.frames(
                        decoded.image, shuttle_track, court, output_fps
                    ):
                        output.write(image)
                    continue

                zoom_frames = int(output_fps * self.config.hit_zoom_seconds)
                if zoom_frames:
                    zoomed = zoom_inset(drawn, shuttle_track.cork_position)
                    for _ in range(zoom_frames):
                        output.write(zoomed)

                # the verdict needs a court; the magnified beat above does not
                if court is None:
                    continue

                closeup = self._hit_closeup(shuttle_track, court, source)
                if closeup is None:
                    logger.warning(
                        "Could not plot the hit close-up for frame %d, skipping",
                        decoded.number,
                    )
                    continue

                held_frames = int(output_fps * self.config.hit_closeup_seconds)
                for _ in range(held_frames):
                    output.write(closeup)
        finally:
            progress.stop()
            output.release()

        logger.info("Review video written -> %s", output_path)

    def _hit_closeup(
        self, shuttle_track: ShuttleTrack, court: Court, source: VideoSource
    ):
        """
        Build the frame-sized court close-up shown after a hit.

        Supersample: render at SS x the frame size and let the centered
        downscale anti-alias it, for crisp lines and text. figsize matches the
        frame aspect so the close-up window fills the frame.
        """
        return hit_closeup(
            court=court,
            shuttle_track=shuttle_track,
            width=source.width,
            height=source.height,
            dpi=self.config.closeup_dpi,
            supersample=self.config.closeup_supersample,
            background=self.config.closeup_background,
        )


class FFmpegWriter:
    """
    Writes frames by piping them to ffmpeg, which encodes H.264.

    cv2.VideoWriter can only use the codecs its wheel was built with — the
    Linux build cannot encode H.264 at all — and exposes no quality control, so
    a review video written through it costs several times the bitrate it needs.
    ffmpeg comes from imageio-ffmpeg, which ships a static binary as a wheel, so
    this needs nothing installed on the host.

    Mirrors the part of the cv2.VideoWriter interface the renderer uses, so
    either can be returned by open_video_writer.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: tuple[int, int],
        crf: int = 23,
        preset: str = "medium",
    ):
        from imageio_ffmpeg import get_ffmpeg_exe

        width, height = frame_size
        self.output_path = output_path
        self._frame_size = (width, height)
        self._frame_bytes = width * height * 3

        self._process = subprocess.Popen(
            [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                # the renderer hands over raw OpenCV frames, which are BGR
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                # yuv420p is what browsers and QuickTime will actually play; an
                # odd dimension has no valid chroma plane, so pad to even first
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                # moov atom first, so the file streams rather than needing a
                # full download before it starts
                "-movflags",
                "+faststart",
                output_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def isOpened(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if self._process is None:
            raise RuntimeError(f"{self.output_path} is already closed")

        # rawvideo carries no frame boundaries, so a wrong-sized frame is not
        # an error to ffmpeg — it just shifts every subsequent frame and the
        # video comes out scrambled. Cheaper to refuse it here.
        if frame.nbytes != self._frame_bytes:
            raise ValueError(
                f"expected {self._frame_bytes}-byte frames "
                f"({self._frame_size[0]}x{self._frame_size[1]} BGR), "
                f"got {frame.nbytes} bytes with shape {frame.shape}"
            )

        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            # ffmpeg died mid-render; its stderr says why
            self.release()

    def release(self) -> None:
        """Close the pipe and wait for ffmpeg to finish writing the file."""
        process, self._process = self._process, None
        if process is None:
            return

        try:
            process.stdin.close()
        except (BrokenPipeError, ValueError):
            pass  # ffmpeg already exited and took the pipe with it

        # Drained rather than left to communicate(), which would try to flush
        # the stdin we just closed. Safe against filling the pipe buffer
        # because -loglevel error keeps ffmpeg near-silent unless it fails.
        stderr = process.stderr.read() if process.stderr else b""
        process.wait()

        if process.returncode:
            raise RuntimeError(
                f"ffmpeg failed to write {self.output_path} "
                f"(exit {process.returncode}): "
                f"{stderr.decode(errors='replace').strip()[-500:]}"
            )


def open_video_writer(
    output_path: str,
    fps: float,
    frame_size: tuple[int, int],
    crf: int = 23,
    preset: str = "medium",
    codec: str = "avc1",
    fallback: str | None = "mp4v",
    prefer_ffmpeg: bool = True,
):
    """
    Open a writer for the review video, preferring ffmpeg's H.264 encoder.

    Falls back to cv2.VideoWriter when ffmpeg cannot be used. Whether
    OpenCV can then encode H.264 depends on how its wheel was built, and the
    failure is a writer that never opens rather than an exception — so each
    fourcc is checked here instead of leaving the caller with a dead writer
    that silently drops every frame.

    :param frame_size: (width, height), the order cv2.VideoWriter expects.
    """
    if prefer_ffmpeg:
        try:
            return FFmpegWriter(output_path, fps, frame_size, crf=crf, preset=preset)
        except (ImportError, OSError) as error:
            logger.warning(
                "Could not start ffmpeg (%s); falling back to OpenCV, which "
                "writes a much larger file for the same picture.",
                error,
            )

    for candidate in (codec, fallback):
        if not candidate:
            continue
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*candidate), fps, frame_size
        )
        if writer.isOpened():
            if candidate != codec:
                logger.warning(
                    "This OpenCV build cannot encode %s either; wrote %s with "
                    "%s, which is several times larger for the same picture.",
                    codec,
                    output_path,
                    candidate,
                )
            return writer
        writer.release()

    raise RuntimeError(
        f"Could not open {output_path} for writing with {codec} or {fallback}"
    )


def hit_closeup(
    court: Court,
    shuttle_track: ShuttleTrack,
    width: int,
    height: int,
    dpi: int = 100,
    supersample: int = 2,
    background: tuple[int, int, int] = (34, 139, 34),
):
    """
    The umpire close-up of one landing, as a frame-sized BGR image.

    A tight top-down view centred on the shuttle, showing the line that decides
    the call and the margin in centimetres (see draw_hit_closeup_matplotlib).

    :returns: None when the court has no homography to plot with.
    """
    fig, _ = court.plot(
        shuttle_position=shuttle_track.cork_position,
        is_shuttle_in=shuttle_track.hit.is_in if shuttle_track.hit else None,
        figsize=(width / dpi, height / dpi),
        dpi=dpi * supersample,
    )
    if not fig:
        return None

    return figure_to_frame(fig, width=width, height=height, background=background)


def figure_to_frame(
    fig,
    width: int,
    height: int,
    background: tuple[int, int, int] = (34, 139, 34),
):
    """
    Rasterize a matplotlib figure and center it on a frame-sized BGR canvas.

    The figure is drawn oversized and scaled down here, so the downscale
    anti-aliases its lines and text; whatever aspect it has is letterboxed
    against `background` rather than stretched.
    """
    figure_bgr = cv2.cvtColor(fig_to_numpy(fig), cv2.COLOR_RGBA2BGR)

    canvas = np.full((height, width, 3), background, dtype=np.uint8)

    figure_height, figure_width = figure_bgr.shape[:2]
    scale = min(width / figure_width, height / figure_height)
    new_w = max(1, int(figure_width * scale))
    new_h = max(1, int(figure_height * scale))
    resized = cv2.resize(figure_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    return canvas
