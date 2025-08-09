"""
Plot how a hit is detected, stage by stage, on one clip.

Mirrors detect_hits: for every frame the shuttle was seen on, fit the direction
it arrived from and the direction it left in, measure how far that turned and
how much speed was lost, then thin the resulting run of suspicious frames down
to one hit.

    uv run python tools/visualize_hit_detection.py VIDEO [--out PNG]
"""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from sokil.hits import HitDetectionConfig, HitDetector, is_flight_broken
from sokil.settings import ModelName, resolve_model_path
from sokil.tracker import ShuttleTracker, TrackerConfig
from sokil.util import angle_between, fit_velocity
from sokil.video import VideoSource

logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
BEFORE = "#2a78d6"  # the shuttle arriving
AFTER = "#eb6834"  # the shuttle leaving, and the accepted hit
GRID = "#e8e7e3"


def per_frame_measurements(frame_numbers, positions, config):
    """
    Recompute what detect_hits works out for every frame.

    :returns: a dict of parallel lists, one entry per detection.
    """
    count = len(frame_numbers)
    out = {
        key: [np.nan] * count
        for key in ("incoming_speed", "outgoing_speed", "turn", "speed_lost", "score")
    }
    out["candidate"] = [False] * count
    out["incoming_velocity"] = [None] * count
    out["outgoing_velocity"] = [None] * count

    for index in range(count):
        frames_before, points_before = [frame_numbers[index]], [positions[index]]
        scan = index - 1
        while (
            scan >= 0
            and len(frames_before) <= config.window
            and not is_flight_broken(frame_numbers, positions, scan + 1, config)
        ):
            frames_before.insert(0, frame_numbers[scan])
            points_before.insert(0, positions[scan])
            scan -= 1
        if len(frames_before) < 2:
            continue

        incoming = fit_velocity(frames_before, points_before)
        if incoming is None:
            continue
        out["incoming_velocity"][index] = incoming
        out["incoming_speed"][index] = float(np.linalg.norm(incoming))

        frames_after, points_after = [frame_numbers[index]], [positions[index]]
        scan = index + 1
        while (
            scan < count
            and len(frames_after) <= config.window
            and not is_flight_broken(frame_numbers, positions, scan, config)
        ):
            frames_after.append(frame_numbers[scan])
            points_after.append(positions[scan])
            scan += 1
        if len(frames_after) < 2:
            continue

        outgoing = fit_velocity(frames_after, points_after)
        if outgoing is None:
            continue
        out["outgoing_velocity"][index] = outgoing
        out["outgoing_speed"][index] = float(np.linalg.norm(outgoing))

        speed_in = out["incoming_speed"][index]
        if speed_in < config.min_incoming_speed:
            continue

        if config.ground_hits_only and incoming[1] <= 0:
            continue

        out["turn"][index] = angle_between(incoming, outgoing)
        out["speed_lost"][index] = max(
            0.0, (speed_in - out["outgoing_speed"][index]) / speed_in
        )
        out["score"][index] = out["turn"][index] + 90.0 * out["speed_lost"][index]
        out["candidate"][index] = (
            out["turn"][index] >= config.angle_change_threshold
            or out["speed_lost"][index] >= config.speed_drop_threshold
        )
    return out


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK_MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK_MUTED)
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#dedddA")


def caption(ax, text, offset=-0.30):
    ax.text(
        0.0,
        offset,
        text,
        transform=ax.transAxes,
        fontsize=7.8,
        color=INK_MUTED,
        va="top",
        ha="left",
        linespacing=1.5,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--out", default="docs/hit-detection-walkthrough.png")
    parser.add_argument("--shuttle-model", default=None)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("ultralytics", "matplotlib", "PIL", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    model_path = args.shuttle_model or resolve_model_path(ModelName.SHUTTLE)
    source = VideoSource(args.video, grayscale=True)
    tracks = ShuttleTracker(
        {"model": str(model_path)},
        TrackerConfig(device="cpu", confidence=args.confidence),
    ).track(source)

    frame_numbers = [t.number for t in tracks]
    positions = [t.cork_position for t in tracks]
    config = HitDetectionConfig()
    config = replace(config, max_jump=config.resolved_max_jump(source.width))

    measured = per_frame_measurements(frame_numbers, positions, config)
    accepted = (
        HitDetector()
        .detect(tracks, None, source.fps, frame_width=source.width)
        .hit_indices
    )
    if not accepted:
        raise SystemExit("no hit was detected in this clip")
    hit = accepted[0]
    hit_frame = frame_numbers[hit]
    logger.info("Hit at frame %d (detection %d of %d)", hit_frame, hit, len(tracks))

    capture = cv2.VideoCapture(args.video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, hit_frame)
    _, frame = capture.read()
    capture.release()

    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])

    fig = plt.figure(figsize=(16.5, 10.2), facecolor=SURFACE)
    grid = GridSpec(2, 3, figure=fig, hspace=0.62, wspace=0.26)
    axes = [fig.add_subplot(grid[i // 3, i % 3]) for i in range(6)]

    # 1 — the flight drawn on the hit frame
    ax = axes[0]
    canvas = frame.copy()
    for k in range(1, len(positions)):
        arriving = k <= hit
        colour_rgb = (42, 120, 214) if arriving else (235, 104, 52)
        cv2.line(
            canvas,
            tuple(np.int32(positions[k - 1])),
            tuple(np.int32(positions[k])),
            colour_rgb[::-1],
            2,
        )
    for velocity, colour in (
        (measured["incoming_velocity"][hit], (42, 120, 214)),
        (measured["outgoing_velocity"][hit], (235, 104, 52)),
    ):
        if velocity is None:
            continue
        tip = np.int32(np.array(positions[hit]) + np.array(velocity) * 6)
        cv2.arrowedLine(
            canvas,
            tuple(np.int32(positions[hit])),
            tuple(tip),
            colour[::-1],
            3,
            tipLength=0.3,
        )
    cv2.circle(canvas, tuple(np.int32(positions[hit])), 9, (255, 255, 255), 2)
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "1. The flight, on the hit frame", fontsize=10.5, color=INK, loc="left", pad=6
    )
    caption(
        ax,
        "blue = arriving, orange = leaving. The two arrows are the\n"
        "fitted directions the detector compares.",
    )

    # 2 — the same path as a plain graph
    ax = axes[1]
    ax.plot(
        xs[: hit + 1],
        ys[: hit + 1],
        "-o",
        color=BEFORE,
        ms=3.4,
        lw=1.6,
        label="arriving",
    )
    ax.plot(xs[hit:], ys[hit:], "-o", color=AFTER, ms=3.4, lw=1.6, label="leaving")
    ax.plot(xs[hit], ys[hit], "o", ms=11, mfc="none", mec=INK, mew=1.6)
    ax.annotate(
        f"hit, frame {hit_frame}",
        (xs[hit], ys[hit]),
        textcoords="offset points",
        xytext=(12, 10),
        fontsize=8,
        color=INK,
    )
    ax.invert_yaxis()
    style(ax, "2. Same path, without the picture", "x (px)", "y (px, down)")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)
    caption(
        ax,
        "the shuttle comes down, then leaves in a new direction.\n"
        "That corner is what the detector is hunting for.",
    )

    # 3 — speed before and after
    ax = axes[2]
    ax.plot(
        frame_numbers,
        measured["incoming_speed"],
        color=BEFORE,
        lw=1.8,
        label="speed arriving",
    )
    ax.plot(
        frame_numbers,
        measured["outgoing_speed"],
        color=AFTER,
        lw=1.8,
        label="speed leaving",
    )
    ax.axvline(hit_frame, color=INK, lw=1, ls="--")
    style(ax, "3. Speed, before and after each frame", "frame", "px per frame")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)
    caption(
        ax,
        "at the impact the arriving speed stays high while the leaving\n"
        "speed collapses — the shuttle is stopped by the floor.",
    )

    # 4 — how far the direction turned
    ax = axes[3]
    ax.plot(frame_numbers, measured["turn"], color=BEFORE, lw=1.8, label="turn")
    ax.axhline(
        config.angle_change_threshold,
        color=AFTER,
        lw=1.4,
        ls="--",
        label=f"threshold {config.angle_change_threshold:g}°",
    )
    ax.axvline(hit_frame, color=INK, lw=1, ls="--")
    style(ax, "4. Ingredient one: the turn", "frame", "degrees")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)
    caption(
        ax,
        "how far the direction changed. In steady flight it sits near\n"
        "zero; a floor hit swings it right round.",
    )

    # 5 — how much speed was lost
    ax = axes[4]
    ax.plot(
        frame_numbers, measured["speed_lost"], color=BEFORE, lw=1.8, label="speed lost"
    )
    ax.axhline(
        config.speed_drop_threshold,
        color=AFTER,
        lw=1.4,
        ls="--",
        label=f"threshold {config.speed_drop_threshold:g}",
    )
    ax.axvline(hit_frame, color=INK, lw=1, ls="--")
    style(ax, "5. Ingredient two: the speed lost", "frame", "fraction of speed lost")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED)
    caption(
        ax,
        "1.0 would mean the shuttle stopped dead. Either ingredient on\n"
        "its own is enough to make a frame suspicious.",
    )

    # 6 — the score, the suspicious frames, and the one that survives
    ax = axes[5]
    score = np.array(measured["score"], dtype=float)
    ax.plot(frame_numbers, score, color=INK_MUTED, lw=1.4, zorder=1)
    cand = [i for i, c in enumerate(measured["candidate"]) if c]
    ax.scatter(
        [frame_numbers[i] for i in cand],
        score[cand],
        s=44,
        color=BEFORE,
        zorder=3,
        label=f"suspicious ({len(cand)})",
    )
    ax.scatter(
        [frame_numbers[i] for i in accepted],
        score[accepted],
        s=150,
        facecolor="none",
        edgecolor=AFTER,
        linewidth=2.4,
        zorder=4,
        label=f"accepted ({len(accepted)})",
    )
    refractory = config.refractory_seconds * source.fps
    ax.axvspan(hit_frame, hit_frame + refractory, color=AFTER, alpha=0.10, zorder=0)
    ax.set_ylim(top=float(np.nanmax(score)) * 1.42)
    ax.annotate(
        "ignored: the bounce",
        (hit_frame + refractory * 0.5, 0),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=7.6,
        color=INK_MUTED,
    )
    style(ax, "6. Both ingredients combined, then thinned", "frame", "impact score")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED, loc="upper right")
    caption(
        ax,
        "several frames around one impact look suspicious. The strongest\n"
        "is kept; anything inside the shaded window after it is the\n"
        "shuttle bouncing, not a new shot.",
    )

    fig.suptitle(
        "How a hit is detected", fontsize=14, color=INK, x=0.008, ha="left", y=0.985
    )
    fig.text(
        0.008,
        0.955,
        f"{Path(args.video).name}  ·  {len(tracks)} frames with a shuttle  ·  "
        f"hit found at frame {hit_frame}",
        fontsize=8.5,
        color=INK_MUTED,
        ha="left",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
