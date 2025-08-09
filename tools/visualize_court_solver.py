"""
Plot the court solver stage by stage, on one clip.

Mirrors CourtSolver.solve: sample frames, find each frame's court lines, keep the
lines that persist across frames, then pick the model-line assignment whose
homography best matches the segmentation mask. Every stage gets a panel, so a
wrong court outline can be traced to the stage that caused it.

    uv run python tools/visualize_court_solver.py VIDEO [--frame N] [--out PNG]
"""

import argparse
import logging
import textwrap
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from sokil.court import (
    Court,
    CourtCameraMode,
    CourtConfig,
    CourtModelType,
    CourtSolverConfig,
    ViewTransformer,
)
from sokil.models import Segmenter
from sokil.settings import ModelName, resolve_model_path
from sokil.util import (
    cluster_parallel_lines,
    consensus_lines,
    correspondences_from_matches,
    detect_line_segments,
    identify_line_candidates,
    split_line_families,
)
from sokil.video import VideoSource

logger = logging.getLogger(__name__)

# from the data-viz reference palette: one hue for magnitude, a second to mark
# the selected item, and text tokens that never take a series colour
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SERIES = "#2a78d6"
ACCENT = "#eb6834"

# BGR, for the OpenCV overlays
COURT_LENGTH_BGR = (60, 76, 231)  # court-length ("vertical") family
CROSS_COURT_BGR = (219, 152, 52)  # cross-court ("horizontal") family
MODEL_BGR = (80, 200, 80)
MASK_BGR = (231, 76, 231)


def infinite_line_points(rho: float, theta: float, span: int = 3000):
    """Two far-apart points on the infinite line rho = x·cos(theta) + y·sin(theta)."""
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    return (
        (int(x0 + span * -b), int(y0 + span * a)),
        (int(x0 - span * -b), int(y0 - span * a)),
    )


def draw_segments(image, segments, color, thickness=2):
    for segment in segments:
        (x1, y1), (x2, y2) = segment["points"]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return image


def draw_lines(image, lines, color, thickness=2):
    for line in lines:
        start, end = infinite_line_points(line["rho"], line["theta"])
        cv2.line(image, start, end, color, thickness)
    return image


def rgb(image):
    """OpenCV BGR (or single-channel) to something imshow renders correctly."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def tinted(image, mask, color, strength=0.32):
    out = image.copy()
    out[mask > 0] = (
        (1 - strength) * out[mask > 0] + strength * np.array(color)
    ).astype(np.uint8)
    return out


def score_assignments(court, verticals, horizontals):
    """
    Every candidate model-line assignment, scored the way the solver scores them.

    :returns: (rows, best) where rows are (x_labels, y_labels, iou, transformer)
        keyed on which model lines the assignment claims, best-first.
    """
    candidates = identify_line_candidates(
        verticals,
        horizontals,
        court.court_model.stripe_center_coords("x"),
        court.court_model.stripe_center_coords("y"),
    )

    best_per_assignment = defaultdict(lambda: (-1.0, None, ()))
    for matches in candidates:
        source_pts, target_pts = correspondences_from_matches(matches)
        if len(source_pts) < 4:
            continue
        try:
            transformer = ViewTransformer(source=source_pts, target=target_pts)
        except ValueError:
            continue

        # gates off, so the panel shows the rejected options too
        scored = court._transformer_score(
            transformer, source_pts, target_pts, max_rms=1e9, min_iou=0.0
        )
        if scored is None:
            continue

        _, iou = scored
        key = tuple(sorted({round(float(v)) for v in target_pts[:, 0]}))
        y_labels = tuple(sorted({round(float(v)) for v in target_pts[:, 1]}))
        if iou > best_per_assignment[key][0]:
            best_per_assignment[key] = (iou, transformer, y_labels)

    rows = [
        (key, value[2], value[0], value[1])
        for key, value in best_per_assignment.items()
    ]
    rows.sort(key=lambda row: -row[2])
    return rows, (rows[0] if rows else None)


def project_model(image, court, transformer):
    out = image.copy()
    model = court.court_model
    length, width = model.length, model.width

    for x in model.stripe_center_coords("x"):
        pts = transformer.inverse_transform_points(
            np.array([[x, 0], [x, length]], np.float32)
        ).astype(int)
        cv2.line(out, tuple(pts[0]), tuple(pts[1]), MODEL_BGR, 2)
    for y in model.stripe_center_coords("y"):
        pts = transformer.inverse_transform_points(
            np.array([[0, y], [width, y]], np.float32)
        ).astype(int)
        cv2.line(out, tuple(pts[0]), tuple(pts[1]), MODEL_BGR, 1)

    boundary = np.array(model.doubles_boundaries(), dtype=np.float32)
    cv2.polylines(
        out,
        [transformer.inverse_transform_points(boundary).astype(np.int32)],
        True,
        (40, 220, 255),
        2,
    )
    return out


def caption(ax, text, width=74, offset=-0.045):
    """Wrapped grey caption under a panel, in axes coordinates."""
    ax.text(
        0.0,
        offset,
        "\n".join(textwrap.wrap(text, width)),
        transform=ax.transAxes,
        fontsize=7.6,
        color=INK_MUTED,
        va="top",
        ha="left",
        linespacing=1.5,
    )


def panel(ax, image, step, title, text):
    ax.imshow(rgb(image))
    ax.set_title(f"{step}. {title}", fontsize=10.5, color=INK, loc="left", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#dedddA")
    caption(ax, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="frame to illustrate with (default: the one the solver solves on)",
    )
    parser.add_argument("--out", default="docs/court-solver-walkthrough.png")
    parser.add_argument("--sample-seconds", type=float, default=None)
    parser.add_argument("--segmentation-model", default=None)
    parser.add_argument(
        "--camera-mode", default="full", choices=[m.value for m in CourtCameraMode]
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("ultralytics", "matplotlib", "PIL", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    model_path = args.segmentation_model or resolve_model_path(ModelName.COURT)
    segmenter = Segmenter(model_path)
    detector_kwargs = {
        "segmentation_model": segmenter,
        "keypoint_model": None,
        "court_config": CourtConfig(
            court_camera_mode=CourtCameraMode(args.camera_mode),
            court_model_type=CourtModelType.DOUBLES,
        ),
    }

    solver_config = CourtSolverConfig()
    if args.sample_seconds is not None:
        solver_config.sample_seconds = args.sample_seconds

    source = VideoSource(args.video, grayscale=True)
    stride = max(1, round((source.fps or 30) * solver_config.sample_seconds))
    sampled = [(d.number, d.image) for d in source.sample(stride)]
    logger.info(
        "%s: sampling every %d frames -> %d frames",
        source.describe(),
        stride,
        len(sampled),
    )

    # --- per-frame line detection, exactly as the solver does it
    per_frame = []
    for number, image in sampled:
        court = Court(**detector_kwargs, image=image)
        verticals, horizontals = court.detect_court_lines()
        per_frame.append((number, image, verticals, horizontals))

    consensus_v = consensus_lines(
        [f[2] for f in per_frame],
        min_frame_fraction=solver_config.min_frame_fraction,
        cluster_eps=solver_config.cluster_eps,
    )
    consensus_h = consensus_lines(
        [f[3] for f in per_frame],
        min_frame_fraction=solver_config.min_frame_fraction,
        cluster_eps=solver_config.cluster_eps,
    )

    # the solver solves on the middle sampled frame; illustrate with that one
    # unless asked otherwise, so the panels match what actually happened
    middle = len(per_frame) // 2
    if args.frame is None:
        number, frame, verticals, horizontals = per_frame[middle]
    else:
        capture = cv2.VideoCapture(args.video)
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise SystemExit(f"could not read frame {args.frame}")
        number = args.frame
        court = Court(**detector_kwargs, image=frame)
        verticals, horizontals = court.detect_court_lines()

    court = Court(**detector_kwargs, image=frame)
    height, width = frame.shape[:2]

    # --- stages
    contour = court.get_contour()
    mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.add(
        cv2.morphologyEx(
            gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        ),
        cv2.morphologyEx(
            gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        ),
    )
    edges = court._court_edge_image()
    segments = detect_line_segments(edges)
    raw_v, raw_h = split_line_families(segments)
    # the same thresholds Court.detect_court_lines just used, so this panel
    # shows the clustering the solve actually ran on
    clustering = {
        "eps": court.court_config.line_cluster_eps,
        "straightness_tau": court.court_config.line_straightness_tau,
        "stripe_merge_eps": court.court_config.stripe_merge_eps,
    }
    clustered_v = cluster_parallel_lines(raw_v, frame, vertical=True, **clustering)
    clustered_h = cluster_parallel_lines(raw_h, frame, vertical=False, **clustering)

    rows, best = score_assignments(court, consensus_v, consensus_h)
    if best is None:
        raise SystemExit("no candidate assignment produced a homography")
    _, _, best_iou, best_transformer = best

    # --- figure
    fig = plt.figure(figsize=(17.0, 13.6), facecolor=SURFACE)
    grid = GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.10)
    axes = [fig.add_subplot(grid[i // 3, i % 3]) for i in range(9)]

    panel(
        axes[0],
        frame,
        1,
        "Sampled frame",
        f"frame {number} of {len(sampled)} sampled every {stride} frames "
        f"({solver_config.sample_seconds:g}s apart)",
    )

    search = tinted(frame, dilated, (60, 220, 220))
    cv2.drawContours(search, [contour], -1, (60, 60, 231), 2)
    panel(
        axes[1],
        search,
        2,
        "Court mask, dilated",
        "the segmentation contour (red) grown by a 20px ellipse (yellow). Lines "
        "are only looked for inside it, so a player's legs cut holes in the "
        "searchable area",
    )

    panel(
        axes[2],
        tophat,
        3,
        "Top-hat: thin bright structures",
        "a morphological top-hat keeps painted stripes and drops the smooth "
        "floor, so no colour threshold is needed",
    )

    panel(
        axes[3],
        edges,
        4,
        "Canny edges inside the mask",
        f"{int((edges > 0).sum())} edge pixels. Each stripe contributes two "
        "edges, one per painted border",
    )

    families = draw_segments(frame.copy(), raw_h, CROSS_COURT_BGR)
    families = draw_segments(families, raw_v, COURT_LENGTH_BGR)
    panel(
        axes[4],
        families,
        5,
        "Segments split into two families",
        f"{len(raw_v)} court-length (red) and {len(raw_h)} cross-court (blue), "
        "separated by image angle",
    )

    clusters = draw_lines(frame.copy(), clustered_h, CROSS_COURT_BGR, 1)
    clusters = draw_lines(clusters, clustered_v, COURT_LENGTH_BGR, 2)
    panel(
        axes[5],
        clusters,
        6,
        "Clustered to one line per stripe",
        f"{len(clustered_v)} court-length and {len(clustered_h)} cross-court "
        "centrelines: a stripe's two edges collapse into the line through its "
        "middle",
    )

    persisted = frame.copy()
    for _, _, frame_v, frame_h in per_frame:
        persisted = draw_lines(persisted, frame_v, (150, 150, 150), 1)
        persisted = draw_lines(persisted, frame_h, (150, 150, 150), 1)
    persisted = draw_lines(persisted, consensus_h, CROSS_COURT_BGR, 1)
    persisted = draw_lines(persisted, consensus_v, COURT_LENGTH_BGR, 2)
    panel(
        axes[6],
        persisted,
        7,
        "Consensus across the sampled frames",
        f"every frame's lines in grey; the {len(consensus_v)}+{len(consensus_h)} "
        f"that recur in ≥{solver_config.min_frame_fraction:.0%} of frames "
        "survive in colour. Occlusions come and go, real lines do not",
    )

    # panel 8: how the winning assignment was chosen
    ax = axes[7]
    ax.set_facecolor(SURFACE)
    shown = rows[:10]
    labels = [f"{r[0][0]} & {r[0][1]} cm" for r in shown]
    values = [r[2] for r in shown]
    colors = [ACCENT if index == 0 else SERIES for index in range(len(shown))]
    positions = np.arange(len(shown))[::-1]

    ax.barh(positions, values, height=0.62, color=colors)
    for position, value in zip(positions, values):
        ax.text(
            value + 0.012,
            position,
            f"{value:.3f}",
            va="center",
            fontsize=7.5,
            color=INK_MUTED,
        )
    ax.set_yticks(positions, labels, fontsize=7.5, color=INK)
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel(
        "mask IoU of the resulting homography",
        fontsize=7.6,
        color=INK_MUTED,
        labelpad=4,
    )
    ax.set_title(
        "8. Candidate assignments, scored", fontsize=10, color=INK, loc="left", pad=6
    )
    ax.tick_params(length=0)
    ax.grid(axis="x", color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#dedddA")
    if len(rows) > 1:
        margin = (
            f"Orange wins on {best_iou:.3f} against {rows[1][2]:.3f}. A margin this "
            f"narrow is decided by mask overlap alone, which a dilated mask measures "
            f"only coarsely."
            if best_iou - rows[1][2] < 0.05
            else f"Orange wins on {best_iou:.3f} against {rows[1][2]:.3f}, a clear "
            f"margin, so the width is settled by the evidence here."
        )
    else:
        margin = (
            f"Only one assignment scored at all ({best_iou:.3f}), so there was "
            f"nothing to choose between."
        )
    caption(
        ax,
        "which pair of court-length lines the two detected stripes are "
        "taken to be — the choice that sets the court's width. " + margin,
        width=66,
        offset=-0.20,
    )

    panel(
        axes[8],
        project_model(frame, court, best_transformer),
        9,
        "The winning homography, applied",
        f"the full court model projected back onto the frame (IoU {best_iou:.3f}). "
        "Lines far from the measured ones are extrapolated, so they drift first",
    )

    fig.suptitle(
        "How the court solver turns a frame into a homography",
        fontsize=14,
        color=INK,
        x=0.008,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.008,
        0.958,
        f"{Path(args.video).name}  ·  camera mode {args.camera_mode}  ·  "
        f"stages 1-6 run per frame, 7 pools them, 8-9 solve once",
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
