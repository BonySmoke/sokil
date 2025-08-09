"""
Describing a review run: the per-frame CSV export and the analytics plots.

Both work purely off the shuttle tracks, so they need neither the video nor
any model — a saved track list is enough to re-export or re-plot a run.
"""

import csv
import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from .frame import ShuttleTrack

logger = logging.getLogger(__name__)


class StatsExporter:
    """Writes the per-frame shuttle stats used for labeling and grid search."""

    def export(
        self,
        tracks: list[ShuttleTrack],
        video: str,
        output_path: str | None = None,
        append: bool = False,
    ) -> list[dict]:
        """
        :param tracks: the shuttle tracks to export.
        :param video: video identifier written into every row.
        :param output_path: if given, also write the rows to this CSV.
        :param append: append to output_path instead of overwriting (writes the
            header only when the file is new/empty) — use this to accumulate
            several videos into one combined dataset.
        :returns: list[dict] rows (load with pandas.DataFrame(...) if desired).
        """
        rows = []
        for shuttle_track in tracks:
            cx, cy = shuttle_track.cork_position
            rows.append(
                {
                    "video": video,
                    "frame": shuttle_track.number,
                    "cx": round(float(cx), 2),
                    "cy": round(float(cy), 2),
                    "speed": round(float(shuttle_track.speed), 3),
                    "angle_change": round(float(shuttle_track.angle_change), 3),
                    "is_moving": bool(shuttle_track.is_moving),
                    "is_hit": shuttle_track.hit is not None,
                    "is_inside_court": (
                        shuttle_track.hit.is_in if shuttle_track.hit else None
                    ),
                    "timestamp": shuttle_track.timestamp,
                }
            )

        if output_path and rows:
            path = Path(output_path)
            mode = "a" if append else "w"
            write_header = not (append and path.exists() and path.stat().st_size > 0)
            with open(path, mode, newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)

            logger.info(
                "%s %d shuttle frames (%s) -> %s",
                "Appended" if append else "Wrote",
                len(rows),
                video,
                output_path,
            )

        return rows


class Analytics:
    """Diagnostic plots of the shuttle motion over a run."""

    def plot(self, tracks: list[ShuttleTrack]):
        """:returns: the matplotlib (fig, axs) — trajectory, speed, angle,
        direction and blur over time, with the detected hits marked."""
        track_history = [t.bbox_center for t in tracks]
        velocity_history = [t.speed for t in tracks]
        angle_history = [t.angle_change for t in tracks]
        timestamp_history = [t.timestamp for t in tracks]
        direction_angles = [np.degrees(np.arctan2(*t.velocity)) for t in tracks]
        blur_quantity = [t.blur for t in tracks]

        hit_candidates = [i for i, t in enumerate(tracks) if t.hit]

        points = np.array(track_history)
        x_centers = points[:, 0]
        y_centers = points[:, 1]

        fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=False)

        axs[0].plot(x_centers, y_centers, "--o", color="purple", label="trajectory")
        for i, (xi, yi) in enumerate(points, start=1):
            axs[0].text(xi + 0.10, yi + 0.10, str(i), fontsize=10, color="red")

        axs[0].set_xlabel("X position (pixels)")
        axs[0].set_ylabel("Y position (pixels)")
        axs[0].set_title("Shuttle Trajectory")
        axs[0].legend()
        axs[0].invert_yaxis()  # Invert Y-axis for image coordinate system
        axs[0].grid(True)

        panels = [
            (1, velocity_history, "green", "speed", "speed (pixels)", "Velocity"),
            (
                2,
                angle_history,
                "orange",
                "angle",
                "angle change",
                "Angle Change Over Time",
            ),
            (
                3,
                direction_angles,
                "blue",
                "direction",
                "direction angle",
                "Direction Angle Over Time",
            ),
            (
                4,
                blur_quantity,
                "yellow",
                "blur",
                "blur quantity",
                "Blur Quantity Over Time",
            ),
        ]

        for index, series, color, label, ylabel, title in panels:
            axis = axs[index]
            axis.plot(timestamp_history, series, "--o", color=color, label=label)
            for i, (timestamp_x, value_y) in enumerate(
                zip(timestamp_history, series), start=1
            ):
                axis.text(timestamp_x, value_y, str(i), fontsize=10, color="red")

            axis.set_xlabel("timestamp")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.legend()
            axis.grid(True)

        # mark the hits on the speed panel
        for hit in hit_candidates:
            axs[1].axvline(
                timestamp_history[hit], color="red", linestyle="--", alpha=0.6
            )

        return fig, axs
