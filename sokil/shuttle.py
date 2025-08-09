import numpy as np


class DirectionTracker:
    """
    Tracks shuttle direction using EMA of last N position differences.
    Reacts to direction changes in 1-2 frames instead of 5+ like Kalman
    """

    def __init__(self, alpha=0.5):
        """
        alpha : EMA smoothing factor
                0.0 = never update (ignore new data)
                1.0 = always use latest (no smoothing)
                0.5 = balanced — reacts in ~2 frames
                0.8 = very reactive — reacts in 1 frame but noisier
        """
        self.alpha = alpha
        self.prev = None
        self.vx_ema = 0.0
        self.vy_ema = 0.0

    def update(self, x, y):
        if self.prev is None:
            self.prev = (x, y)
            return 0.0, 0.0

        raw_vx = x - self.prev[0]
        raw_vy = y - self.prev[1]

        self.vx_ema = self.alpha * raw_vx + (1 - self.alpha) * self.vx_ema
        self.vy_ema = self.alpha * raw_vy + (1 - self.alpha) * self.vy_ema

        self.prev = (x, y)
        return self.vx_ema, self.vy_ema

    def direction(self):
        """Returns unit direction vector (vx, vy)."""
        norm = np.sqrt(self.vx_ema**2 + self.vy_ema**2) + 1e-9
        return self.vx_ema / norm, self.vy_ema / norm

    def velocity(self):
        return self.vx_ema, self.vy_ema


def estimate_cork_position(
    frame_gray,
    previous_grays,
    xyxy,
    velocity,
    minimum_history=3,
    lead_percentile=92.0,
    noise_floor=0.25,
):
    """
    Locate the cork.

    The bounding box is axis aligned, so projecting the flight direction onto
    its border puts the cork in a corner the shuttle does not reach. This works
    from the shuttle's own pixels instead.

    The camera is static, so the median of this window over the preceding frames
    is the court behind the shuttle: it keeps the floor and any painted line,
    and votes the shuttle away because the shuttle is only in the window on the
    current frame. Whatever differs from that median is the shuttle.

    The difference is used as a weight rather than being thresholded into a
    silhouette — a shuttle over a painted line does not threshold cleanly, and
    the cork is only wanted to a pixel or two. Every weighted pixel is projected
    onto the flight direction and the leading slice is averaged, which puts the
    result at the front of the shuttle without depending on a clean outline.

    :param frame_gray: the current frame, single channel.
    :param previous_grays: recent frames, most recent last, same size.
    :param xyxy: the shuttle's bounding box on this frame.
    :param velocity: the direction the shuttle is travelling.
    :returns: the (x, y) cork position, or None when it cannot be measured.
    """
    height, width = frame_gray.shape[:2]
    x1, y1, x2, y2 = (round(value) for value in xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None

    history = [
        previous[y1:y2, x1:x2]
        for previous in previous_grays
        if previous.shape[:2] == (height, width)
    ]
    if len(history) < minimum_history:
        return None

    current = frame_gray[y1:y2, x1:x2].astype(np.float32)
    background = np.median(np.stack(history).astype(np.float32), axis=0)

    weight = np.abs(current - background)
    strongest = float(weight.max())
    if strongest <= 0:
        return None

    # anything this faint is sensor noise, not the shuttle
    weight[weight < noise_floor * strongest] = 0.0
    if not weight.any():
        return None

    velocity_x, velocity_y = velocity
    speed = float(np.hypot(velocity_x, velocity_y))
    if speed <= 0:
        return None

    unit_x, unit_y = velocity_x / speed, velocity_y / speed

    rows, columns = np.nonzero(weight)
    weights = weight[rows, columns]
    along_flight = columns * unit_x + rows * unit_y

    # weighted percentile: how far along the flight the leading slice starts
    order = np.argsort(along_flight)
    share = np.cumsum(weights[order]) / weights.sum()
    cut = float(
        along_flight[order][
            min(int(np.searchsorted(share, lead_percentile / 100.0)), len(order) - 1)
        ]
    )

    leading = along_flight >= cut
    if not leading.any():
        return None

    cork_x = float(np.average(columns[leading], weights=weights[leading]))
    cork_y = float(np.average(rows[leading], weights=weights[leading]))
    return (round(x1 + cork_x), round(y1 + cork_y))
