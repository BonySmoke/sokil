"""
Hit detection and in/out judging.
"""

import logging
from dataclasses import dataclass, field, replace

import numpy as np

from .court import Court
from .frame import Hit, ShuttleTrack
from .util import angle_between, fit_velocity

logger = logging.getLogger(__name__)


@dataclass
class HitDetectionConfig:
    """
    Knobs for the trajectory hit detector (detect_hits). Pass an instance to
    HitDetector (or to a single detect call) and sweep the fields in a grid
    search. See HitDetector.detect for the algorithm rationale.
    """

    window: int = 3  # frames fit on each side of a candidate
    angle_change_threshold: float = 110.0  # degrees between pre/post velocity
    speed_drop_threshold: float = 0.85  # post speed < 45% of pre speed
    min_incoming_speed: float = 4.0  # px/frame; below this the direction is noise
    max_step: int = 3  # max frame gap bridged between consecutive detections
    # px/frame above which two consecutive detections cannot be the same flight:
    # the shuttle teleported, so the run is cut. Guards against merged clips
    # (across a cut the shuttle jumps to wherever it is in the next clip) and
    # against the detector latching onto another object. Left as None it is
    # derived from the frame width, see max_jump_frame_fraction.
    max_jump: float | None = None
    # the fallback for max_jump, as a share of the frame width per frame.
    # Labeled rallies peak at ~0.13 of the frame width per frame, so this is
    # about double the fastest real flight. None disables the jump check.
    max_jump_frame_fraction: float | None = 0.25

    def resolved_max_jump(self, frame_width: float | None) -> float | None:
        """The jump threshold in px, given the frame width (None if unknown)."""
        if self.max_jump is not None:
            return self.max_jump
        if self.max_jump_frame_fraction is None or not frame_width:
            return None
        return self.max_jump_frame_fraction * frame_width

    nms_window: int | None = None  # non-max-suppression half-width (None -> window)
    # Only accept hits reached by a descending shuttle
    ground_hits_only: bool = True
    refractory_seconds: float = 0.5  # cooldown collapsing the post-landing bounce
    min_trajectory_len: int = 5  # shortest smooth run kept as a drawable trajectory

    @property
    def resolved_nms_window(self) -> int:
        return self.window if self.nms_window is None else self.nms_window


@dataclass
class HitResult:
    """
    What HitDetector.detect produces.

    :param hit_indices: indices into the tracks list that were accepted as hits
        (the accepted tracks also carry a Hit on their `hit` field).
    :param trajectories: the smooth runs of track indices between hits.
    :param candidate_count: candidates before the court-area gate, for logging.
    """

    hit_indices: list[int] = field(default_factory=list)
    trajectories: list[list[int]] = field(default_factory=list)
    candidate_count: int = 0


def is_flight_broken(numbers, centers, i, config: HitDetectionConfig) -> bool:
    """
    Whether the shuttle cannot have flown from detection i-1 to detection i.

    Two things break a flight: a long gap with no detection, and a step too
    large for the shuttle to have covered.
    """
    step = numbers[i] - numbers[i - 1]
    if step > config.max_step:
        return True

    if config.max_jump is None:
        return False

    (previous_x, previous_y), (x, y) = centers[i - 1], centers[i]
    return float(np.hypot(x - previous_x, y - previous_y)) / max(step, 1) > (
        config.max_jump
    )


def find_contact_frame(shuttle_positions, index, search, is_contiguous):
    """
    The frame where the shuttle is at its lowest, near a detected hit.

    The detector fires one frame early by construction, so the accepted index is
    normally the last airborne frame. The shuttle is lowest at the moment it
    touches the floor, and higher again once it bounces, so the largest image y
    nearby is the contact. The search stops at a break in the flight so it
    cannot wander into a different one.

    :param is_contiguous: called with (a, b) for neighbouring indices; False
        when the shuttle cannot have flown between them.
    """
    lowest = index
    for step in (1, -1):
        scan = index
        while abs(scan + step - index) <= search:
            following = scan + step
            if not 0 <= following < len(shuttle_positions):
                break
            first, second = min(scan, following), max(scan, following)
            if not is_contiguous(first, second):
                break
            if shuttle_positions[following][1] > shuttle_positions[lowest][1]:
                lowest = following
            scan = following
    return lowest


def detect_hits(
    shuttle_frame_numbers, shuttle_positions, fps, config: HitDetectionConfig
):
    """
    Find the frames where the shuttle hit something.

    The idea in one sentence: a shuttle in free flight keeps going roughly the
    same way, so the frames where it suddenly changes direction or loses most of
    its speed are the frames where it hit the floor, a racket, or the net.

    For every frame the shuttle was seen on, this:

    1. looks at the few frames BEFORE it and works out which way the shuttle was
       travelling, and how fast (the "incoming" direction and speed);
    2. does the same for the few frames AFTER it (the "outgoing" direction);
    3. measures how far the direction turned between the two, and how much speed
       was lost;
    4. turns those two numbers into one impact_score, and marks the frame as a
       possible hit if either the turn or the speed loss is large enough.

    A single real impact makes several neighbouring frames look suspicious,
    because steps 1 and 2 both peek a few frames either side of the impact. So
    the last two steps thin the results down: keep only the strongest frame in
    each little group, then ignore anything that follows too soon after an
    accepted hit (that is the bounce, not a new rally shot).

    :param shuttle_frame_numbers: frame numbers where the shuttle is visible,
        aligned with shuttle_positions.
    :param shuttle_positions: (x, y) shuttle pixel positions.
    :param fps: frames per second, used to turn refractory_seconds into frames.
    :param config: the thresholds, see HitDetectionConfig.
    :returns: indices into the input lists that were accepted as hits.
    """
    detection_count = len(shuttle_frame_numbers)
    peak_window = config.resolved_nms_window

    impact_scores = [0.0] * detection_count
    looks_like_a_hit = [False] * detection_count

    for index in range(detection_count):
        # Walk backwards collecting the frames just before this one, stopping if
        # the shuttle was missing for too long or jumped impossibly far — those
        # frames belong to a different flight and would corrupt the direction.
        frames_before = [shuttle_frame_numbers[index]]
        positions_before = [shuttle_positions[index]]
        scan = index - 1
        while (
            scan >= 0
            and len(frames_before) <= config.window
            and not is_flight_broken(
                shuttle_frame_numbers, shuttle_positions, scan + 1, config
            )
        ):
            frames_before.insert(0, shuttle_frame_numbers[scan])
            positions_before.insert(0, shuttle_positions[scan])
            scan -= 1

        # one point cannot show a direction
        if len(frames_before) < 2:
            continue

        incoming_velocity = fit_velocity(frames_before, positions_before)
        if incoming_velocity is None:
            continue

        incoming_speed = float(np.linalg.norm(incoming_velocity))
        # a barely-moving shuttle has no meaningful direction, only noise
        if incoming_speed < config.min_incoming_speed:
            continue

        # A floor hit is always reached on the way DOWN. Image y grows downward,
        # so a descending shuttle has a positive y velocity; anything else is
        # the top of a lob, not a landing.
        if config.ground_hits_only and incoming_velocity[1] <= 0:
            continue

        # the same walk forwards, for the direction the shuttle leaves in
        frames_after = [shuttle_frame_numbers[index]]
        positions_after = [shuttle_positions[index]]
        scan = index + 1
        while (
            scan < detection_count
            and len(frames_after) <= config.window
            and not is_flight_broken(
                shuttle_frame_numbers, shuttle_positions, scan, config
            )
        ):
            frames_after.append(shuttle_frame_numbers[scan])
            positions_after.append(shuttle_positions[scan])
            scan += 1

        if len(frames_after) < 2:
            continue

        outgoing_velocity = fit_velocity(frames_after, positions_after)
        if outgoing_velocity is None:
            continue
        outgoing_speed = float(np.linalg.norm(outgoing_velocity))

        # the two numbers that describe an impact
        direction_change_degrees = angle_between(incoming_velocity, outgoing_velocity)
        fraction_of_speed_lost = max(
            0.0, (incoming_speed - outgoing_speed) / incoming_speed
        )

        # One number combining both, used to rank neighbouring frames against
        # each other. The 90 is a unit conversion, not a tuning knob: it puts
        # "lost all its speed" on the same footing as "turned 90 degrees".
        impact_scores[index] = direction_change_degrees + 90.0 * fraction_of_speed_lost
        looks_like_a_hit[index] = (
            direction_change_degrees >= config.angle_change_threshold
            or fraction_of_speed_lost >= config.speed_drop_threshold
        )

    # One impact makes a small run of neighbouring frames look like hits, since
    # the before/after windows either side of it all straddle the same turn.
    # Keep only the strongest frame of each run — that frame is the best guess
    # at the moment of impact.
    #
    # Only frames that are themselves possible hits are compared. A frame can
    # score well without passing either threshold, and letting such a frame win
    # would leave the run with no hit at all; this step is meant to thin a run,
    # never to empty it.
    strongest_in_window = []
    for index in range(detection_count):
        if not looks_like_a_hit[index]:
            continue
        window_start = max(0, index - peak_window)
        window_end = min(detection_count, index + peak_window + 1)
        nearby_scores = [
            impact_scores[other]
            for other in range(window_start, window_end)
            if looks_like_a_hit[other]
        ]
        if impact_scores[index] >= max(nearby_scores):
            strongest_in_window.append(index)

    # Take the surviving hits in time order, ignoring any that follow too soon
    # after one already accepted: the shuttle bouncing after it lands is not a
    # second hit.
    minimum_frames_between_hits = config.refractory_seconds * fps if fps else 0
    accepted_indices = []
    last_accepted_frame = None
    for index in strongest_in_window:
        too_soon = (
            last_accepted_frame is not None
            and shuttle_frame_numbers[index] - last_accepted_frame
            < minimum_frames_between_hits
        )
        if too_soon:
            continue

        if config.ground_hits_only:
            index = find_contact_frame(
                shuttle_positions,
                index,
                peak_window,
                lambda first, second: (
                    not is_flight_broken(
                        shuttle_frame_numbers, shuttle_positions, second, config
                    )
                    if second == first + 1
                    else False
                ),
            )

        accepted_indices.append(index)
        last_accepted_frame = shuttle_frame_numbers[index]

    return accepted_indices


class HitDetector:
    """
    Detects hits as discontinuities in the shuttle trajectory.
    """

    def __init__(self, config: HitDetectionConfig | None = None):
        self.config = config or HitDetectionConfig()

    def detect(
        self,
        tracks: list[ShuttleTrack],
        court: Court | None,
        fps: float,
        config: HitDetectionConfig | None = None,
        frame_width: float | None = None,
    ) -> HitResult:
        """
        :param tracks: the shuttle tracks, in decode order.
        :param court: the session court, or None when none was solved (the
            court-area gate is then skipped).
        :param fps: frames per second, for the refractory period.
        :param config: overrides self.config for this call (lets a grid search
            sweep params without rebuilding the detector).
        :param frame_width: video width in px; the teleport threshold is scaled
            to it unless the config sets max_jump outright.
        """
        config = config or self.config
        # freeze the frame-relative knobs into px, so everything downstream
        # (including the pure detector) sees one plain threshold
        config = replace(config, max_jump=config.resolved_max_jump(frame_width))

        for shuttle_track in tracks:
            shuttle_track.hit = None  # reset across reruns

        numbers = [shuttle_track.number for shuttle_track in tracks]
        cork_positions = [shuttle_track.cork_position for shuttle_track in tracks]

        candidates = detect_hits(numbers, cork_positions, fps, config)

        hit_indices = []
        for i in candidates:
            shuttle_track: ShuttleTrack = tracks[i]
            # court-area gate: reject candidates outside the observable court
            # (skipped when no reliable court was found)
            if (
                court is not None
                and court.shuttle_inside_max_observable_area(
                    shuttle_track.cork_position
                )
                is False
            ):
                continue
            shuttle_track.hit = Hit(position=shuttle_track.cork_position)
            hit_indices.append(i)

        trajectories = self._build_trajectories(
            numbers, cork_positions, hit_indices, config
        )

        logger.info(
            "Hit detection done: %d hits accepted (%d candidates, %d rejected by "
            "the court-area gate), %d trajectories",
            len(hit_indices),
            len(candidates),
            len(candidates) - len(hit_indices),
            len(trajectories),
        )
        logger.debug("Hit frames: %s", [tracks[i].number for i in hit_indices])

        return HitResult(
            hit_indices=hit_indices,
            trajectories=trajectories,
            candidate_count=len(candidates),
        )

    @staticmethod
    def _build_trajectories(
        numbers: list[int], shuttle_positions: tuple[int, int], hit_indices, config
    ):
        """
        Smooth flights of the shuttle between hits
        """
        trajectories = []
        hit_set = set(hit_indices)
        run = []

        def close(run):
            if len(run) >= config.min_trajectory_len:
                trajectories.append(run)

        for i in range(len(numbers)):
            if i in hit_set:
                close(run)
                run = []
                continue

            if run and is_flight_broken(numbers, shuttle_positions, i, config):
                close(run)
                run = []

            run.append(i)

        close(run)
        return trajectories


class HitJudge:
    """Adjudicates every detected hit as IN or OUT against the session court."""

    def judge(self, tracks: list[ShuttleTrack], court: Court | None) -> int:
        """
        Set `hit.is_in` on every track carrying a hit.

        :returns: how many hits were judged (0 without a court).
        """
        if court is None:
            logger.warning("No reliable court homography; cannot judge hit positions")
            return 0

        judged = 0
        for shuttle_track in tracks:
            if not shuttle_track.hit:
                continue

            is_inside = court.shuttle_intersects_court(shuttle_track.cork_position)
            if is_inside is not None:
                shuttle_track.hit.is_in = is_inside
                judged += 1
                logger.debug(
                    "Hit at frame %d: %s",
                    shuttle_track.number,
                    "IN" if is_inside else "OUT",
                )

        logger.info("In/out judged for %d hits", judged)
        return judged
