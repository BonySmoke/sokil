"""
Hit detection: finding the frames where the shuttle stopped flying freely, and
adjudicating each one against the court.
"""

import pytest

from sokil.frame import Hit
from sokil.hits import (
    HitDetectionConfig,
    HitDetector,
    HitJudge,
    HitResult,
    detect_hits,
    find_contact_frame,
    is_flight_broken,
)

FPS = 30.0


def straight_flight(count: int = 12, step: tuple[int, int] = (10, 6), start=(100, 100)):
    """A shuttle travelling in a straight line at a constant speed."""
    numbers = list(range(count))
    positions = [(start[0] + step[0] * i, start[1] + step[1] * i) for i in range(count)]
    return numbers, positions


def bounce_flight(before: int = 8, after: int = 8):
    """
    A shuttle descending, then reversing sharply: a landing.

    Image y grows downward, so the incoming leg has a positive y step and the
    outgoing leg a negative one. The turn happens at index `before`.
    """
    numbers = list(range(before + after))
    positions = [(100 + 3 * i, 100 + 14 * i) for i in range(before)]
    contact_x, contact_y = positions[-1]
    positions += [
        (contact_x + 3 * (i + 1), contact_y - 14 * (i + 1)) for i in range(after)
    ]
    return numbers, positions


class TestHitDetectionConfig:
    def test_an_explicit_max_jump_wins_over_the_frame_fraction(self):
        config = HitDetectionConfig(max_jump=42.0, max_jump_frame_fraction=0.25)

        assert config.resolved_max_jump(1920) == 42.0

    def test_the_threshold_scales_with_the_frame_width(self):
        config = HitDetectionConfig(max_jump=None, max_jump_frame_fraction=0.25)

        assert config.resolved_max_jump(1920) == pytest.approx(480.0)

    def test_an_unknown_frame_width_disables_the_jump_check(self):
        config = HitDetectionConfig(max_jump=None, max_jump_frame_fraction=0.25)

        assert config.resolved_max_jump(None) is None

    def test_no_fraction_disables_the_jump_check_outright(self):
        config = HitDetectionConfig(max_jump=None, max_jump_frame_fraction=None)

        assert config.resolved_max_jump(1920) is None

    def test_the_nms_window_defaults_to_the_fit_window(self):
        assert HitDetectionConfig(window=4).resolved_nms_window == 4
        assert HitDetectionConfig(window=4, nms_window=1).resolved_nms_window == 1


class TestIsFlightBroken:
    CONFIG = HitDetectionConfig(max_step=3, max_jump=100.0)

    def test_consecutive_nearby_detections_are_one_flight(self):
        assert is_flight_broken([0, 1], [(0, 0), (10, 10)], 1, self.CONFIG) is False

    def test_a_long_gap_with_no_detection_breaks_the_flight(self):
        assert is_flight_broken([0, 9], [(0, 0), (10, 10)], 1, self.CONFIG) is True

    def test_a_teleport_breaks_the_flight(self):
        assert is_flight_broken([0, 1], [(0, 0), (900, 0)], 1, self.CONFIG) is True

    def test_the_jump_is_measured_per_frame_not_per_step(self):
        # 200 px over 3 frames is 66 px/frame, under the 100 px/frame threshold.
        assert is_flight_broken([0, 3], [(0, 0), (200, 0)], 1, self.CONFIG) is False

    def test_without_a_threshold_only_the_gap_matters(self):
        config = HitDetectionConfig(max_step=3, max_jump=None)

        assert is_flight_broken([0, 1], [(0, 0), (9999, 0)], 1, config) is False


class TestFindContactFrame:
    def test_moves_to_the_lowest_point_of_the_flight(self):
        # image y grows downward, so the largest y is the lowest point
        positions = [(0, 10), (0, 20), (0, 35), (0, 25), (0, 15)]

        assert (
            find_contact_frame(positions, 1, search=3, is_contiguous=lambda a, b: True)
            == 2
        )

    def test_stops_at_a_break_in_the_flight(self):
        positions = [(0, 10), (0, 20), (0, 999)]

        # index 2 belongs to a different flight, so it is never considered
        def contiguous(first, second):
            return (first, second) != (1, 2)

        assert find_contact_frame(positions, 1, search=3, is_contiguous=contiguous) == 1

    def test_never_scans_beyond_the_ends_of_the_track(self):
        positions = [(0, 10), (0, 20)]

        assert (
            find_contact_frame(positions, 1, search=5, is_contiguous=lambda a, b: True)
            == 1
        )


class TestDetectHits:
    CONFIG = HitDetectionConfig(max_jump=500.0)

    def test_a_free_flight_has_no_hits(self):
        numbers, positions = straight_flight()

        assert detect_hits(numbers, positions, FPS, self.CONFIG) == []

    def test_a_sharp_reversal_on_the_way_down_is_a_hit(self):
        numbers, positions = bounce_flight()

        hits = detect_hits(numbers, positions, FPS, self.CONFIG)

        assert len(hits) == 1
        # the accepted index is the lowest point, which is the turn itself
        assert hits[0] == 7

    def test_only_the_strongest_frame_of_a_run_survives(self):
        # Several frames either side of the turn straddle it, so all of them
        # look like hits; non-max suppression is what thins them to one. The
        # refractory period is switched off here so it is not what does the
        # thinning.
        numbers, positions = bounce_flight()
        common = {
            "max_jump": 500.0,
            "ground_hits_only": False,
            "refractory_seconds": 0.0,
        }

        unsuppressed = detect_hits(
            numbers, positions, FPS, HitDetectionConfig(nms_window=0, **common)
        )
        suppressed = detect_hits(numbers, positions, FPS, HitDetectionConfig(**common))

        assert len(unsuppressed) > 1
        assert len(suppressed) == 1
        assert suppressed[0] in unsuppressed

    def test_a_shuttle_rising_into_the_turn_is_not_a_ground_hit(self):
        # the mirror image of bounce_flight: it goes UP into the turn, which is
        # the top of a lob, not a landing
        numbers, positions = bounce_flight()
        rising = [(x, -y) for x, y in positions]

        assert detect_hits(numbers, rising, FPS, self.CONFIG) == []

    def test_the_same_turn_is_a_hit_once_ground_hits_only_is_off(self):
        numbers, positions = bounce_flight()
        rising = [(x, -y) for x, y in positions]
        config = HitDetectionConfig(max_jump=500.0, ground_hits_only=False)

        assert len(detect_hits(numbers, rising, FPS, config)) == 1

    def test_a_barely_moving_shuttle_has_no_meaningful_direction(self):
        numbers, positions = bounce_flight()
        crawling = [(x // 20, y // 20) for x, y in positions]

        assert detect_hits(numbers, crawling, FPS, self.CONFIG) == []

    def test_the_refractory_period_collapses_the_bounce_after_a_landing(self):
        # two turns three frames apart: at 30 fps the 0.5 s cooldown covers 15
        # frames, so the second is the bounce, not a new shot
        numbers = list(range(20))
        positions = [(100 + 3 * i, 100 + 14 * i) for i in range(8)]
        for _ in range(3):
            positions.append((positions[-1][0] + 3, positions[-1][1] - 14))
        for _ in range(9):
            positions.append((positions[-1][0] + 3, positions[-1][1] + 14))

        hits = detect_hits(numbers, positions, FPS, self.CONFIG)

        assert len(hits) == 1

    def test_two_turns_far_apart_are_two_hits(self):
        first_numbers, first_positions = bounce_flight(before=8, after=8)
        second_numbers, second_positions = bounce_flight(before=8, after=8)
        offset = 60
        numbers = first_numbers + [n + offset for n in second_numbers]
        positions = first_positions + second_positions

        # the flight is broken between the two rallies, which is what the gap
        # in frame numbers says
        hits = detect_hits(numbers, positions, FPS, self.CONFIG)

        assert len(hits) == 2

    def test_an_empty_track_yields_nothing(self):
        assert detect_hits([], [], FPS, self.CONFIG) == []


class FakeCourt:
    """Stands in for a solved Court in the detector's two gates."""

    def __init__(self, observable=True, inside=True):
        self._observable = observable
        self._inside = inside
        self.judged = []

    def shuttle_inside_max_observable_area(self, position, padding=50):
        return self._observable

    def shuttle_intersects_court(self, position):
        self.judged.append(position)
        return self._inside


def tracks_from(numbers, positions, make_track):
    return [
        make_track(
            number=number, xyxy=(x - 5, y - 5, x + 5, y + 5), measured_cork=(x, y)
        )
        for number, (x, y) in zip(numbers, positions)
    ]


class TestHitDetector:
    def test_marks_the_accepted_track_with_a_hit(self, make_track):
        numbers, positions = bounce_flight()
        tracks = tracks_from(numbers, positions, make_track)

        result = HitDetector().detect(tracks, None, FPS, frame_width=1920)

        assert len(result.hit_indices) == 1
        hit_track = tracks[result.hit_indices[0]]
        assert isinstance(hit_track.hit, Hit)
        assert hit_track.hit.position == hit_track.cork_position

    def test_the_court_area_gate_rejects_a_candidate_outside_the_court(
        self, make_track
    ):
        numbers, positions = bounce_flight()
        tracks = tracks_from(numbers, positions, make_track)

        result = HitDetector().detect(
            tracks, FakeCourt(observable=False), FPS, frame_width=1920
        )

        assert result.hit_indices == []
        # the candidate was found and then rejected, which is what the count is for
        assert result.candidate_count == 1
        assert all(track.hit is None for track in tracks)

    def test_a_rerun_clears_the_hits_of_the_previous_one(self, make_track):
        numbers, positions = bounce_flight()
        tracks = tracks_from(numbers, positions, make_track)
        detector = HitDetector()

        detector.detect(tracks, None, FPS, frame_width=1920)
        detector.detect(tracks, FakeCourt(observable=False), FPS, frame_width=1920)

        assert all(track.hit is None for track in tracks)

    def test_a_per_call_config_overrides_the_detectors_own(self, make_track):
        numbers, positions = bounce_flight()
        rising = [(x, -y) for x, y in positions]
        tracks = tracks_from(numbers, rising, make_track)

        default = HitDetector().detect(tracks, None, FPS, frame_width=1920)
        overridden = HitDetector().detect(
            tracks,
            None,
            FPS,
            config=HitDetectionConfig(ground_hits_only=False),
            frame_width=1920,
        )

        assert default.hit_indices == []
        assert len(overridden.hit_indices) == 1

    def test_trajectories_are_the_smooth_runs_between_hits(self, make_track):
        numbers, positions = bounce_flight(before=8, after=8)
        tracks = tracks_from(numbers, positions, make_track)

        result = HitDetector().detect(tracks, None, FPS, frame_width=1920)

        hit_index = result.hit_indices[0]
        assert len(result.trajectories) == 2
        assert all(hit_index not in trajectory for trajectory in result.trajectories)

    def test_runs_shorter_than_the_minimum_are_not_drawable_trajectories(
        self, make_track
    ):
        numbers, positions = straight_flight(count=4)
        tracks = tracks_from(numbers, positions, make_track)

        result = HitDetector().detect(
            tracks,
            None,
            FPS,
            config=HitDetectionConfig(min_trajectory_len=5),
            frame_width=1920,
        )

        assert result.trajectories == []

    def test_a_gap_in_the_track_splits_the_trajectory(self, make_track):
        numbers = list(range(6)) + list(range(20, 26))
        positions = [(100 + 10 * i, 100 + 5 * i) for i in range(12)]
        tracks = tracks_from(numbers, positions, make_track)

        result = HitDetector().detect(
            tracks,
            None,
            FPS,
            config=HitDetectionConfig(min_trajectory_len=3, max_jump=None),
            frame_width=1920,
        )

        assert len(result.trajectories) == 2

    def test_an_empty_track_list_gives_an_empty_result(self):
        result = HitDetector().detect([], None, FPS, frame_width=1920)

        assert result == HitResult(hit_indices=[], trajectories=[], candidate_count=0)


class TestHitJudge:
    def test_sets_the_verdict_on_every_hit(self, make_track):
        tracks = [make_track(number=i) for i in range(3)]
        tracks[1].hit = Hit(position=(100, 100))
        court = FakeCourt(inside=True)

        judged = HitJudge().judge(tracks, court)

        assert judged == 1
        assert tracks[1].hit.is_in is True
        assert court.judged == [tracks[1].cork_position]

    def test_an_out_verdict_is_recorded_too(self, make_track):
        track = make_track(number=0)
        track.hit = Hit(position=(100, 100))

        HitJudge().judge([track], FakeCourt(inside=False))

        assert track.hit.is_in is False

    def test_nothing_is_judged_without_a_court(self, make_track):
        track = make_track(number=0)
        track.hit = Hit(position=(100, 100))

        assert HitJudge().judge([track], None) == 0
        assert track.hit.is_in is None

    def test_tracks_without_a_hit_are_skipped(self, make_track):
        tracks = [make_track(number=i) for i in range(3)]

        assert HitJudge().judge(tracks, FakeCourt()) == 0

    def test_an_undecidable_hit_leaves_the_verdict_unset(self, make_track):
        track = make_track(number=0)
        track.hit = Hit(position=(100, 100))
        court = FakeCourt()
        court.shuttle_intersects_court = lambda position: None

        assert HitJudge().judge([track], court) == 0
        assert track.hit.is_in is None
