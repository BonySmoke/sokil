"""
ShuttleTrack — one detection and everything derived from it — and the frames
drawn from it.
"""

import cv2
import numpy as np
import pytest

from sokil.frame import (
    MIN_MOTION_SPEED,
    Frame,
    Hit,
    ShuttleFrame,
    ShuttleTrack,
    tracks_by_number,
)


class TestBoundingBox:
    def test_the_centre_is_the_middle_of_the_box(self, make_track):
        assert make_track(xyxy=(10, 20, 30, 60)).bbox_center == (20, 40)

    def test_the_width_is_the_horizontal_span(self, make_track):
        assert make_track(xyxy=(10, 20, 30, 60)).bbox_width == 20

    def test_the_height_is_the_vertical_span(self, make_track):
        assert make_track(xyxy=(10, 20, 30, 60)).bbox_height == 40

    def test_the_height_does_not_depend_on_where_the_box_sits(self, make_track):
        # the same box, moved across the frame: only its size decides the height
        assert (
            make_track(xyxy=(500, 20, 520, 60)).bbox_height
            == make_track(xyxy=(10, 20, 30, 60)).bbox_height
        )

    def test_the_contour_walks_the_box_corners(self, make_track):
        contour = make_track(xyxy=(10, 20, 30, 60)).contour_from_bbox()

        assert contour.tolist() == [[10, 20], [30, 20], [30, 60], [10, 60]]


class TestMotion:
    def test_a_fast_shuttle_is_moving(self, make_track):
        assert make_track(velocity=(10.0, 0.0)).is_moving

    def test_a_shuttle_below_the_noise_floor_is_not(self, make_track):
        assert not make_track(velocity=(MIN_MOTION_SPEED / 2, 0.0)).is_moving

    def test_the_direction_arrow_starts_at_the_box_centre(self, make_track):
        track = make_track(xyxy=(10, 20, 30, 60))

        assert track.direction_start == track.bbox_center

    def test_the_direction_arrow_points_along_the_velocity(self, make_track):
        track = make_track(xyxy=(0, 0, 10, 10), velocity=(5.0, 0.0))

        end_x, end_y = track.direction_end

        assert end_x > track.bbox_center[0]
        assert end_y == track.bbox_center[1]

    def test_the_arrow_is_long_enough_to_cross_any_contour(self, make_track):
        track = make_track(xyxy=(0, 0, 10, 10), velocity=(1.0, 0.0))

        assert track.direction_end[0] - track.bbox_center[0] == pytest.approx(
            1000, abs=1
        )


class TestCorkPosition:
    def test_a_measured_cork_is_used_as_it_is(self, make_track):
        track = make_track(velocity=(20.0, 0.0), measured_cork=(123, 456))

        assert track.cork_position == (123, 456)

    def test_a_stationary_shuttle_falls_back_to_the_box_centre(self, make_track):
        # its heading is noise, so projecting along it would make the cork jump
        track = make_track(
            xyxy=(10, 20, 30, 60), velocity=(0.1, 0.1), measured_cork=(123, 456)
        )

        assert track.cork_position == track.bbox_center

    def test_without_a_measurement_the_cork_is_projected_onto_the_box(self, make_track):
        track = make_track(xyxy=(0, 0, 20, 20), velocity=(10.0, 0.0))

        # travelling right, so the cork is on the right edge, level with the centre
        assert track.cork_position == (20, 10)

    def test_a_measured_contour_is_preferred_over_the_box(self, make_track):
        # a contour that stops short of the bounding box's right edge
        contour = np.array([[0, 0], [14, 0], [14, 20], [0, 20]], dtype=np.int32)
        track = make_track(
            xyxy=(0, 0, 20, 20), velocity=(10.0, 0.0), shuttle_contour=contour
        )

        # the box would put the cork at x=20; the measured outline at x=14
        assert track.cork_position == (14, 10)


class TestTracksByNumber:
    def test_indexes_the_tracks_by_frame_number(self, make_track):
        tracks = [make_track(number=n) for n in (3, 9, 12)]

        index = tracks_by_number(tracks)

        assert sorted(index) == [3, 9, 12]
        assert index[9] is tracks[1]

    def test_an_empty_track_list_indexes_to_nothing(self):
        assert tracks_by_number([]) == {}


class TestFrameDrawing:
    def test_drawing_leaves_the_original_frame_alone(self, frame):
        drawn = Frame(frame=frame, number=7).draw(show_frame_number=True)

        assert drawn is not frame
        assert not np.array_equal(drawn, frame)

    def test_without_the_frame_number_nothing_is_drawn(self, frame):
        drawn = Frame(frame=frame, number=7).draw()

        assert np.array_equal(drawn, frame)

    def test_the_repr_names_the_frame_number(self, frame):
        assert repr(Frame(frame=frame, number=7)) == "Frame 7"


class TestShuttleFrameDrawing:
    def build(self, frame, xyxy=(60, 40, 80, 60), **kwargs):
        track = ShuttleTrack(
            number=3,
            xyxy=xyxy,
            speed=12.5,
            angle_change=4.0,
            velocity=(10.0, 2.0),
            timestamp=0.1,
            blur=30.0,
            **kwargs,
        )
        return ShuttleFrame(frame=frame, number=3, track=track)

    def test_the_detection_box_is_drawn(self, frame):
        drawn = self.build(frame).draw()

        assert drawn.shape == frame.shape
        assert not np.array_equal(drawn, frame)

    def test_every_overlay_switched_on_still_produces_a_frame(self, frame):
        shuttle_frame = self.build(frame)
        shuttle_frame.track.track_color = (10, 200, 10)
        shuttle_frame.track.track_points = [(60, 40), (70, 50)]

        drawn = shuttle_frame.draw(
            show_speed=True,
            show_angle_change=True,
            show_direction=True,
            show_track=True,
            show_frame_number=True,
        )

        assert drawn.shape == frame.shape

    def test_an_in_hit_and_an_out_hit_are_drawn_differently(self, frame):
        inside = self.build(frame)
        inside.track.hit = Hit(position=(70, 50), is_in=True)
        outside = self.build(frame)
        outside.track.hit = Hit(position=(70, 50), is_in=False)

        assert not np.array_equal(inside.draw(), outside.draw())

    def test_the_court_outline_is_drawn_when_one_is_supplied(self, frame):
        without = self.build(frame)
        contour = np.array(
            [(i % 6 * 20, i // 6 * 20) for i in range(30)], dtype=np.int32
        )
        with_court = ShuttleFrame(
            frame=frame, number=3, track=without.track, court_contour=contour
        )

        assert not np.array_equal(without.draw(), with_court.draw())

    def test_a_hit_can_be_zoomed_into(self, frame):
        # the default box is off the diagonal, so this also covers the zoom
        # crop being non-empty for a box whose x1 != y1
        shuttle_frame = self.build(frame)
        shuttle_frame.track.hit = Hit(position=(70, 50), is_in=True)

        drawn = shuttle_frame.draw(zoom_in_on_hit=True)

        # the zoom crops and rescales, so the frame size is preserved
        assert drawn.shape == frame.shape

    def test_the_court_rectangle_overlay_is_optional(self, frame):
        rectangle = np.array([[10, 10], [150, 10], [150, 110], [10, 110]], np.int32)
        shuttle_frame = ShuttleFrame(
            frame=frame,
            number=3,
            track=self.build(frame).track,
            court_rectangle=rectangle,
        )

        assert np.array_equal(
            shuttle_frame.draw(show_court_rectangle=False),
            self.build(frame).draw(),
        )
        assert not np.array_equal(
            shuttle_frame.draw(show_court_rectangle=True),
            self.build(frame).draw(),
        )


class TestCv2Contract:
    def test_the_contour_shape_is_what_opencv_expects(self, make_track):
        # a guard on the dtype: cv2 rejects a float contour outright
        contour = make_track().contour_from_bbox()

        assert cv2.contourArea(contour) > 0
