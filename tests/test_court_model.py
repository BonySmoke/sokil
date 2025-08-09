"""
CourtModel: the rule-book court in centimetres, and what each camera mode of it
is allowed to adjudicate.
"""

import pytest

from sokil.court import CourtCameraMode, CourtModel, CourtModelType


@pytest.fixture
def model() -> CourtModel:
    return CourtModel()


class TestVertices:
    def test_there_are_thirty_of_them(self, model):
        assert len(model.vertices) == 30

    def test_the_corners_span_the_full_court(self, model):
        assert model.vertices[0] == (0, 0)
        assert model.vertices[4] == (model.width, 0)
        assert model.vertices[25] == (model.width, model.length)
        assert model.vertices[29] == (0, model.length)

    def test_the_centre_line_runs_down_the_middle(self, model):
        assert model.vertices[2][0] == model.width / 2

    def test_the_side_corridor_is_inset_by_its_width(self, model):
        assert model.vertices[1] == (model.side_corridor_width, 0)
        assert model.vertices[3] == (model.width - model.side_corridor_width, 0)

    def test_every_edge_names_an_existing_vertex(self, model):
        for start, end in model.edges:
            assert 1 <= start <= 30
            assert 1 <= end <= 30

    def test_no_edge_is_listed_twice(self, model):
        undirected = {tuple(sorted(edge)) for edge in model.edges}

        assert len(undirected) == len(model.edges)


class TestBoundaries:
    def test_the_doubles_court_is_the_full_rectangle(self, model):
        assert model.doubles_boundaries() == [
            (0, 0),
            (model.width, 0),
            (model.width, model.length),
            (0, model.length),
        ]

    def test_the_singles_court_drops_the_side_corridors(self, model):
        corners = model.singles_boundaries()

        assert corners[0] == (model.side_corridor_width, 0)
        assert corners[1] == (model.width - model.side_corridor_width, 0)

    def test_get_boundaries_follows_the_court_type(self):
        doubles = CourtModel(court_model_type=CourtModelType.DOUBLES)
        singles = CourtModel(court_model_type=CourtModelType.SINGLES)

        assert doubles.get_boundaries() == doubles.doubles_boundaries()
        assert singles.get_boundaries() == singles.singles_boundaries()

    def test_a_corridor_view_is_bounded_by_the_vertices_it_can_see(self):
        model = CourtModel(court_camera_mode=CourtCameraMode.LEFT_CORRIDOR)

        boundary = model.get_boundaries()

        assert boundary == [
            (0, 0),
            (model.side_corridor_width, 0),
            (model.side_corridor_width, model.length),
            (0, model.length),
        ]


class TestOutBoundaries:
    def test_a_full_view_adjudicates_all_four_lines(self, model):
        assert model.out_boundaries() == [
            ("x", 0, -1),
            ("x", model.width, 1),
            ("y", 0, -1),
            ("y", model.length, 1),
        ]

    def test_singles_watches_the_inner_sidelines(self):
        model = CourtModel(court_model_type=CourtModelType.SINGLES)

        left, right, *_ = model.out_boundaries()

        assert left == ("x", model.side_corridor_width, -1)
        assert right == ("x", model.width - model.side_corridor_width, 1)

    @pytest.mark.parametrize(
        ("mode", "unwatched"),
        [
            (CourtCameraMode.LEFT_CORRIDOR, ("x", 1)),
            (CourtCameraMode.RIGHT_CORRIDOR, ("x", -1)),
            (CourtCameraMode.BOTTOM_CORRIDOR, ("y", 1)),
            (CourtCameraMode.TOP_CORRIDOR, ("y", -1)),
        ],
    )
    def test_a_corridor_view_leaves_the_far_side_open(self, mode, unwatched):
        model = CourtModel(court_camera_mode=mode)

        watched = {(axis, sign) for axis, _, sign in model.out_boundaries()}

        assert len(model.out_boundaries()) == 3
        assert unwatched not in watched


class TestCorridorSubsets:
    @pytest.mark.parametrize(
        "vertices_of",
        [
            "left_corridor_vertices",
            "right_corridor_vertices",
            "bottom_corridor_vertices",
            "top_corridor_vertices",
        ],
    )
    def test_every_corridor_vertex_belongs_to_the_full_model(self, model, vertices_of):
        subset = getattr(model, vertices_of)()

        assert subset
        assert all(vertex in model.vertices for vertex in subset)

    def test_the_left_corridor_holds_only_the_leftmost_two_columns(self, model):
        xs = {x for x, _ in model.left_corridor_vertices()}

        assert xs == {0, model.side_corridor_width}

    def test_the_bottom_corridor_holds_only_the_near_two_rows(self, model):
        ys = {y for _, y in model.bottom_corridor_vertices()}

        assert ys == {0, model.back_corridor_width}

    def test_a_corridors_edges_stay_inside_its_own_vertices(self, model):
        vertices = set(map(tuple, model.left_corridor_vertices()))

        for start, end in model.left_corridor_edges():
            assert tuple(model.vertices[start - 1]) in vertices
            assert tuple(model.vertices[end - 1]) in vertices

    def test_get_court_model_returns_the_subset_for_the_mode(self):
        model = CourtModel(court_camera_mode=CourtCameraMode.TOP_CORRIDOR)

        vertices, edges = model.get_court_model()

        assert vertices == model.top_corridor_vertices()
        assert edges == model.top_corridor_edges()

    def test_the_full_mode_returns_everything(self, model):
        vertices, edges = model.get_court_model()

        assert vertices == model.vertices
        assert edges == model.edges


class TestLineCoords:
    def test_the_full_court_has_five_lines_across_it(self, model):
        # the two sidelines, the two singles sidelines and the centre line
        assert model.unique_line_coords("x") == [
            0,
            model.side_corridor_width,
            model.width / 2,
            model.width - model.side_corridor_width,
            model.width,
        ]

    def test_the_coords_are_ascending_and_unique(self, model):
        coords = model.unique_line_coords("y")

        assert coords == sorted(set(coords))

    def test_a_corridor_view_sees_fewer_lines(self):
        corridor = CourtModel(court_camera_mode=CourtCameraMode.LEFT_CORRIDOR)

        assert corridor.unique_line_coords("x") == [0, corridor.side_corridor_width]


class TestStripeCentres:
    def test_the_centre_line_straddles_the_service_courts(self, model):
        assert model.stripe_center_offset("x", model.width / 2) == 0.0

    def test_a_sideline_stripe_lies_inside_the_court(self, model):
        half = model.line_width / 2

        assert model.stripe_center_offset("x", 0) == half
        assert model.stripe_center_offset("x", model.width) == -half

    def test_a_back_line_stripe_lies_inside_the_court(self, model):
        half = model.line_width / 2

        assert model.stripe_center_offset("y", 0.0) == half
        assert model.stripe_center_offset("y", model.length) == -half

    def test_a_short_service_line_stripe_lies_away_from_the_net(self, model):
        short_service = model.back_corridor_width + model.net_corridor_width

        # near half: the stripe is on the far side of the coord, toward the net
        assert model.stripe_center_offset("y", short_service) == -model.line_width / 2

    def test_the_stripe_centres_stay_within_half_a_line_of_the_model_coords(
        self, model
    ):
        boundaries = model.unique_line_coords("x")
        centres = model.stripe_center_coords("x")

        assert len(centres) == len(boundaries)
        for boundary, centre in zip(boundaries, centres):
            assert abs(centre - boundary) <= model.line_width / 2

    def test_the_stripe_centres_are_ascending(self, model):
        centres = model.stripe_center_coords("y")

        assert centres == sorted(centres)
