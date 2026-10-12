"""
The CVAT skeleton definition generated from the court model.

It is generated so that there is one definition of what a court vertex is. If
the node order here ever stops matching CourtModel.vertices, every annotation
made against it silently describes a different court.
"""

import json
import re

from sokil.court import CourtModel
from sokil.dataset.court import resolve_vertex
from sokil.training.cvat import (
    court_skeleton_label,
    court_skeleton_svg,
    write_court_skeleton_label,
)

LABEL = "court dimensions"


def nodes(svg: str) -> dict[int, tuple[float, float]]:
    found = re.findall(
        r'<circle[^>]*cx="([\d.]+)" cy="([\d.]+)"[^>]*data-node-id="(\d+)"', svg
    )
    return {int(node): (float(x), float(y)) for x, y, node in found}


class TestSvg:
    def test_has_a_node_per_court_vertex(self):
        assert set(nodes(court_skeleton_svg())) == set(
            range(1, len(CourtModel().vertices) + 1)
        )

    def test_has_an_edge_per_court_edge(self):
        assert court_skeleton_svg().count("<line") == len(CourtModel().edges)

    def test_every_edge_joins_nodes_that_exist(self):
        svg = court_skeleton_svg()
        drawn = set(nodes(svg))

        edges = re.findall(r'data-node-from="(\d+)" data-node-to="(\d+)"', svg)
        assert edges
        assert all({int(a), int(b)} <= drawn for a, b in edges)

    def test_stays_inside_the_box_cvat_draws_in(self):
        placed = nodes(court_skeleton_svg()).values()

        assert all(0 <= x <= 100 and 0 <= y <= 100 for x, y in placed)

    def test_keeps_the_court_the_right_way_up(self):
        # Vertex 1 is on one baseline and vertex 30 on the other, so the court's
        # length has to run down the drawing, not across it.
        placed = nodes(court_skeleton_svg())

        assert placed[1][1] > placed[30][1]

    def test_lays_the_nodes_out_as_the_court_is(self):
        # A similarity fit back onto the court model: if the node order were
        # wrong, no single transform would explain all thirty positions.
        import numpy as np

        placed = nodes(court_skeleton_svg())
        model = np.array(CourtModel().vertices, dtype=float)
        drawn = np.array([placed[i + 1] for i in range(len(model))])

        design = np.hstack([drawn, np.ones((len(drawn), 1))])
        coefficients, *_ = np.linalg.lstsq(design, model, rcond=None)
        residual = np.linalg.norm(design @ coefficients - model, axis=1)

        assert residual.max() < 1.0  # centimetres, on a 610x1340 court

    def test_binds_each_node_to_the_sublabel_id_it_is_given(self):
        svg = court_skeleton_svg({n: n + 100 for n in range(1, 31)})

        pairs = re.findall(r'data-node-id="(\d+)" data-label-id="(\d+)"', svg)
        assert pairs
        assert all(int(label) == int(node) + 100 for node, label in pairs)


class TestLabel:
    def test_names_its_nodes_so_the_importer_can_map_them_back(self):
        sublabels = court_skeleton_label(LABEL)[0]["sublabels"]

        assert sorted(resolve_vertex(sub["name"]) for sub in sublabels) == list(
            range(1, len(CourtModel().vertices) + 1)
        )

    def test_the_svg_refers_only_to_sublabels_it_defines(self):
        label = court_skeleton_label(LABEL)[0]

        ids = {sub["id"] for sub in label["sublabels"]}
        refs = {int(m) for m in re.findall(r'data-label-id="(\d+)"', label["svg"])}
        assert refs == ids

    def test_is_written_as_json_cvat_can_take(self, tmp_path):
        path = write_court_skeleton_label(tmp_path / "skeleton.json", LABEL)

        document = json.loads(path.read_text())
        assert document[0]["type"] == "skeleton"
        assert document[0]["name"] == LABEL
        assert len(document[0]["sublabels"]) == len(CourtModel().vertices)
