from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import TypeAlias

import numpy as np
from loguru import logger
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

Vec3d: TypeAlias = tuple[float, float, float]


@dataclass(frozen=True)
class Edge:
    curve: BRepAdaptor_Curve
    idx: int

    def __repr__(self):
        return f"Edge {self.idx}"

    @property
    def start_param(self) -> float:
        return self.curve.FirstParameter()

    @property
    def end_param(self) -> float:
        return self.curve.LastParameter()

    @property
    def first_p(self) -> Vec3d:
        return self.get_point(self.start_param)

    @property
    def last_p(self) -> Vec3d:
        return self.get_point(self.end_param)

    def sample_points(self, n_points: int) -> np.ndarray:
        """Sample points along an edge, including the first and last points."""

        length = GCPnts_AbscissaPoint.Length_s(
            self.curve, self.start_param, self.end_param
        )
        n_points = 2 + int(length / 3)

        # Generate parameter values
        param_values = np.linspace(self.start_param, self.end_param, n_points)

        # Sample points
        points = []
        for param in param_values:
            pnt = gp_Pnt()
            self.curve.D0(param, pnt)
            points.append(pnt)

        return np.round([(pnt.X(), pnt.Y(), pnt.Z()) for pnt in points], decimals=3)

    def get_point(self, param: float) -> Vec3d:
        p = gp_Pnt()
        self.curve.D0(param, p)
        return round(p.X(), 3), round(p.Y(), 3), round(p.Z(), 3)


def get_edge_graph(edges: list[Edge]) -> dict[Edge, list[Edge]]:
    edge_graph: dict[Edge, list[Edge]] = defaultdict(list)
    points_to_edge: dict[Vec3d, list[int]] = defaultdict(list)
    for idx, edge in enumerate(edges):
        points_to_edge[edge.first_p].append(idx)
        points_to_edge[edge.last_p].append(idx)

    for idx, edge in enumerate(edges):
        edge_graph[edge] = [
            edges[i]
            for i in chain(points_to_edge[edge.first_p], points_to_edge[edge.last_p])
            if i != idx
        ]

    return edge_graph


def get_edges(shape: TopoDS_Shape) -> tuple[list[TopoDS.Face_s], dict[int, list[Edge]]]:
    """Extract edges from a shape."""

    # Iterate over faces
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    edge_idx = face_idx = 0
    face_edge_map = dict()
    while face_explorer.More():
        face = face_explorer.Current()
        face_explorer.Next()

        # Extract edges from the face
        face = TopoDS.Face_s(face)
        faces.append(face)
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        edges = []
        while edge_explorer.More():
            curve = BRepAdaptor_Curve(TopoDS.Edge_s(edge_explorer.Current()))
            edge = Edge(curve, edge_idx)
            edge_idx += 1
            edges.append(edge)
            edge_explorer.Next()
        face_edge_map[face_idx] = edges
        face_idx += 1

        # from src.faces.cone import project_point_to_uv
        # from OCP.BRep import BRep_Tool
        #
        # surface = BRep_Tool.Surface_s(face)
        # project_point_to_uv(surface, gp_Pnt(*points[0]))

        surface = BRepAdaptor_Surface(face)
        logger.debug(f"face: {face_idx} => {surface.GetType()}")
        # if surface.GetType() == ga.GeomAbs_BSplineCurve:
        #     pass

    if not faces:
        raise ValueError("No faces found in the shape.")

    return faces, face_edge_map


def get_edge_loops(edges: list["Edge"]) -> list[list[tuple["Edge", bool]]]:
    graph: dict["Vec3d", list[tuple["Vec3d", "Edge"]]] = defaultdict(list)
    for edge in edges:
        p1 = tuple(round(p, 2) for p in edge.first_p)
        p2 = tuple(round(p, 2) for p in edge.last_p)
        graph[p1].append((p2, edge))
        graph[p2].append((p1, edge))

    if not all(len(neighbors) == 2 for neighbors in graph.values()):
        raise RuntimeError("Edge graph could not be built!")

    edge_loops: list[list[tuple["Edge", bool]]] = []
    visited_edges = set()

    # Process all edges until all are visited
    while len(visited_edges) < len(edges):
        # Find a starting point with an unvisited edge
        start_point = None
        for point in graph:
            for _, edge in graph[point]:
                if edge not in visited_edges:
                    start_point = point
                    break
            if start_point:
                break

        if not start_point:
            break  # No more unvisited edges

        current = start_point
        loop = []

        # Explore the loop starting from the current point
        while True:
            has_next = False
            for pnt, edge in graph[current]:
                if edge not in visited_edges:
                    has_next = True
                    visited_edges.add(edge)
                    # Check edge orientation
                    if np.allclose(pnt, edge.first_p, atol=1e-2):
                        loop.append((edge, True))
                    else:
                        loop.append((edge, False))
                    current = pnt
                    break

            if not has_next:
                break

        if loop:
            edge_loops.append(loop)

    return edge_loops
