from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import TypeAlias

import numpy as np
from loguru import logger
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.GeomAbs import GeomAbs_CurveType
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
        return f"Edge {self.idx}, {self.first_p}, {self.last_p}"

    def __eq__(self, other: "Edge") -> bool:
        return np.allclose(self.first_p, other.first_p, atol=1e-2) and np.allclose(
            self.last_p, other.last_p, atol=1e-2
        )

    def __hash__(self) -> int:
        # Round the points to match the precision used in __eq__
        first_p_rounded = tuple(round(p, 2) for p in self.first_p)
        last_p_rounded = tuple(round(p, 2) for p in self.last_p)

        # Create a hashable representation of the points
        hashable_repr = (first_p_rounded, last_p_rounded)

        # Return the hash of this representation
        return hash(hashable_repr)

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

    def sample_points_old(self, n_points: int) -> np.ndarray:
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

    def sample_points(
        self, fixed_distance: float = 1.0, fixed_angular_distance: float = 10.0
    ) -> np.ndarray:
        """Sample points along an edge, including the first and last points.

        Args:
            fixed_distance: The fixed distance between points on a line edge.
            fixed_angular_distance: The fixed angular distance (in degrees) between points on a circle edge.

        Returns:
            np.ndarray: An array of sampled points.
        """

        # Determine the type of the curve
        curve_type = self.curve.GetType()

        if curve_type == GeomAbs_CurveType.GeomAbs_Circle:
            # Circle edge: sample points at fixed angular distance
            start_angle = self.start_param
            end_angle = self.end_param
            angular_range = end_angle - start_angle

            # Convert angular distance to radians
            fixed_angular_distance_rad = np.deg2rad(fixed_angular_distance)

            # Calculate the number of points
            n_points = int(angular_range / fixed_angular_distance_rad) + 1

            # Generate parameter values
            param_values = np.linspace(start_angle, end_angle, n_points)

        else:
            # if curve_type == GeomAbs_CurveType.GeomAbs_Line:
            # Line edge: sample points at fixed distance
            length = GCPnts_AbscissaPoint.Length_s(
                self.curve, self.start_param, self.end_param
            )
            n_points = int(length / fixed_distance) + 1

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


def get_edges(shape: TopoDS_Shape) -> tuple[list[TopoDS.Face_s], dict[int, set[Edge]]]:
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
        edges = set()
        while edge_explorer.More():
            curve = BRepAdaptor_Curve(TopoDS.Edge_s(edge_explorer.Current()))
            edge = Edge(curve, edge_idx)
            edge_idx += 1
            edges.add(edge)
            edge_explorer.Next()
        face_edge_map[face_idx] = edges
        face_idx += 1
        surface = BRepAdaptor_Surface(face)
        logger.debug(f"face: {face_idx} => {surface.GetType()}")

    if not faces:
        raise ValueError("No faces found in the shape.")

    return faces, face_edge_map


def get_edge_loops(edges: set[Edge]) -> list[list[tuple[Edge, bool]]]:
    graph: dict["Vec3d", list[tuple["Vec3d", Edge]]] = defaultdict(list)
    for edge in edges:
        p1 = tuple(round(p, 2) for p in edge.first_p)
        p2 = tuple(round(p, 2) for p in edge.last_p)
        graph[p1].append((p2, edge))
        graph[p2].append((p1, edge))

    if not all(len(neighbors) > 1 for neighbors in graph.values()):
        raise RuntimeError("Edge graph could not be built!")

    edge_loops: list[list[tuple[Edge, bool]]] = []
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


def get_edge_loops_new(edges: set[Edge]) -> list[list[tuple[Edge, bool]]]:
    # Build the graph
    graph: dict[tuple, list[tuple[tuple, Edge]]] = defaultdict(list)
    for edge in edges:
        p1 = tuple(round(p, 2) for p in edge.first_p)
        p2 = tuple(round(p, 2) for p in edge.last_p)
        graph[p1].append((p2, edge))
        graph[p2].append((p1, edge))

    if not graph:
        raise RuntimeError("Edge graph could not be built!")

    edge_loops: list[list[tuple[Edge, bool]]] = []
    visited_edges = set()

    def trace_path(start_vertex, current_vertex, path=None):
        if path is None:
            path = []

        # Check all connecting edges
        for next_vertex, edge in graph[current_vertex]:
            # Skip visited edges
            if edge in visited_edges:
                continue

            # Mark this edge as visited
            visited_edges.add(edge)

            # Determine direction based on current vertex
            direction = not np.allclose(current_vertex, edge.first_p, atol=1e-2)

            # Add edge to current path
            path.append((edge, direction))

            # If next vertex is our starting point, we've found a loop
            if np.allclose(next_vertex, start_vertex, atol=1e-2):
                return path

            # Otherwise, continue tracing
            result = trace_path(start_vertex, next_vertex, path)
            if result:
                return result

            # If we reach here, this path didn't lead to a loop
            # Remove the edge from our path and try another branch
            path.pop()

        return None

    # Find all loops
    for vertex in graph:
        while True:
            # Try to find a loop starting from this vertex
            path = trace_path(vertex, vertex)

            if not path:
                break  # No more loops from this vertex

            edge_loops.append(path)

    return edge_loops
