
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain

from OCP.OCP.STEPControl import STEPControl_Reader
from OCP.TopoDS import TopoDS_Shape, TopoDS
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
import numpy as np
import polyscope as ps
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.gp import gp_Pnt


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
    def first_p(self) -> tuple[float, float, float]:
        return self.get_point(self.start_param)

    @property
    def last_p(self) -> tuple[float, float, float]:
        return self.get_point(self.end_param)

    def sample_points(self, n_points: int) -> np.ndarray:
        """Sample points along an edge, including the first and last points."""

        # Generate parameter values
        param_values = np.linspace(self.start_param, self.end_param, n_points)

        # Sample points
        points = []
        for param in param_values:
            pnt = gp_Pnt()
            self.curve.D0(param, pnt)
            points.append([pnt.X(), pnt.Y(), pnt.Z()])

        return np.round(points, decimals=3)

    def get_point(self, param: float) -> tuple[float, float, float]:
        p = gp_Pnt()
        self.curve.D0(param, p)
        return round(p.X(), 3), round(p.Y(), 3), round(p.Z(), 3)


def get_edge_graph(edges: list[Edge]) -> dict[Edge, list[Edge]]:
    edge_graph: dict[Edge, list[Edge]] = defaultdict(list)
    points_to_edge: dict[tuple[float, float, float], list[int]] = defaultdict(list)
    for idx, edge in enumerate(edges):
        points_to_edge[edge.first_p].append(idx)
        points_to_edge[edge.last_p].append(idx)

    for idx, edge in enumerate(edges):
        edge_graph[edge] = [edges[i] for i in chain(points_to_edge[edge.first_p], points_to_edge[edge.last_p]) if i != idx]

    return edge_graph


def get_edges(shape: TopoDS_Shape) -> dict[int, list[Edge]]:
    """Extract edges from a shape."""

    # Iterate over faces
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []

    while face_explorer.More():
        face = face_explorer.Current()
        faces.append(face)
        face_explorer.Next()

    if not faces:
        raise ValueError("No faces found in the shape.")

    face_edges = dict()
    edge_idx = 0
    for face_idx, face in enumerate(faces):
        # Extract edges from the face
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        edges = []

        while edge_explorer.More():
            curve = BRepAdaptor_Curve(TopoDS.Edge_s(edge_explorer.Current()))
            edge = Edge(curve, edge_idx)
            edge_idx += 1
            edges.append(edge)
            edge_explorer.Next()
        face_edges[face_idx] = edges

    return face_edges


def get_face_edges(face_edge_map: dict[int, list[Edge]], face_idx: int) -> None:

    if face_idx not in face_edge_map:
        raise IndexError(f"Face index {face_idx} not in face edge map.")

    edges = face_edge_map[face_idx]
    edge_graph = get_edge_graph(edges)

    # -------
    # |     |
    # |     |
    # -------

    loops = []
    while edge_graph:
        current = next(iter(edge_graph.keys()))
        # loop_points = [current.sample_points(10)]
        loop = [(current, True)]
        visited = {current}
        while True:
            has_next = False
            for neigh in edge_graph[current]:
                if neigh not in visited and current.last_p in (neigh.first_p, neigh.last_p):
                    if neigh.first_p == current.last_p:
                        loop.append((neigh, True))
                    elif neigh.last_p == current.last_p:
                        loop.append((neigh, False))
                    # else:
                    #     raise ValueError
                    # loop_points.append(current.sample_points(10))
                    has_next = True
                    current = neigh
                    visited.add(current)
                    break
            if not has_next:
                break

        # while True:
        #     has_next = False
        #     for neigh in edge_graph[current]:
        #         if neigh not in visited:
        #             has_next = True
        #             current = neigh
        #             new_points = current.sample_points(10)
        #             if new_points[-1] == loop_points[-1][-1]:
        #                 loop_points.append(new_points)
        #             else:
        #                 loop_points.append(new_points[::-1])
        #             visited.add(current)
        #             break
        #     if not has_next:
        #         break

        loops.append(np.vstack(loop_points))
        edge_graph = {k: v for k, v in edge_graph.items() if k not in visited}

    # visualize
    loop_points = []
    for loop in loops:
        all_points = []
        for edge in loop:
            points = edge.sample_points(10)
            all_points.append(points[:-1])  # last point is first of the next edge
        loop_points.append(np.vstack(all_points))
        break

    ps.init()
    for idx, points in enumerate(loop_points):
        edges = np.array([(i, i+1) for i in range(len(points) - 1)])
        ps.register_curve_network(f"loop_{idx}", points, edges)
    ps.show()


def main():
    reader = STEPControl_Reader()
    step_file = "/home/nico/spanflug/bm_parts/2906/repaired.step"
    reader.ReadFile(step_file)
    reader.TransferRoots()
    shape = reader.OneShape()
    face_edge_map = get_edges(shape)
    print(f"Found {len(face_edge_map)} faces.")
    face_edges = get_face_edges(face_edge_map, 0)


if __name__ == "__main__":
    main()
