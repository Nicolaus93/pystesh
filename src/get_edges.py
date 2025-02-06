from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain

from OCP.OCP.IFSelect import IFSelect_ReturnStatus
from OCP.OCP.STEPControl import STEPControl_Reader
from OCP.TopoDS import TopoDS_Shape, TopoDS
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
import numpy as np
import polyscope as ps
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.gp import gp_Pnt
from typing import TypeAlias
import numpy.typing as npt

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

        # Generate parameter values
        param_values = np.linspace(self.start_param, self.end_param, n_points)

        # Sample points
        points = []
        for param in param_values:
            pnt = gp_Pnt()
            self.curve.D0(param, pnt)
            points.append([pnt.X(), pnt.Y(), pnt.Z()])

        return np.round(points, decimals=3)

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


def get_edges(shape: TopoDS_Shape) -> dict[int, list[Edge]]:
    """Extract edges from a shape."""

    # Iterate over faces
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    edge_idx = face_idx = 0
    face_edges = dict()
    while face_explorer.More():
        face = face_explorer.Current()
        faces.append(face)
        face_explorer.Next()

        # Extract edges from the face
        face = TopoDS.Face_s(face)
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        edges = []
        while edge_explorer.More():
            curve = BRepAdaptor_Curve(TopoDS.Edge_s(edge_explorer.Current()))
            edge = Edge(curve, edge_idx)
            edge_idx += 1
            edges.append(edge)
            edge_explorer.Next()
        face_edges[face_idx] = edges
        face_idx += 1

        surface = BRepAdaptor_Surface(face)
        print(face_idx, surface.GetType())
        # if surface.GetType() == ga.GeomAbs_BSplineCurve:
        #     pass

    if not faces:
        raise ValueError("No faces found in the shape.")

    return face_edges


def get_edge_loops(edges: list[Edge]) -> list[list[tuple[Edge, bool]]]:
    graph: dict[Vec3d, list[tuple[Vec3d, Edge]]] = defaultdict(list)
    for idx, edge in enumerate(edges):
        graph[edge.first_p].append((edge.last_p, edge))
        graph[edge.last_p].append((edge.first_p, edge))

    # -------
    # |     |
    # |     |
    # -------

    edge_loops: list[list[tuple[Edge, bool]]] = []
    current = next(iter(graph.keys()))
    loop = []
    visited = {current}
    while True:
        has_next = False
        neighs = graph[current]
        for pnt, edge in neighs:
            if pnt not in visited:
                has_next = True
                current = pnt
                visited.add(current)
                # check edge orientation
                if current == edge.first_p:
                    loop.append((edge, True))
                else:
                    loop.append((edge, False))
                break
        if not has_next:
            break
    # add last edge
    edge = next(e for p, e in graph[current] if e not in loop)
    if current == edge.first_p:
        loop.append((edge, False))
    else:
        loop.append((edge, True))

    edge_loops.append(loop)
    return edge_loops


def get_visualization(face_edge_map: dict[int, list[Edge]], face_idx: int) -> None:
    if face_idx not in face_edge_map:
        raise IndexError(f"Face index {face_idx} not in face edge map.")

    edges = face_edge_map[face_idx]
    edge_loops = get_edge_loops(edges)

    # visualize
    loop_points = []
    for loop in edge_loops:
        # for edge, reverse in loop:
        #     if reverse:
        #         print(edge.last_p, edge.first_p)
        #     else:
        #         print(edge.first_p, edge.last_p)

        all_points = []
        for edge, reverse in loop:
            points = edge.sample_points(10)
            if reverse:
                points = points[::-1]
            all_points.append(points[:-1])  # last point is first of the next edge
        loop_points.append(np.vstack(all_points))
        break

    ps.init()
    for idx, points in enumerate(loop_points):
        edges = np.array(
            [(i, i + 1) for i in range(len(points) - 1)] + [(len(points) - 1, 0)]
        )
        ps.register_curve_network(f"loop_{idx}", points, edges)
    ps.show()

    return


def project_points_to_2d(
    points: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Project 3D points onto a 2D plane using SVD.
    """
    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Perform PCA to find the best-fitting plane
    _, eigenvalues, eigenvectors = np.linalg.svd(centered_points)
    normal = eigenvectors[2]  #  third principal component is the direction of least variance, orthogonal to the plane

    # Choose two axes orthogonal to the normal vector
    axis1 = eigenvectors[0]  # is the first principal component (direction of maximum variance).
    axis2 = eigenvectors[1]  # is the second principal component (direction of second maximum variance).

    # Return the projected points and the transformation matrix
    to_2d = np.vstack((axis1, axis2))
    projected = centered_points @ to_2d.T
    # circle_center_3d = centroid + to_2d.T @ circle_center_2d
    return projected, to_2d, normal, centroid


def run(step_file: str) -> None:
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError("Error reading file")

    reader.TransferRoots()
    shape = reader.OneShape()
    face_edge_map = get_edges(shape)
    print(f"Found {len(face_edge_map)} faces.")
    get_visualization(face_edge_map, 3)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filename", help="input step file")
    args = parser.parse_args()
    run(args.filename)
