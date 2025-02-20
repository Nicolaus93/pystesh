import contextlib
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import singledispatch

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polyscope as ps
from OCP.BRep import BRep_Tool
from OCP.Geom import Geom_ConicalSurface
from loguru import logger
from OCP.OCP.IFSelect import IFSelect_ReturnStatus
from OCP.OCP.STEPControl import STEPControl_Reader

from src.faces.cone import get_2d_points_cone, get_uv_points_from_2d
from src.faces.get_edges import get_edge_loops, get_edges
from src.faces.utils import map_points_to_uv, map_uv_to_3d
from src.triangle_example import create_constrained_triangulation, plot_triangulation


@dataclass
class CurveNetwork:
    points: npt.NDArray[np.floating]
    edges: npt.NDArray[np.integer]


@dataclass
class FaceMesh:
    vertices: npt.NDArray[np.floating]
    triangles: npt.NDArray[np.integer]
    boundary: CurveNetwork | None = None


@dataclass
class Points2d:
    uv_points: npt.NDArray[np.floating]
    proj_points: npt.NDArray[np.floating]


def get_visualization(face, edges, debug: bool = False) -> FaceMesh:
    edge_loops = get_edge_loops(edges)

    # visualize
    loop_points = []
    edge_points = []
    for loop in edge_loops:
        if debug:
            for edge, reverse in loop:
                if reverse:
                    logger.debug(f"{edge.last_p}, {edge.first_p}")
                else:
                    logger.debug(f"{edge.first_p}, {edge.last_p}")

        edge_uv_points = dict()
        edge_3d_points = dict()
        for edge, reverse in loop:
            edge_points_3d = edge.sample_points(3)
            uv_points = map_points_to_uv(face, edge_points_3d)
            if reverse:
                uv_points = uv_points[::-1]
            edge_uv_points[edge.idx] = uv_points
            edge_3d_points[edge.idx] = edge_points_3d

        loop_points.append(edge_uv_points)
        edge_points.append(edge_3d_points)
        break  # TODO: remove (we could have more than 1 loop)

    if len(loop_points) > 1:
        logger.error("Not handling multiple edge loops yet..")
        raise NotImplementedError

    # Create triangulation
    loop = loop_points[0]  # TODO: for all loops
    points_2d_dict = dict()
    for key, uv_points in loop.items():
        surface = BRep_Tool.Surface_s(face)
        if isinstance(surface, Geom_ConicalSurface):
            points_2d_dict[key] = get_2d_points_cone(uv_points)
        else:
            raise NotImplementedError

    points_2d = np.vstack(list(points_2d_dict.values()))
    edges = np.array(
        [(i, i + 1) for i in range(len(points_2d) - 1)] + [(len(points_2d) - 1, 0)]
    )

    triangulation = create_constrained_triangulation(points_2d, edges)
    if not triangulation:
        raise ValueError("Triangulation Failed")

    if debug:
        plt.figure()
        for key, points in points_2d_dict.items():
            plt.scatter(points[:, 0], points[:, 1], label=f"Edge {key}")

        # Add labels, legend, and title
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Scatter Plot of Points by Idx")
        plt.legend()

        # save the plot
        plt.savefig("scatter_plot.png")

        # save the triangulation
        plot_triangulation(points_2d, edges, triangulation)

    # map back to 3d
    triangles = triangulation["triangles"]
    vertices = triangulation["vertices"]
    uv_points = get_uv_points_from_2d(vertices)
    # uv_points = np.column_stack((u, v))
    vertices_3d = map_uv_to_3d(face, uv_points)

    # add edges for debugging
    curve_network = CurveNetwork(points=vertices_3d, edges=edges)
    face_mesh = FaceMesh(
        vertices=vertices_3d,
        triangles=triangles,
        boundary=curve_network
    )
    return face_mesh


def run(step_file: str) -> None:
    logger.info("Reading step file..")
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError("Error reading file")

    reader.TransferRoots()
    shape = reader.OneShape()
    logger.info("Retrieving face edges..")
    faces, face_edge_map = get_edges(shape)
    logger.info(f"Found {len(face_edge_map)} faces.")
    # idx = 55
    mesh = dict()
    for idx, face in enumerate(faces):
        logger.info(f"Meshing face {idx}")
        with contextlib.suppress(NotImplementedError):
            face_mesh = get_visualization(face, face_edge_map[idx])
            mesh[idx] = face_mesh
        break

    ps.init()
    for idx, face_mesh in mesh.items():
        ps.register_curve_network(f"loop_{idx}", face_mesh.boundary.points, face_mesh.boundary.edges)
        ps.register_surface_mesh(
            f"face_{idx}", face_mesh.vertices, face_mesh.triangles, smooth_shade=True
        )
    ps.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filename", help="input step file")
    args = parser.parse_args()
    run(args.filename)
