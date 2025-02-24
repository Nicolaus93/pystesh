import contextlib
from argparse import ArgumentParser
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polyscope as ps
from loguru import logger
from OCP.BRep import BRep_Tool
from OCP.Geom import Geom_ConicalSurface
from OCP.OCP.Geom import Geom_Plane
from OCP.OCP.IFSelect import IFSelect_ReturnStatus
from OCP.OCP.STEPControl import STEPControl_Reader

import PythonCDT as cdt
from src.faces.cone import (
    get_2d_points_cone,
    get_3d_points_from_2d,
)
from src.faces.get_edges import get_edge_loops, get_edges
from src.faces.utils import map_points_to_uv, map_uv_to_3d


@dataclass
class CurveNetwork:
    points: npt.NDArray[np.floating]
    edges: npt.NDArray[np.integer]


@dataclass
class FaceMesh:
    vertices: npt.NDArray[np.floating]
    triangles: npt.NDArray[np.integer]
    boundary: CurveNetwork | None = None


def plot_triangulation(triangles: np.ndarray, vertices: np.ndarray) -> None:
    """
    Plot a 2D triangulation.

    Parameters:
    - triangles: An Nx3 numpy array of integers, where each row represents the indices of the vertices of a triangle.
    - vertices: An Mx2 numpy array of floats, where each row represents the coordinates of a vertex in 2D space.
    """
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the triangles
    for triangle in triangles:
        # Get the vertices of the current triangle
        tri_vertices = vertices[triangle]
        # Plot the triangle edges
        ax.plot(
            [
                tri_vertices[0, 0],
                tri_vertices[1, 0],
                tri_vertices[2, 0],
                tri_vertices[0, 0],
            ],
            [
                tri_vertices[0, 1],
                tri_vertices[1, 1],
                tri_vertices[2, 1],
                tri_vertices[0, 1],
            ],
            "b-",
        )

    # Plot the vertices
    ax.plot(vertices[:, 0], vertices[:, 1], "ro")  # 'ro' for red dots

    # Set labels for the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Set equal scaling for the axes
    ax.set_aspect("equal")

    # Show the plot
    plt.savefig("triang.png")


def plot_points_with_edges(points: np.ndarray) -> None:
    """
    Plot 2D points and connect each point to the next one with a line edge.

    Parameters:
    - points: An Nx2 numpy array of floats, where each row represents the coordinates of a point in 2D space.
    """
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the points
    ax.plot(points[:, 0], points[:, 1], "ro")  # 'ro' for red dots

    # Plot the edges (lines connecting each point to the next)
    for i in range(len(points) - 1):
        ax.plot(
            [points[i, 0], points[i + 1, 0]], [points[i, 1], points[i + 1, 1]], "b-"
        )  # 'b-' for blue lines

    # Optionally, connect the last point back to the first point to close the shape
    ax.plot([points[-1, 0], points[0, 0]], [points[-1, 1], points[0, 1]], "b-")

    # Set labels for the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Set equal scaling for the axes
    ax.set_aspect("equal")

    # Show the plot
    plt.savefig("points2d.png")


def plot_edge_points(points_2d):
    plt.figure()
    for idx, points in enumerate(points_2d):
        plt.scatter(points[:, 0], points[:, 1], label=f"Edge {idx}")

    # Add labels, legend, and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatter Plot of Points by Idx")
    plt.legend()

    # save the plot
    plt.savefig("scatter_plot.png")


def get_fast_visualization(face, edges, debug: bool = False) -> FaceMesh:
    edge_loops = get_edge_loops(edges)

    if len(edge_loops) > 1:
        logger.error("Not handling multiple edge loops yet..")
        raise NotImplementedError

    surface = BRep_Tool.Surface_s(face)
    loop_uv_points = []
    for loop_idx, loop in enumerate(edge_loops):
        for edge, reverse in loop:
            edge_points_3d = edge.sample_points(3)
            uv_points = map_points_to_uv(surface, edge_points_3d)
            if reverse:
                uv_points = uv_points[::-1]
            loop_uv_points.append(uv_points)

    if loop_idx > 0:
        raise NotImplementedError

    if isinstance(surface, Geom_ConicalSurface):
        loop_2d_points = get_2d_points_cone(
            uv_points=np.vstack(loop_uv_points),
            reference_radius=surface.Cone().RefRadius(),
            half_angle=surface.Cone().SemiAngle(),
        )
    elif isinstance(surface, Geom_Plane):
        loop_2d_points = np.vstack(loop_uv_points)
    else:
        raise NotImplementedError

    unique_2d_points, idx = np.unique(
        np.round(loop_2d_points, 3), axis=0, return_index=True
    )
    unique_2d_points = unique_2d_points[np.argsort(idx)]
    edges = np.array(
        [(i, i + 1) for i in range(len(unique_2d_points) - 1)]
        + [(len(unique_2d_points) - 1, 0)],
        dtype=np.uintc,
    )
    triangulation = cdt.Triangulation(
        cdt.VertexInsertionOrder.AS_PROVIDED,
        cdt.IntersectingConstraintEdges.NOT_ALLOWED,
        0.0,
    )
    triangulation.insert_vertices(unique_2d_points)
    triangulation.insert_edges(edges)
    triangulation.erase_outer_triangles()

    triangles = np.array([t.vertices for t in triangulation.triangles])
    vertices_2d = np.array([(v.x, v.y) for v in triangulation.vertices])
    if debug:
        temp = [
            get_2d_points_cone(
                p, surface.Cone().RefRadius(), surface.Cone().SemiAngle()
            )
            for p in loop_uv_points
        ]
        plot_edge_points(temp)
        plot_points_with_edges(unique_2d_points)
        plot_triangulation(triangles, vertices_2d)

    if isinstance(surface, Geom_ConicalSurface):
        vertices_3d = get_3d_points_from_2d(surface, vertices_2d)
    elif isinstance(surface, Geom_Plane):
        vertices_3d = map_uv_to_3d(surface, vertices_2d)
    else:
        raise NotImplementedError
    return FaceMesh(vertices=vertices_3d, triangles=triangles, boundary=None)


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
            face_mesh = get_fast_visualization(face, face_edge_map[idx])
            mesh[idx] = face_mesh
        logger.info(f"Done face {idx}")
        if idx > 3:
            break

    ps.init()
    for idx, face_mesh in mesh.items():
        # ps.register_curve_network(f"loop_{idx}", face_mesh.boundary.points, face_mesh.boundary.edges)
        ps.register_surface_mesh(
            f"face_{idx}", face_mesh.vertices, face_mesh.triangles, smooth_shade=True
        )
    ps.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filename", help="input step file")
    args = parser.parse_args()
    run(args.filename)
