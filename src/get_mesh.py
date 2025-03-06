from argparse import ArgumentParser
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import polyscope as ps
from loguru import logger
from numpy.typing import NDArray
from OCP.BRep import BRep_Tool
from OCP.Geom import (
    Geom_BSplineSurface,
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_Plane,
)
from OCP.OCP.IFSelect import IFSelect_ReturnStatus
from OCP.OCP.STEPControl import STEPControl_Reader

import PythonCDT as cdt
from src.faces import cylinder
from src.faces.cone import (
    get_2d_points_cone,
    get_3d_points_from_2d,
)
from src.faces.get_edges import Edge, get_edges
from src.faces.utils import map_points_to_uv, map_uv_to_3d


@dataclass
class CurveNetwork:
    points: NDArray[np.floating]
    edges: NDArray[np.integer]


@dataclass
class FaceMesh:
    vertices: NDArray[np.floating]
    triangles: NDArray[np.integer]
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


def plot_points_with_edges(
    points: NDArray[np.floating], edges: NDArray[np.integer]
) -> None:
    """
    Plot 2D points and connect specified points with line edges.

    Parameters:
    - points: An Nx2 numpy array of floats, where each row represents the coordinates of a point in 2D space.
    - edges: A list of tuples, where each tuple contains the indices of two points to be connected by an edge.
    """
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the points
    ax.plot(points[:, 0], points[:, 1], "ro")  # 'ro' for red dots

    # Plot the edges (lines connecting specified points)
    for edge in edges:
        start, end = edge
        ax.plot(
            [points[start, 0], points[end, 0]], [points[start, 1], points[end, 1]], "b-"
        )  # 'b-' for blue lines

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


def remap_edges_after_deduplication(
    points: NDArray[np.floating], edges: NDArray[np.integer]
) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """
    Remove duplicate points while preserving original order and remap edge indices accordingly.

    Parameters:
    -----------
    points : NDArray[np.floating]
        Array of shape (n, 2) containing 2D points.
    edges : NDArray[np.integer]
        Array of shape (m, 2) containing pairs of indices representing edges between points.

    Returns:
    --------
    Tuple[NDArray[np.floating], NDArray[np.integer]]
        A tuple containing:
        - unique_points: NDArray[np.floating] of shape (k, 2) with duplicates removed, preserving original order.
        - remapped_edges: NDArray[np.integer] of shape (m, 2) with indices updated to reference unique_points.
    """
    # Round points to 3 decimal places
    rounded_points = np.round(points, 3)

    # Get unique points with both indices and inverse mapping
    unique_points, indices, indices_map = np.unique(
        rounded_points, axis=0, return_index=True, return_inverse=True
    )

    # Sort unique points according to their first appearance in the original array
    sorted_order = np.argsort(indices)
    unique_points = unique_points[sorted_order]

    # Create a mapping from original unique indices to new sorted indices
    position_map = np.zeros(len(sorted_order), dtype=np.int64)
    position_map[sorted_order] = np.arange(len(sorted_order))

    # Update the inverse indices to reflect the new sorting
    new_indices_map = position_map[indices_map]

    # Remap edges using updated inverse indices - using vectorized operations
    remapped_edges = new_indices_map[edges]

    return unique_points, remapped_edges.astype(np.uintc)


def get_face_mesh(
    face, face_edges: set[Edge], idx: int, debug: bool = False
) -> FaceMesh:
    surface = BRep_Tool.Surface_s(face)
    if isinstance(surface, Geom_BSplineSurface):
        raise NotImplementedError

    edges_uv_points = list()
    edges = list()
    offset = 0
    for face_edge in face_edges:
        edge_points_3d = face_edge.sample_points(3)
        uv_points = map_points_to_uv(
            surface, edge_points_3d
        )  # TODO: extremely slow for Bspline surfaces
        edges_uv_points.append(uv_points)
        edges.append([(i + offset, i + 1 + offset) for i in range(len(uv_points) - 1)])
        offset += len(uv_points)

    all_uv_points = np.vstack(edges_uv_points)
    try:
        constrained_edges = np.vstack(edges)
    except ValueError:
        raise NotImplementedError

    if isinstance(surface, Geom_ConicalSurface):
        face_2d_points = get_2d_points_cone(
            uv_points=all_uv_points,
            reference_radius=surface.Cone().RefRadius(),
            half_angle=surface.Cone().SemiAngle(),
        )
    elif isinstance(surface, Geom_Plane):
        face_2d_points = all_uv_points
    elif isinstance(surface, Geom_CylindricalSurface):
        face_2d_points = cylinder.get_2d_points(all_uv_points)
    else:
        raise NotImplementedError

    points, constrained_edges = remap_edges_after_deduplication(
        face_2d_points, constrained_edges
    )

    triangulation = cdt.Triangulation(
        cdt.VertexInsertionOrder.AS_PROVIDED,
        cdt.IntersectingConstraintEdges.NOT_ALLOWED,
        0.0,
    )
    triangulation.insert_vertices(points)
    triangulation.insert_edges(constrained_edges)
    # triangulation.erase_super_triangle()
    triangulation.erase_outer_triangles_and_holes()

    triangles = np.array([t.vertices for t in triangulation.triangles])
    vertices_2d = np.array([(v.x, v.y) for v in triangulation.vertices])
    if debug:
        if isinstance(surface, Geom_ConicalSurface):
            temp = [
                get_2d_points_cone(
                    edge, surface.Cone().RefRadius(), surface.Cone().SemiAngle()
                )
                for edge in edges_uv_points
            ]
        elif isinstance(surface, Geom_Plane):
            temp = edges_uv_points
        elif isinstance(surface, Geom_CylindricalSurface):
            temp = [cylinder.get_2d_points(edge) for edge in edges_uv_points]
        else:
            raise NotImplementedError

        plot_edge_points(temp)
        plot_points_with_edges(points, constrained_edges)
        plot_triangulation(triangles, vertices_2d)

    if isinstance(surface, Geom_ConicalSurface):
        vertices_3d = get_3d_points_from_2d(surface, vertices_2d)
    elif isinstance(surface, Geom_Plane):
        vertices_3d = map_uv_to_3d(surface, vertices_2d)
    elif isinstance(surface, Geom_CylindricalSurface):
        vertices_3d = cylinder.get_3d_points_from_2d(
            surface, vertices_2d, min(all_uv_points[:, 1])
        )
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
    mesh = dict()
    for idx, face in enumerate(faces):
        try:
            face_mesh = get_face_mesh(face, face_edge_map[idx], idx)
            mesh[idx] = face_mesh
            logger.info(f"Done face {idx}")
        except Exception as e:
            logger.warning(f"Skipped face {idx} because of {e}")
        # if idx > 40:
        #     break

    ps.init()
    for idx, face_mesh in mesh.items():
        # ps.register_curve_network(f"loop_{idx}", face_mesh.boundary.points, face_mesh.boundary.edges)
        try:
            ps.register_surface_mesh(
                f"face_{idx}",
                face_mesh.vertices,
                face_mesh.triangles,
                smooth_shade=True,
            )
        except ValueError:
            logger.error(f"Failed to mesh face {idx}")
    ps.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filename", help="input step file")
    args = parser.parse_args()
    run(args.filename)
