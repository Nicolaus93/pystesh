from collections import defaultdict

import numpy as np
import PythonCDT as cdt
from loguru import logger
from matplotlib import pyplot as plt
from OCP.BRep import BRep_Tool
from OCP.Geom import Geom_BSplineSurface, Geom_ConicalSurface, Geom_Plane
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer

from src.faces.cone import (
    get_2d_points_cone,
    get_3d_points_from_2d,
    get_uv_points_from_2d,
)
from src.faces.get_edges import Edge, get_edge_loops, get_edges
from src.faces.utils import map_points_to_uv, map_uv_to_3d
from src.get_mesh import (
    CurveNetwork,
    FaceMesh,
    plot_edge_points,
    plot_points_with_edges,
)
from src.triangle_example import create_constrained_triangulation, plot_triangulation


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
        vertices=vertices_3d, triangles=triangles, boundary=curve_network
    )
    return face_mesh


def get_face_mesh(face, edges: set[Edge], idx: int, debug: bool = False) -> FaceMesh:
    edge_loops = get_edge_loops(edges)
    if len(edge_loops) > 3:
        if debug:
            # Create a STEP writer
            step_writer = STEPControl_Writer()

            # Add the shape to the STEP writer
            step_writer.Transfer(face, STEPControl_AsIs)
            output_file = f"/tmp/face_{idx}.step"
            status = step_writer.Write(output_file)

            # # Check if the export was successful
            # if status == IFSelect_ReturnStatus.IFSelect_RetDone:
            #     print(f"STEP file '{output_file}' successfully exported.")
            # else:
            #     print("Failed to export STEP file.")

        raise ValueError(f"Face {idx} contains {len(edge_loops)} edge loops!")

    surface = BRep_Tool.Surface_s(face)
    if isinstance(surface, Geom_BSplineSurface):
        raise NotImplementedError

    loop_uv_points = defaultdict(list)
    for loop_idx, loop in enumerate(edge_loops):
        for edge, reverse in loop:
            edge_points_3d = edge.sample_points(3)
            uv_points = map_points_to_uv(
                surface, edge_points_3d
            )  # TODO: extremely slow for Bspline surfaces
            if reverse:
                uv_points = uv_points[::-1]
            loop_uv_points[loop_idx].append(uv_points)

    edges = []
    points = []
    for loop_idx, group in loop_uv_points.items():
        temp = np.vstack(group)
        if isinstance(surface, Geom_ConicalSurface):
            loop_2d_points = get_2d_points_cone(
                uv_points=temp,
                reference_radius=surface.Cone().RefRadius(),
                half_angle=surface.Cone().SemiAngle(),
            )
        elif isinstance(surface, Geom_Plane):
            loop_2d_points = temp
        else:
            raise NotImplementedError
        unique_2d_points, idx = np.unique(
            np.round(loop_2d_points, 3), axis=0, return_index=True
        )
        unique_2d_points = unique_2d_points[np.argsort(idx)]
        points.append(unique_2d_points)
        offset = len(edges[-1]) if edges else 0
        new_edges = (
            np.array(
                [(i, i + 1) for i in range(len(unique_2d_points) - 1)]
                + [(len(unique_2d_points) - 1, 0)],
                dtype=np.uintc,
            )
            + offset
        )
        edges.append(new_edges)

    points = np.vstack(points)
    edges = np.vstack(edges)
    triangulation = cdt.Triangulation(
        cdt.VertexInsertionOrder.AS_PROVIDED,
        cdt.IntersectingConstraintEdges.TRY_RESOLVE,
        0.0,
    )
    triangulation.insert_vertices(points)
    triangulation.insert_edges(edges)
    triangulation.erase_outer_triangles_and_holes()

    triangles = np.array([t.vertices for t in triangulation.triangles])
    vertices_2d = np.array([(v.x, v.y) for v in triangulation.vertices])
    if debug:
        if isinstance(surface, Geom_ConicalSurface):
            temp = [
                get_2d_points_cone(
                    edge, surface.Cone().RefRadius(), surface.Cone().SemiAngle()
                )
                for k, group in loop_uv_points.items()
                for edge in group
            ]
        elif isinstance(surface, Geom_Plane):
            temp = [edge for k, group in loop_uv_points.items() for edge in group]
        else:
            raise NotImplementedError

        plot_edge_points(temp)
        plot_points_with_edges(points, edges)
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
    reader.ReadFile(step_file)
    reader.TransferRoots()
    shape = reader.OneShape()
    logger.info("Retrieving face edges..")
    faces, face_edge_map = get_edges(shape)
    logger.info(f"Found {len(face_edge_map)} faces.")
    mesh = dict()
    for idx, face in enumerate(faces):
        logger.info(f"Meshing face {idx}")
        try:
            face_mesh = get_face_mesh(face, face_edge_map[idx], idx)
            mesh[idx] = face_mesh
            logger.info(f"Done face {idx}")
        except NotImplementedError:
            logger.info(f"Skipped face {idx}")


if __name__ == "__main__":
    run("/home/nini/projects/pystesh/assets/face42.step")
