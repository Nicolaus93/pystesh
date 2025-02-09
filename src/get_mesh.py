import contextlib
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polyscope as ps
from loguru import logger
from OCP.OCP.IFSelect import IFSelect_ReturnStatus
from OCP.OCP.STEPControl import STEPControl_Reader

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
    boundary: CurveNetwork


def get_visualization(face, edges, debug: bool = False) -> FaceMesh:
    edge_loops = get_edge_loops(edges)

    # visualize
    loop_points = []
    for loop in edge_loops:
        if debug:
            for edge, reverse in loop:
                if reverse:
                    logger.debug(f"{edge.last_p}, {edge.first_p}")
                else:
                    logger.debug(f"{edge.first_p}, {edge.last_p}")

        all_points = []
        for edge, reverse in loop:
            uv_points = edge.sample_points(10)
            uv_points = map_points_to_uv(face, uv_points)
            if reverse:
                uv_points = uv_points[::-1]
            all_points.append(uv_points[:-1])  # last point is first of the next edge
        loop_points.append(np.vstack(all_points))
        break  # TODO: remove (we could have more than 1 loop)

    if len(loop_points) > 1:
        logger.error("Not handling multiple edge loops yet..")
        raise NotImplementedError

    uv_points = loop_points[0]
    edges = np.array(
        [(i, i + 1) for i in range(len(uv_points) - 1)] + [(len(uv_points) - 1, 0)]
    )

    # Create triangulation
    triangulation = create_constrained_triangulation(uv_points, edges)
    if not triangulation:
        raise ValueError("Triangulation Failed")

    # debug = True
    if debug:
        plot_triangulation(uv_points, edges, triangulation)

    # map back to 3d
    vertices = triangulation["vertices"]
    vertices_3d = map_uv_to_3d(face, vertices)
    triangles = triangulation["triangles"]
    # add edges for debugging
    points_3d = map_uv_to_3d(face, uv_points)
    curve_network = CurveNetwork(points=points_3d, edges=edges)
    face_mesh = FaceMesh(
        vertices=vertices_3d, triangles=triangles, boundary=curve_network
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
        if idx == 42:
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
