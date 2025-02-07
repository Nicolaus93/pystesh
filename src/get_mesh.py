from argparse import ArgumentParser

from OCP.OCP.BRep import BRep_Tool
from OCP.OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.OCP.IFSelect import IFSelect_ReturnStatus
from OCP.OCP.STEPControl import STEPControl_Reader
from OCP.OCP.TopoDS import TopoDS_Face
from OCP.OCP.gp import gp_Pnt

from src.get_edges import get_edge_loops, get_edges
import numpy as np
from src.triangle_example import create_constrained_triangulation, plot_triangulation
import polyscope as ps

import matplotlib.pyplot as plt


def map_points_to_uv(
    face: TopoDS_Face,
    points_3d: np.ndarray,
    tolerance: float = 1e-6,
    debug: bool = False,
) -> np.ndarray:
    """Map 3D points to (u, v) parametric coordinates on the B-spline surface."""
    surface = BRep_Tool.Surface_s(
        face
    )  # Surface_s automatically handles the extraction of the correct surface type
    uv_coords = []
    for point in points_3d:
        pnt = gp_Pnt(point[0], point[1], point[2])
        projector = GeomAPI_ProjectPointOnSurf(pnt, surface, tolerance)
        if projector.NbPoints() > 0:
            u, v = projector.LowerDistanceParameters()
            uv_coords.append([u, v])
        else:
            raise ValueError(f"Point {point} could not be projected onto the surface.")

    uv_coords = np.array(uv_coords)
    if debug:
        # points_2d, to_2d, normal, centroid = project_points_to_2d(points_3d)
        plt.scatter(uv_coords[:, 0], uv_coords[:, 1], color="blue", alpha=0.5)

        # Add title and labels
        plt.title("Scatter Plot Example")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        # Show the plot
        plt.savefig("scatter_plot.png")
    return np.array(uv_coords)


def map_uv_to_3d(face: TopoDS_Face, uv_coords: np.ndarray) -> np.ndarray:
    """
    Map (u, v) parametric coordinates to 3D points on the B-spline surface.

    Args:
        face: The TopoDS_Face representing the B-spline surface.
        uv_coords: A numpy array of shape (n, 2) containing (u, v) coordinates.

    Returns:
        A numpy array of shape (n, 3) containing the corresponding 3D points.
    """
    # Extract the geometric surface from the TopoDS_Face
    surface = BRep_Tool.Surface_s(face)

    # Map (u, v) to 3D
    points_3d = []
    for u, v in uv_coords:
        pnt = gp_Pnt()
        surface.D0(u, v, pnt)  # Evaluate the surface at (u, v)
        points_3d.append([pnt.X(), pnt.Y(), pnt.Z()])

    return np.array(points_3d)


def get_visualization(face, edges, debug: bool = False) -> None:
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
            uv_points = edge.sample_points(10)
            uv_points = map_points_to_uv(face, uv_points)
            if reverse:
                uv_points = uv_points[::-1]
            all_points.append(uv_points[:-1])  # last point is first of the next edge
        loop_points.append(np.vstack(all_points))
        break  # TODO: remove (we could have more than 1 loop)

    for idx, uv_points in enumerate(loop_points):
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
        points_3d = map_uv_to_3d(face, uv_points)
        vertices_3d = map_uv_to_3d(face, vertices)
        triangles = triangulation["triangles"]
        print()

        ps.init()
        ps.register_curve_network(f"loop_{idx}", points_3d, edges)
        ps.register_surface_mesh("face", vertices_3d, triangles, smooth_shade=True)
        ps.show()

    return


def run(step_file: str) -> None:
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError("Error reading file")

    reader.TransferRoots()
    shape = reader.OneShape()
    faces, face_edge_map = get_edges(shape)
    print(f"Found {len(face_edge_map)} faces.")
    idx = 55
    get_visualization(faces[idx], face_edge_map[idx])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filename", help="input step file")
    args = parser.parse_args()
    run(args.filename)
