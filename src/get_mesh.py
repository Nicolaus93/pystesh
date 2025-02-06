from argparse import ArgumentParser

from OCP.OCP.BRep import BRep_Tool
from OCP.OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.OCP.IFSelect import IFSelect_ReturnStatus
from OCP.OCP.STEPControl import STEPControl_Reader
from OCP.OCP.gp import gp_Pnt

from src.get_edges import get_edge_loops, get_edges
import numpy as np
from src.triangle_example import create_constrained_triangulation, plot_triangulation
from src.utils import project_points_to_2d

import matplotlib.pyplot as plt


def map_points_to_uv(
    face, points_3d: np.ndarray, tolerance: float = 1e-6
) -> np.ndarray:
    """Map 3D points to (u, v) parametric coordinates on the B-spline surface."""
    surface = BRep_Tool.Surface_s(face)

    uv_coords = []
    for point in points_3d:
        pnt = gp_Pnt(point[0], point[1], point[2])
        projector = GeomAPI_ProjectPointOnSurf(pnt, surface)
        if projector.NbPoints() > 0:
            u, v = projector.LowerDistanceParameters()
            uv_coords.append([u, v])
        else:
            raise ValueError(f"Point {point} could not be projected onto the surface.")
    return np.array(uv_coords)


def get_visualization(face, edges) -> None:
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
            # temp = map_points_to_uv(face, points)
            if reverse:
                points = points[::-1]
            all_points.append(points[:-1])  # last point is first of the next edge
        loop_points.append(np.vstack(all_points))
        break  # TODO: remove (we could have more than 1 loop)

    # import polyscope as ps
    # ps.init()
    for idx, points in enumerate(loop_points):
        edges = np.array(
            [(i, i + 1) for i in range(len(points) - 1)] + [(len(points) - 1, 0)]
        )
        points_2d, to_2d, normal, centroid = project_points_to_2d(loop_points[0])
        plt.scatter(points_2d[:, 0], points_2d[:, 1], color="blue", alpha=0.5)

        # Add title and labels
        plt.title("Scatter Plot Example")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        # Show the plot
        plt.savefig("scatter_plot.png")

        # Create triangulation
        result = create_constrained_triangulation(points_2d, edges)

        # Plot results
        if result:
            plot_triangulation(points_2d, edges, result)
        else:
            print("Triangulation failed")

        # ps.register_curve_network(f"loop_{idx}", points, edges)
    # ps.show()

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
    idx = 0
    get_visualization(faces[idx], face_edge_map[idx])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filename", help="input step file")
    args = parser.parse_args()
    run(args.filename)
