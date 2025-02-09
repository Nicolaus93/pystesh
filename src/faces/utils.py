import matplotlib.pyplot as plt
import numpy as np
from OCP.OCP.BRep import BRep_Tool
from OCP.OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.OCP.gp import gp_Pnt
from OCP.OCP.TopoDS import TopoDS_Face


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
            assert np.allclose(map_uv_to_3d(face, [(u, v)]), pnt.Coord(), atol=1e-3)
            uv_coords.append([u, v])
        else:
            raise ValueError(f"Point {point} could not be projected onto the surface.")

        # from OCP.OCP.Geom import Geom_ConicalSurface
        # if isinstance(surface, Geom_ConicalSurface):
        #     from src.faces.cone import project_point_to_uv
        #
        #     temp = project_point_to_uv(face, pnt)
        #     print(u, v, temp)

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
