import numpy as np
import numpy.typing as npt
from OCP.OCP.BRep import BRep_Tool
from OCP.OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.OCP.gp import gp_Pnt


def map_points_to_uv(
    surface: BRep_Tool.Surface_s,
    points_3d: npt.NDArray[np.floating],
    tolerance: float = 1e-6,
) -> npt.NDArray[np.floating]:
    """Map 3D points to (u, v) parametric coordinates on the B-spline surface."""
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
    return np.array(uv_coords)


def map_uv_to_3d(
    surface: BRep_Tool.Surface_s, uv_coords: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Map (u, v) parametric coordinates to 3D points on the input surface.

    Args:
        surface: input (sur)face
        uv_coords: A numpy array of shape (n, 2) containing (u, v) coordinates.

    Returns:
        A numpy array of shape (n, 3) containing the corresponding 3D points.
    """
    # Map (u, v) to 3D
    points_3d = []
    for u, v in uv_coords:
        pnt = gp_Pnt()
        surface.D0(u, v, pnt)  # Evaluate the surface at (u, v)
        points_3d.append([pnt.X(), pnt.Y(), pnt.Z()])

    return np.array(points_3d)
