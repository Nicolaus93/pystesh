import numpy as np
import numpy.typing as npt
from OCP.Geom import Geom_ConicalSurface
from OCP.gp import gp_Pnt, gp_Vec


def get_2d_points_cone(
    uv_points: npt.NDArray[np.floating], reference_radius: float, half_angle: float
) -> npt.NDArray[np.floating]:
    theta = uv_points[:, 0]
    v = uv_points[:, 1]

    # Following the parametric equation:
    # P(u,v) = O + (R + v*sin(Ang))*(cos(u)*XDir + sin(u)*YDir) + v*cos(Ang)*ZDir
    # For 2D projection, we only need the radius term: (R + v*sin(Ang))
    radius = reference_radius + v * np.sin(half_angle)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack((x, y)).T


def get_uv_points_from_2d(
    points_2d: npt.NDArray[np.floating], reference_radius: float, half_angle: float
) -> npt.NDArray[np.floating]:
    """
    Converts 2D points back to UV parameter space for a cone.

    Args:
        points_2d: numpy array of shape (N, 2) containing x,y coordinates
        reference_radius: radius of the cone at the reference plane
        half_angle: half-angle at the apex of the cone (in radians)

    Returns:
        numpy array of shape (N, 2) containing u,v coordinates
        where u is angle [0, 2π] and v is distance along cone surface
    """
    # Calculate theta (u parameter)
    theta = np.atan2(points_2d[:, 1], points_2d[:, 0]) % (2 * np.pi)

    # Calculate radius from origin
    radius = np.sqrt(points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2)

    # Solve for v using the radius equation:
    # radius = reference_radius + v * sin(half_angle)
    v = (radius - reference_radius) / np.sin(half_angle)

    return np.stack((theta, v)).T


def get_3d_points_from_2d(surface, points_2d):
    uv_points = get_uv_points_from_2d(
        points_2d=points_2d,
        reference_radius=surface.Cone().RefRadius(),
        half_angle=surface.Cone().SemiAngle(),
    )
    points_3d = []
    for p in uv_points:
        pnt = gp_Pnt()
        surface.D0(p[0], p[1], pnt)
        points_3d.append(pnt.Coord())
    return np.array(points_3d)
