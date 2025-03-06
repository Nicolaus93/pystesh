import numpy as np
from numpy.typing import NDArray
from OCP.gp import gp_Pnt

"""
This class defines the infinite cylindrical surface.

Every cylindrical surface is set by the following equation:

S(U,V) = Location + R*cos(U)*XAxis + R*sin(U)*YAxis + V*ZAxis,
where R is cylinder radius.

The local coordinate system of the CylindricalSurface is defined with an axis placement (see class ElementarySurface).

The "ZAxis" is the symmetry axis of the CylindricalSurface, it gives the direction of increasing parametric value V.

The parametrization range is :

U [0, 2*PI],  V ]- infinite, + infinite[
The "XAxis" and the "YAxis" define the placement plane of the surface (Z = 0, and parametric value V = 0) perpendicular to the symmetry axis. The "XAxis" defines the origin of the parameter U = 0. The trigonometric sense gives the positive orientation for the parameter U.

When you create a CylindricalSurface the U and V directions of parametrization are such that at each point of the surface the normal is oriented towards the "outside region".

The methods UReverse VReverse change the orientation of the surface.


"""


def get_2d_points(uv_points: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    TODO
    """
    theta = uv_points[:, 0]
    v = uv_points[:, 1]
    custom_v = v - min(v) + 10

    # Following the parametric equation:
    # S(U,V) = Location + R*cos(U)*XAxis + R*sin(U)*YAxis + V*ZAxis,
    # For 2D projection, we define a custom radius to differentiate top/bottom faces of the cylinder
    x = custom_v * np.cos(theta)
    y = custom_v * np.sin(theta)
    return np.stack((x, y)).T


def get_uv_points_from_2d(
    points_2d: NDArray[np.floating], min_v: float
) -> NDArray[np.floating]:
    """
    Converts 2D points back to UV parameter space for a cone.

    Args:
        points_2d: numpy array of shape (N, 2) containing x,y coordinates
        min_v:

    Returns:
        numpy array of shape (N, 2) containing u,v coordinates
        where u is angle [0, 2π] and v is distance along cylinder surface
    """
    # S(U,V) = Location + R*cos(U)*XAxis + R*sin(U)*YAxis + V*ZAxis,
    y = points_2d[:, 1]
    x = points_2d[:, 0]
    theta = np.atan2(y, x) % (2 * np.pi)

    # Compute custom_v (radius in cylindrical coordinates)
    custom_v = np.sqrt(x**2 + y**2)

    # Reverse the custom_v calculation
    # In original: custom_v = v - min(v) + 1
    # Therefore: v = custom_v + min(v) - 1
    v = custom_v + min_v - 1

    # Stack the results
    return np.stack((theta, v)).T


def get_3d_points_from_2d(
    surface, points_2d: NDArray[np.floating], min_v: float
) -> NDArray[np.floating]:
    uv_points = get_uv_points_from_2d(points_2d=points_2d, min_v=min_v)
    points_3d = []
    for p in uv_points:
        pnt = gp_Pnt()
        surface.D0(p[0], p[1], pnt)
        points_3d.append(pnt.Coord())
    return np.array(points_3d)
