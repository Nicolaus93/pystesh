from dataclasses import dataclass

import numpy as np
from OCP.Geom import Geom_ConicalSurface
from OCP.gp import gp_Pnt, gp_Vec
from OCP.OCP.gp import gp_Ax1


def get_2d_points_cone(uv_points, reference_radius, half_angle):
    theta = uv_points[:, 0]
    v = uv_points[:, 1]

    # Following the parametric equation:
    # P(u,v) = O + (R + v*sin(Ang))*(cos(u)*XDir + sin(u)*YDir) + v*cos(Ang)*ZDir
    # For 2D projection, we only need the radius term: (R + v*sin(Ang))
    radius = reference_radius + v * np.sin(half_angle)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack((x, y)).T


def get_uv_points_from_2d(points_2d, reference_radius, half_angle):
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


# def get_uv_points_from_2d(points_2d):
#     """
#     1. Finds the angle θ using arctan2
#     2. Creates a unit vector in that direction (cos(θ), sin(θ))
#     If the original point was pointing in roughly the same direction as this unit vector (positive dot product),
#     v should be positive. If the original point was pointing in roughly the opposite direction (negative dot product),
#     v should be negative.
#     This correctly recovers the original v value because in the forward transformation:
#
#     - A positive v creates a point in the same direction as (cos(θ), sin(θ))
#     - A negative v creates a point in the opposite direction
#     """
#     theta = np.atan2(points_2d[:, 1], points_2d[:, 0]) % (2 * np.pi)
#     v_magnitude = np.sqrt(points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2)
#
#     # Determine the sign of v based on the direction of the point
#     # If the point is in the same direction as the angle, v is positive; otherwise, it's negative
#     x_direction = np.cos(theta)
#     y_direction = np.sin(theta)
#
#     # Dot product to determine if the point is in the same direction as the angle
#     dot_product = points_2d[:, 0] * x_direction + points_2d[:, 1] * y_direction
#     v_sign = np.sign(dot_product)
#
#     v = v_magnitude * v_sign
#
#     return np.vstack((theta, v)).T


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


@dataclass
class ConeParams:
    """
    A cone is a ruled surface, and its natural parameterization is typically in terms of:
    - u: The angular parameter (around the axis of the cone).
    - v: The height parameter (along the axis of the cone).

    The parametric equations for a cone (centered at the origin and aligned with the z-axis) are:

    x=(R+v⋅tan(α))⋅cos(u)
    y=(R+v⋅tan(α))⋅sin(u)
    z=v

    Where:
    - R is the base radius of the cone.
    - α is the half-angle of the cone.
    - u ranges from 0 to 2π (angular parameter).
    - v ranges from 0 to the height of the cone.

    To get a point from 3d to uv:
    v=z
    y/x = tan(u) => u = atan(y/x)
    """

    axis: gp_Ax1
    apex: gp_Pnt
    radius: float
    half_angle: float


def get_occt_reference_direction(cone_surface: Geom_ConicalSurface) -> gp_Vec:
    """
    Get the reference direction used by OCCT for parameterizing the cone.

    Args:
        cone_surface: The Geom_ConicalSurface representing the cone.

    Returns:
        ref_dir: The reference direction (u = 0) as a gp_Vec.
    """
    # Evaluate the cone at u = 0, v = 0
    u = 0.0
    v = 0.0
    pnt = gp_Pnt()
    cone_surface.D0(u, v, pnt)

    # Vector from apex to the point at u = 0
    apex = cone_surface.Cone().Apex()
    ref_dir = gp_Vec(apex, pnt)
    ref_dir.Normalize()  # Normalize to make it a unit vector

    return ref_dir


def project_point_to_uv(surface: Geom_ConicalSurface, point_3d: gp_Pnt):
    """
    Project a 3D point on the cone to (u, v) parametric coordinates.

    Args:
        surface: The conical face
        point_3d: The 3D point to project

    Returns:
        tuple: (u, v) parameters
    """
    # # Get the underlying geometric surface
    # surface = BRep_Tool.Surface_s(cone_face)
    # if not isinstance(surface, Geom_ConicalSurface):
    #     raise ValueError("The face is not a cone.")

    # Get cone geometric properties
    cone = surface.Cone()
    apex = cone.Apex()
    axis = cone.Axis()
    semi_angle = cone.SemiAngle()

    # Vector from apex to point
    vec_to_point = gp_Vec(apex, point_3d)

    # Get height along axis (v parameter)
    axis_dir = gp_Vec(axis.Direction())
    v = vec_to_point.Dot(axis_dir)

    # Get radius at this height
    radius_at_v = abs(v) * np.tan(semi_angle)

    # Project point onto plane perpendicular to axis at height v
    proj_point = vec_to_point - (v * axis_dir)

    # Get reference direction (u = 0)
    ref_vec = gp_Vec(cone.Position().XDirection())

    # Compute perpendicular direction
    perp_vec = axis_dir.Crossed(ref_vec)
    perp_vec.Normalize()

    # Calculate angle for u parameter
    cos_u = proj_point.Dot(ref_vec) / radius_at_v
    sin_u = proj_point.Dot(perp_vec) / radius_at_v

    u = np.arctan2(sin_u, cos_u)
    if u < 0:
        u += 2 * np.pi  # Ensure u is in [0, 2π]

    return u, v
