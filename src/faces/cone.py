from dataclasses import dataclass

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.Geom import Geom_ConicalSurface
from OCP.gp import gp_Pnt, gp_Vec
from OCP.OCP.gp import gp_Ax1


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


# def project_point_to_uv(cone_face: GeomAbs_Cone, point_3d: gp_Pnt):
#     """
#     Project a 3D point on the cone to (u, v) parametric coordinates.
#     step 1: compute v
#     - v is the projection of the vector from the apex to the point onto the cone's axis.
#     - Let d be the unit direction vector of the cone's axis.
#     - Let be the vector from the apex to the point: p=(x−x0,y−y0,z−z0)
#     - Then, p = dot(v, d)
#     Step 2: Compute u
#     - Project p onto the plane perpendicular to the cone's axis: p_proj = p − dot(v, d)
#     - Compute the angle u between p_proj and a reference direction (e.g., the x-axis):
#         u = atan( dot(p_proj, y_ref) / dot(p_proj, x_ref) )
#     where x_ref and y_ref are orthogonal vectors in the plane perpendicular to the axis.
#     """
#     surface = BRep_Tool.Surface_s(cone_face)
#     if not isinstance(surface, Geom_ConicalSurface):
#         raise ValueError("The face is not a cone.")
#
#     cone = surface.Cone()
#     axis = cone.Axis()
#     apex = cone.Apex()
#     # radius = cone.RefRadius()
#     # half_angle = cone.SemiAngle()
#
#     # Vector from apex to the point
#     vec = gp_Vec(apex, point_3d)
#
#     # Project the vector onto the cone's axis to get v (height)
#     axis_vec = gp_Vec(axis.Direction())
#     v = vec.Dot(axis_vec)
#
#     # Project the vector onto the plane perpendicular to the axis
#     p_proj = vec - axis_vec * v
#
#     # Compute reference vectors
#     x_ref = get_occt_reference_direction(surface)
#     y_ref = axis_vec.Crossed(x_ref)
#     y_ref.Normalize()  # Normalize to make it a unit vector
#
#     # Compute u (angle in the plane)
#     u = np.arctan2(p_proj.Dot(y_ref), p_proj.Dot(x_ref))
#     if u < 0:
#         u += 2 * np.pi  # Ensure u is in [0, 2π]
#
#     return u, v


def project_point_to_uv(cone_face, point_3d: gp_Pnt):
    """
    Project a 3D point on the cone to (u, v) parametric coordinates.

    Args:
        cone_face: The conical face
        point_3d: The 3D point to project

    Returns:
        tuple: (u, v) parameters
    """
    # Get the underlying geometric surface
    surface = BRep_Tool.Surface_s(cone_face)
    if not isinstance(surface, Geom_ConicalSurface):
        raise ValueError("The face is not a cone.")

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
