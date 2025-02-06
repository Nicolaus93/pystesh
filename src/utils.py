import numpy as np
import numpy.typing as npt


def project_points_to_2d(
    points: npt.NDArray[np.floating],
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """
    Project 3D points onto a 2D plane using SVD.
    """
    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Perform SVD to find the best-fitting plane
    _, eigenvalues, eigenvectors = np.linalg.svd(centered_points)
    normal = eigenvectors[
        2
    ]  #  third principal component is the direction of least variance, orthogonal to the plane
    # Choose two axes orthogonal to the normal vector
    axis1 = eigenvectors[
        0
    ]  # is the first principal component (direction of maximum variance).
    axis2 = eigenvectors[
        1
    ]  # is the second principal component (direction of second maximum variance).

    # Return the projected points and the transformation matrix
    to_2d = np.vstack((axis1, axis2))
    projected = centered_points @ to_2d.T
    # circle_center_3d = centroid + to_2d.T @ circle_center_2d
    return projected, to_2d, normal, centroid
