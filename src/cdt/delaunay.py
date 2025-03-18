import numpy as np
from numpy.typing import NDArray


def point_inside_triangle(
    triangle: NDArray[np.floating], point: NDArray[np.floating]
) -> bool:
    """
    Check if a point lies inside a triangle using the determinant method with numpy.
    - triangle: List of three vertices [(x1, y1), (x2, y2), (x3, y3)].
    - point: The point to check (x, y).
    Returns True if the point is inside the triangle, False otherwise.
    """
    a, b, c = np.array(triangle)
    p = np.array(point)

    # Compute vectors
    ab = b - a
    ap = p - a

    bc = c - b
    bp = p - b

    ca = a - c
    cp = p - c

    det1 = np.cross(ab, ap)
    det2 = np.cross(bc, bp)
    det3 = np.cross(ca, cp)

    # Check if all determinants have the same sign (inside the triangle)
    if (det1 >= 0 and det2 >= 0 and det3 >= 0) or (
        det1 <= 0 and det2 <= 0 and det3 <= 0
    ):
        return True
    else:
        return False


def in_circumcircle(triangle, point):
    """
    Check if a point lies inside the circumcircle of a triangle.
    Returns True if inside, False otherwise.
    """
    a, b, c = triangle

    # Matrix for the determinant calculation
    matrix = np.array(
        [
            [
                a[0] - point[0],
                a[1] - point[1],
                (a[0] - point[0]) ** 2 + (a[1] - point[1]) ** 2,
            ],
            [
                b[0] - point[0],
                b[1] - point[1],
                (b[0] - point[0]) ** 2 + (b[1] - point[1]) ** 2,
            ],
            [
                c[0] - point[0],
                c[1] - point[1],
                (c[0] - point[0]) ** 2 + (c[1] - point[1]) ** 2,
            ],
        ]
    )

    # Calculate the determinant
    det = np.linalg.det(matrix)

    # If det > 0, the point is inside the circumcircle
    # Add small epsilon to handle numerical precision issues with colinear points
    eps = 1e-10
    return det > eps


def find_containing_triangle(
    all_points: NDArray[np.floating],
    triangle_vertices: NDArray[np.integer],
    triangle_neighbors: NDArray[np.integer],
    point: NDArray[np.floating],
    last_triangle_idx: int,
) -> int:
    """
    Implementation of Lawson's algorithm to find the triangle containing a point.
    Starts from the most recently added triangle and "walks" towards the point.
    Uses the adjacency information from triangle_neighbors to improve efficiency.

    Parameters:
    - all_points: Array of all point coordinates
    - triangle_vertices: Matrix where each row contains vertex indices of a triangle
    - triangle_neighbors: Matrix where each row contains adjacent triangle indices
    - point: The point to locate
    - last_triangle_idx: Index of the last formed triangle (starting point for search)

    Returns:
    - Index of the triangle containing the point, or None if not found
    """
    if triangle_vertices.shape[0] == 0 or last_triangle_idx < 0:
        raise ValueError("No triangles available or invalid starting triangle")

    # Start from the last added triangle
    triangle_idx = last_triangle_idx

    # Keep track of visited triangles to avoid cycles
    visited = {triangle_idx}
    while True:
        # Get the current triangle vertices
        v_indices = triangle_vertices[triangle_idx]
        triangle = all_points[v_indices]

        # Check if the point is inside this triangle
        if point_inside_triangle(triangle, point):
            return triangle_idx

        # If not inside, find which edge to cross using the adjacent triangles information
        edges = [(0, 1), (1, 2), (2, 0)]  # Edge indices in the triangle
        next_idx = None
        for i, (e1, e2) in enumerate(edges):
            # Vector from edge to point
            edge_vector = triangle[e2] - triangle[e1]
            point_vector = point - triangle[e1]

            # If cross product is negative, the point is on the "outside" of this edge
            if np.cross(edge_vector, point_vector) < 0:
                # Get the adjacent triangle for this edge
                adjacent_idx = triangle_neighbors[triangle_idx, i]

                # If there's an adjacent triangle (not a boundary) and we haven't visited it
                if adjacent_idx != 0 and adjacent_idx not in visited:
                    next_idx = adjacent_idx
                    visited.add(adjacent_idx)
                    break

        if next_idx is None:
            raise ValueError(f"Couldn't find a triangle containing {point}")
        triangle_idx = next_idx


def get_sorted_points(
    points: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """
    Sort points into a spatially coherent order to improve incremental insertion efficiency.

    :param points: input points
    :return: sorted points and their original indices
    """
    # Find the min and max along each axis (x and y)
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Normalize the points to [0, 1] range
    normalized_points = (points - min_vals) / (max_vals - min_vals)

    # Step 2: sort the points into bins
    grid_size = int(np.sqrt(len(points)))
    grid_size = max(grid_size, 4)  # Minimum grid size of 4x4

    y_idxs = (0.99 * grid_size * normalized_points[:, 1]).astype(int)
    x_idxs = (0.99 * grid_size * normalized_points[:, 0]).astype(int)

    # Create bin numbers in a snake-like pattern
    bin_numbers = np.zeros(len(points), dtype=int)
    for i in range(len(points)):
        y, x = y_idxs[i], x_idxs[i]
        if y % 2 == 0:
            bin_numbers[i] = y * grid_size + x
        else:
            bin_numbers[i] = (y + 1) * grid_size - x - 1

    # Sort the points by their bin numbers
    sorted_indices = np.argsort(bin_numbers)
    sorted_points = normalized_points[sorted_indices]

    return sorted_points, sorted_indices


def find_adjacent_edge(
    triangle_neighbors: NDArray[np.integer],
    neigh_idx: int,
    containing_triangle_idx: int,
    new_triangle_idx1: int,
    new_triangle_idx2: int,
) -> NDArray[np.integer]:
    """
    Create new neighbors array for a new triangle, handling adjacency with existing triangles.

    :param triangle_neighbors: Matrix of triangle neighbor indices
    :param neigh_idx: Index of a neighbor of the containing triangle
    :param containing_triangle_idx: Index of the triangle being split
    :param new_triangle_idx1: Index of first new triangle
    :param new_triangle_idx2: Index of second new triangle
    :return: Array of neighbor indices for the new triangle
    """
    if neigh_idx != 0:
        # Find the edge in neigh_idx triangle that's adjacent to the containing triangle
        # and update its neighbor to the new triangle
        found_edge = False
        for i in range(3):
            if triangle_neighbors[neigh_idx, i] == containing_triangle_idx:
                triangle_neighbors[neigh_idx, i] = new_triangle_idx1
                found_edge = True
                break

        if not found_edge:
            raise ValueError("Adjacent edge not found in neighbor triangle")

    # Create and return the neighbors array for the new triangle
    return np.array([neigh_idx, new_triangle_idx2, containing_triangle_idx])


def lawson_swapping(
    point_idx,
    stack,
    triangle_vertices,
    triangle_neighbors,
    all_points,
):
    """
    After the insertion of P, all the triangles which are now opposite P are placed on a stack. Each triangle is then
    removed from the stack, one at a time, and a check is made to see if P lies inside its circumcircle. If this is the
    case, then the two triangles which share the edge opposite P violate the Delaunay condition and form a convex
    quadrilateral with the diagonal drawn in the wrong direction. To satisfy the Delaunay constraint that each triangle
    has an empty circumcircle, the diagonal of the quadrilateral is simply swapped, and any triangles which are now
    opposite P are placed on the stack. This process is repeated until the stack is empty, which signals that the
    triangulation has been restored to a Delaunay triangulation,

    :param point_idx: Index of the newly inserted point
    :param stack: Stack of triangle pairs to check
    :param triangle_vertices: Matrix of triangle vertex indices
    :param triangle_neighbors: Matrix of triangle neighbor indices
    :param all_points: Array of all point coordinates
    """
    while stack:
        t1_idx, t2_idx = stack.pop()

        # Find the vertex in t1 that's not in t2
        t1_vertices = set(triangle_vertices[t1_idx])
        t2_vertices = set(triangle_vertices[t2_idx])

        # Find the non-shared vertex in t1
        t1_vertex_left = t1_vertices - t2_vertices
        if len(t1_vertex_left) != 1:
            raise RuntimeError(f"Triangles {t1_idx} and {t2_idx} should share 2 vertices")
        p1_idx = t1_vertex_left.pop()

        # Find the non-shared vertex in t2
        t2_vertex_left = t2_vertices - t1_vertices
        if len(t2_vertex_left) != 1:
            raise RuntimeError(f"Triangles {t1_idx} and {t2_idx} should share 2 vertices")
        p2_idx = t2_vertex_left.pop()

        # Find the shared edge vertices
        shared_vertices = list(t1_vertices.intersection(t2_vertices))
        if len(shared_vertices) != 2:
            raise RuntimeError(f"Triangles {t1_idx} and {t2_idx} should share 2 vertices")

        p1 = all_points[p1_idx]
        p2 = all_points[p2_idx]
        s1 = all_points[shared_vertices[0]]
        s2 = all_points[shared_vertices[1]]

        # Get the actual points for in_circumcircle test
        if p1_idx == point_idx:
            to_check = [p2, s1, s2]
            p = all_points[p1_idx]
        elif p2_idx == point_idx:
            to_check = [p1, s1, s2]
            p = all_points[p2_idx]
        else:
            raise RuntimeError(f"{point_idx} not found in stack triangles {t1_idx} and {t2_idx}")

        # Check if edge flip is needed (Delaunay condition)
        if in_circumcircle(np.array(to_check), p):
            # We need to flip the edge

            # Update the vertex indices of the triangles
            triangle_vertices[t1_idx] = np.array([p1_idx, p2_idx, shared_vertices[0]])
            triangle_vertices[t2_idx] = np.array([p1_idx, p2_idx, shared_vertices[1]])

            # Find the neighbors of t1 and t2 that are opposite to the shared vertices
            # First for t1
            n1_opp_idx = 0
            for i in range(3):
                v1, v2 = triangle_vertices[t1_idx, (i + 1) % 3], triangle_vertices[t1_idx, (i + 2) % 3]
                if shared_vertices[0] not in [v1, v2]:
                    n1_opp_idx = triangle_neighbors[t1_idx, i]
                    break

            # Then for t2
            n2_opp_idx = 0
            for i in range(3):
                v1, v2 = triangle_vertices[t2_idx, (i + 1) % 3], triangle_vertices[t2_idx, (i + 2) % 3]
                if shared_vertices[1] not in [v1, v2]:
                    n2_opp_idx = triangle_neighbors[t2_idx, i]
                    break

            # Update the neighbors for t1
            for i in range(3):
                v1, v2 = triangle_vertices[t1_idx, (i + 1) % 3], triangle_vertices[t1_idx, (i + 2) % 3]
                if p1_idx in [v1, v2] and shared_vertices[0] in [v1, v2]:
                    triangle_neighbors[t1_idx, i] = n1_opp_idx
                elif p2_idx in [v1, v2] and shared_vertices[0] in [v1, v2]:
                    triangle_neighbors[t1_idx, i] = t2_idx
                elif p1_idx in [v1, v2] and p2_idx in [v1, v2]:
                    triangle_neighbors[t1_idx, i] = n2_opp_idx

            # Update the neighbors for t2
            for i in range(3):
                v1, v2 = triangle_vertices[t2_idx, (i + 1) % 3], triangle_vertices[t2_idx, (i + 2) % 3]
                if p1_idx in [v1, v2] and shared_vertices[1] in [v1, v2]:
                    triangle_neighbors[t2_idx, i] = n2_opp_idx
                elif p2_idx in [v1, v2] and shared_vertices[1] in [v1, v2]:
                    triangle_neighbors[t2_idx, i] = t1_idx
                elif p1_idx in [v1, v2] and p2_idx in [v1, v2]:
                    triangle_neighbors[t2_idx, i] = n1_opp_idx

            # Update neighbor triangles to point to the correct triangles after flip
            if n1_opp_idx != 0:
                for i in range(3):
                    if triangle_neighbors[n1_opp_idx, i] == t1_idx:
                        triangle_neighbors[n1_opp_idx, i] = t2_idx
                        break

            if n2_opp_idx != 0:
                for i in range(3):
                    if triangle_neighbors[n2_opp_idx, i] == t2_idx:
                        triangle_neighbors[n2_opp_idx, i] = t1_idx
                        break

            # Add the affected triangles to the stack
            if n1_opp_idx != 0:
                stack.append((t2_idx, n1_opp_idx))
            if n2_opp_idx != 0:
                stack.append((t1_idx, n2_opp_idx))


def triangulate(points: NDArray[np.floating]):
    """
    Implement Delaunay triangulation using the incremental algorithm with efficient
    adjacency tracking.

    :param points: Input points to triangulate
    :return: List of triangles forming the Delaunay triangulation
    """
    sorted_points, sorted_indices = get_sorted_points(points)

    # Step 3: Establish the super-triangle
    # Create a triangle that encompasses all points (in 0-1)
    margin: float = 10.0  # Extra margin to ensure all points are inside
    super_vertices = np.array(
        [
            [-margin, -margin],
            [margin, -margin],
            [0.5, margin],
        ]
    )

    # Add super-triangle vertices to the points array
    all_points = np.vstack([sorted_points, super_vertices])

    n_original_points = len(sorted_points)

    # Initial triangle is the super-triangle
    # Vertex indices are n_original_points, n_original_points+1, n_original_points+2
    triangle_vertices = np.array(
        [[n_original_points, n_original_points + 1, n_original_points + 2]]
    )

    # Initial triangle has no neighbors (all boundaries)
    triangle_neighbors = np.zeros((1, 3), dtype=int)
    last_triangle_idx = 0

    # Step 4-7: Loop over each point and insert into triangulation
    for point_idx, point in enumerate(sorted_points):
        # Step 5: Find the triangle containing the point
        containing_idx = find_containing_triangle(
            all_points, triangle_vertices, triangle_neighbors, point, last_triangle_idx
        )

        # Get vertices of the containing triangle
        v1_idx, v2_idx, v3_idx = triangle_vertices[containing_idx]

        # Create three new triangles by connecting point to each vertex
        # The new triangles will replace the containing triangle and add two more
        # First, replace the containing triangle with the first new triangle
        triangle_vertices[containing_idx] = [point_idx, v1_idx, v2_idx]

        # Then add two more triangles
        triangle_vertices = np.vstack(
            (
                triangle_vertices,
                [[point_idx, v2_idx, v3_idx]],
                [[point_idx, v3_idx, v1_idx]],
            )
        )

        # Update triangle count
        triangle_count = len(triangle_vertices)
        new_triangle_idx1 = triangle_count - 2
        new_triangle_idx2 = triangle_count - 1

        # Get neighboring triangles of the containing triangle
        neigh_idx1, neigh_idx2, neigh_idx3 = triangle_neighbors[containing_idx]

        # Update neighbors for the new triangles
        # First, new triangle (the one that replaced the containing triangle)
        triangle_neighbors[containing_idx] = [
            neigh_idx1,
            new_triangle_idx1,
            new_triangle_idx2,
        ]

        # Second new triangle
        new_neighbors1 = find_adjacent_edge(
            triangle_neighbors=triangle_neighbors,
            neigh_idx=neigh_idx2,
            containing_triangle_idx=containing_idx,
            new_triangle_idx1=new_triangle_idx1,
            new_triangle_idx2=new_triangle_idx2,
        )

        # Third new triangle
        new_neighbors2 = find_adjacent_edge(
            triangle_neighbors=triangle_neighbors,
            neigh_idx=neigh_idx3,
            containing_triangle_idx=containing_idx,
            new_triangle_idx1=new_triangle_idx2,  # swapped compared to second triangle
            new_triangle_idx2=new_triangle_idx1,  # swapped compared to second triangle
        )

        # Add the new neighbor arrays
        triangle_neighbors = np.vstack((triangle_neighbors, [new_neighbors1], [new_neighbors2]))

        # Step 6: Initialize stack for edge flipping
        # Add potentially affected edges to the stack
        stack = []

        if neigh_idx1 != 0:
            stack.append((containing_idx, neigh_idx1))
        if neigh_idx2 != 0:
            stack.append((new_triangle_idx1, neigh_idx2))
        if neigh_idx3 != 0:
            stack.append((new_triangle_idx2, neigh_idx3))

        # Step 7: Restore Delaunay triangulation (edge flipping)
        lawson_swapping(point_idx, stack, triangle_vertices, triangle_neighbors, all_points)

        # Update the last triangle index
        last_triangle_idx = triangle_count - 1

    # Remove triangles that contain vertices of the supertriangle
    final_triangles = []
    for i in range(triangle_vertices.shape[0]):
        has_super_vertex = False
        for j in range(3):
            if triangle_vertices[i, j] >= n_original_points:
                has_super_vertex = True
                break

        if not has_super_vertex:
            # Convert vertex indices back to original points
            v1, v2, v3 = triangle_vertices[i]
            # Map back to the original indices if needed
            if v1 < n_original_points:
                v1 = sorted_indices[v1]
            if v2 < n_original_points:
                v2 = sorted_indices[v2]
            if v3 < n_original_points:
                v3 = sorted_indices[v3]
            triangle = np.array([points[v1], points[v2], points[v3]])
            final_triangles.append(triangle)

    return final_triangles


if __name__ == "__main__":
    arr = np.array(
        [
            [24.311, -7.358],
            [23.574, -9.456],
            [22.657, -11.481],
            [21.566, -13.419],
            [20.31, -15.253],
            [18.899, -16.971],
            [17.342, -18.558],
            [15.653, -20.004],
            [13.844, -21.296],
            [11.929, -22.425],
            [6.53, -24.546],
            [0.791, -25.388],
            [-4.989, -24.905],
            [-10.509, -23.124],
            [-15.48, -20.137],
            [-19.645, -16.1],
            [-22.786, -11.224],
            [-24.738, -5.762],
            [-25.4, 0.0],
            [-23.868, 8.687],
            [-19.458, 16.327],
            [-12.7, 21.997],
            [-4.411, 25.014],
            [4.411, 25.014],
            [12.7, 21.997],
            [19.458, 16.327],
            [23.868, 8.687],
            [25.4, 0.0],
            [25.386, -0.829],
            [25.346, -1.658],
            [25.278, -2.485],
            [25.184, -3.309],
            [25.062, -4.129],
            [24.914, -4.945],
            [24.739, -5.756],
            [24.538, -6.561],
        ]
    )
    triangulate(arr)
