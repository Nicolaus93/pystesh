from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray


@dataclass
class Triangulation:
    all_points: NDArray[np.floating]
    triangle_vertices: NDArray[np.integer]
    triangle_neighbors: NDArray[np.integer]
    last_triangle_idx: int = 0

    def plot(self, show: bool = True, title: str = "Triangulation", point_labels: bool = False) -> None:
        """
        Plot the triangulation using matplotlib.

        :param show: Whether to call plt.show() after plotting
        :param title: Title of the plot
        :param point_labels: Whether to label points with their indices
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Draw triangles
        for tri in self.triangle_vertices:
            pts = self.all_points[tri]
            tri_closed = np.vstack([pts, pts[0]])  # Close the triangle
            ax.plot(tri_closed[:, 0], tri_closed[:, 1], 'k-', linewidth=1)

        # Draw points
        ax.plot(self.all_points[:, 0], self.all_points[:, 1], 'ro', markersize=3)

        # Optionally label points
        if point_labels:
            for idx, (x, y) in enumerate(self.all_points):
                ax.text(x, y, str(idx), fontsize=8, ha='right', va='bottom', color='blue')

        ax.set_aspect('equal')
        ax.set_title(title)

        if show:
            plt.show()


def point_inside_triangle(triangle: NDArray[np.floating], point: NDArray[np.floating]) -> bool:
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
    determinants = np.array([det1, det2, det3])

    # Check if all determinants have the same sign (inside the triangle)
    if np.all(determinants > 0) or np.all(determinants < 0):
        return True
    else:
        return False


def in_circumcircle(triangle, point) -> bool:
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
    triangulation: Triangulation,
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
    if triangulation.triangle_vertices.shape[0] == 0 or last_triangle_idx < 0:
        raise ValueError("No triangles available or invalid starting triangle")

    # Start from the last added triangle
    triangle_idx = last_triangle_idx

    # Keep track of visited triangles to avoid cycles
    visited = {triangle_idx}
    while True:
        # Get the current triangle vertices
        v_indices = triangulation.triangle_vertices[triangle_idx]
        triangle = triangulation.all_points[v_indices]

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
                adjacent_idx = triangulation.triangle_neighbors[triangle_idx, i]

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
    debug: bool = False
) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """
    Sort points into a spatially coherent order to improve incremental insertion efficiency.

    :param points: input points
    :param debug: if True, show a plot of the grid and points
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

    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the grid
        for i in range(grid_size + 1):
            ax.axhline(i / grid_size, color='gray', linestyle='--', linewidth=0.5)
            ax.axvline(i / grid_size, color='gray', linestyle='--', linewidth=0.5)

        # Plot the points
        ax.scatter(normalized_points[:, 0], normalized_points[:, 1], c='blue', label='Original points')
        ax.scatter(sorted_points[:, 0], sorted_points[:, 1], c='red', s=10, label='Sorted points', alpha=0.6)

        ax.set_title("Debug: Normalized Points and Grid")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.legend()
        plt.grid(False)
        plt.show()

    return sorted_points, sorted_indices



def initialize_triangulation(
    sorted_points: NDArray[np.floating],
    margin: float = 10.0,
) -> Triangulation:
    """
    Initialize the triangulation with a super triangle.

    :param sorted_points: Sorted input points
    :param margin: extra margin to ensure all points are inside the super triangle
    :param debug: if True, show a plot with the super triangle
    :return: All points array, triangle vertices, triangle neighbors, and last triangle index
    """
    super_vertices = np.array([
        [-margin, -margin],
        [margin, -margin],
        [0.5, margin],
    ])

    # Add super-triangle vertices to the points array
    all_points = np.vstack([sorted_points, super_vertices])
    n_original_points = len(sorted_points)

    # Initial triangle is the super-triangle
    # Vertex indices are n_original_points, n_original_points+1, n_original_points+2
    triangle_vertices = np.array(
        [[n_original_points, n_original_points + 1, n_original_points + 2]]
    )

    # Initial triangle has no neighbors (all boundaries)
    triangle_neighbors = np.array([-1, -1, -1]).reshape(1, -1)

    return Triangulation(
        all_points=all_points,
        triangle_vertices=triangle_vertices,
        triangle_neighbors=triangle_neighbors,
    )


def get_triangle_relation(
    t1_idx: int,
    t2_idx: int,
    triangle_vertices: NDArray[np.integer]
) -> tuple[int, int, list[int]]:
    """
    Find the shared edges between two triangles.
    """
    t1 = sorted(triangle_vertices[t1_idx])
    t1_edges = {(t1[0], t1[1]), (t1[1], t1[2]), (t1[2], t1[0])}

    t2 = sorted(triangle_vertices[t2_idx])
    t2_edges = {(t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0])}

    shared = list(t1_edges & t2_edges)
    unique_t1 = list(t1_edges - t2_edges)
    unique_t2 = list(t2_edges - t1_edges)
    if len(shared) != 2 or len(unique_t1) != 1 or len(unique_t2) != 1:
        raise RuntimeError(f"Invalid triangle pair: {t1_idx}, {t2_idx}")
    return unique_t1[0], unique_t2[0], shared



def lawson_swapping(
    point_idx: int,
    stack: list[tuple[int, int]],
    triangulation: Triangulation,
) -> None:
    """
    Restore the Delaunay condition by flipping edges as necessary.
    After the insertion of P, all the triangles which are now opposite P are placed on a stack. Each triangle is then
    removed from the stack, one at a time, and a check is made to see if P lies inside its circumcircle. If this is the
    case, then the two triangles which share the edge opposite P violate the Delaunay condition and form a convex
    quadrilateral with the diagonal drawn in the wrong direction. To satisfy the Delaunay constraint that each triangle
    has an empty circumcircle, the diagonal of the quadrilateral is simply swapped, and any triangles which are now
    opposite P are placed on the stack.

    This function performs Lawson's algorithm by using a stack to iteratively check
    whether a newly inserted point lies within the circumcircle of neighboring triangles.
    If it does, the shared edge is flipped and new candidate edges are added to the stack.

    :param point_idx: Index of the newly inserted point
    :param stack: Stack of triangle pairs (t1, t2) to check for legality
    :param triangulation: Triangulation structure containing geometry and topology
    """
    vertices = triangulation.triangle_vertices
    neighbors = triangulation.triangle_neighbors
    points = triangulation.all_points

    while stack:
        t1_idx, t2_idx = stack.pop()
        p1_idx, p2_idx, shared_vertices = get_triangle_relation(t1_idx, t2_idx, vertices)

        # Get the opposite and shared points
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        s1 = points[shared_vertices[0]]
        s2 = points[shared_vertices[1]]

        # Determine which triangle contains the inserted point
        if point_idx in (p1_idx, p2_idx):
            p = points[point_idx]
            to_check = [p2, s1, s2] if p1_idx == point_idx else [p1, s1, s2]
        else:
            raise RuntimeError(f"{point_idx} not found in triangles {t1_idx}, {t2_idx}")

        # Perform circumcircle test
        if in_circumcircle(np.array(to_check), p):
            logger.info(f"Point {p} lies in circumcircle! Restoring Delaunay condition by flipping edge")

            # Step 1: Flip triangle vertex definitions
            vertices[t1_idx] = [p1_idx, p2_idx, shared_vertices[0]]
            vertices[t2_idx] = [p1_idx, p2_idx, shared_vertices[1]]

            # Step 2: Find opposite neighbors across the shared edge
            def find_opposite_neighbor(t_idx: int, vertex: int) -> int:
                for i in range(3):
                    v1, v2 = vertices[t_idx][(i + 1) % 3], vertices[t_idx][(i + 2) % 3]
                    if vertex not in (v1, v2):
                        return neighbors[t_idx, i]
                return 0  # Boundary

            n1_opp_idx = find_opposite_neighbor(t1_idx, shared_vertices[0])
            n2_opp_idx = find_opposite_neighbor(t2_idx, shared_vertices[1])

            # Step 3: Update neighbor indices for flipped triangles
            def update_neighbors(t_idx, p1, p2, shared_v, opp_n, other_t, other_opp_n):
                for i in range(3):
                    v1, v2 = vertices[t_idx][(i + 1) % 3], vertices[t_idx][(i + 2) % 3]
                    if p1 in (v1, v2) and shared_v in (v1, v2):
                        neighbors[t_idx, i] = opp_n
                    elif p2 in (v1, v2) and shared_v in (v1, v2):
                        neighbors[t_idx, i] = other_t
                    elif p1 in (v1, v2) and p2 in (v1, v2):
                        neighbors[t_idx, i] = other_opp_n

            update_neighbors(t1_idx, p1_idx, p2_idx, shared_vertices[0], n1_opp_idx, t2_idx, n2_opp_idx)
            update_neighbors(t2_idx, p1_idx, p2_idx, shared_vertices[1], n2_opp_idx, t1_idx, n1_opp_idx)

            # Step 4: Redirect adjacency in neighboring triangles
            def redirect_neighbor(neigh_idx, old_idx, new_idx):
                if neigh_idx != 0:
                    for i in range(3):
                        if neighbors[neigh_idx, i] == old_idx:
                            neighbors[neigh_idx, i] = new_idx
                            break

            redirect_neighbor(n1_opp_idx, t1_idx, t2_idx)
            redirect_neighbor(n2_opp_idx, t2_idx, t1_idx)

            # Step 5: Add new triangle pairs to stack for continued flipping
            if n1_opp_idx != 0:
                stack.append((t2_idx, n1_opp_idx))
            if n2_opp_idx != 0:
                stack.append((t1_idx, n2_opp_idx))



def insert_point(
    point_idx: int,
    point: NDArray[np.floating],
    triangulation: Triangulation,
    debug: bool = False,
) -> Triangulation:
    """
    Insert a point into the triangulation.

    :param point_idx: Index of the point to insert
    :param point: Coordinates of the point
    :param triangulation:
    :param debug:
    :return: Updated triangle_vertices, triangle_neighbors, last_triangle_idx
    """
    # Find the triangle containing the point
    containing_idx = find_containing_triangle(triangulation, point, triangulation.last_triangle_idx)

    # Get vertices of the containing triangle
    v1_idx, v2_idx, v3_idx = triangulation.triangle_vertices[containing_idx]
    logger.info(f"Triangle {containing_idx} with vertices {v1_idx}, {v2_idx}, {v3_idx} contains point {point_idx}")

    # Get the original neighbors before we modify anything
    orig_neighbors = triangulation.triangle_neighbors[containing_idx].copy()
    neighbor_12 = orig_neighbors[0]  # neighbor opposite to v3 (across edge v1-v2)
    neighbor_23 = orig_neighbors[1]  # neighbor opposite to v1 (across edge v2-v3)
    neighbor_31 = orig_neighbors[2]  # neighbor opposite to v2 (across edge v3-v1)

    # Create three new triangles by connecting point to each vertex
    new_triangle_1 = [point_idx, v1_idx, v2_idx]
    new_triangle_2 = [point_idx, v2_idx, v3_idx]
    new_triangle_3 = [point_idx, v3_idx, v1_idx]

    # Update triangulation data structure
    # first, update containing_idx
    triangulation.triangle_vertices[containing_idx] = new_triangle_1

    # Then add two more triangles
    triangulation.triangle_vertices = np.vstack(
        (
            triangulation.triangle_vertices,
            [new_triangle_2],
            [new_triangle_3],
        )
    )

    # Get indices for the new triangles
    triangle_count = len(triangulation.triangle_vertices)
    new_triangle_1_idx = containing_idx  # reusing the original index
    new_triangle_2_idx = triangle_count - 2
    new_triangle_3_idx = triangle_count - 1

    # Update neighbors array to accommodate new triangles
    # Add rows for the two new triangles
    triangulation.triangle_neighbors = np.vstack(
        (
            triangulation.triangle_neighbors,
            np.zeros((1, 3)),
            np.zeros((1, 3)),
        )
    )

    # Set up neighbor relationships for the three new triangles
    # Triangle 1 (point, v1, v2): neighbors are [neighbor_12, triangle_3, triangle_2]
    # - Edge (point, v1) is internal to triangle_3
    # - Edge (v1, v2) borders the original neighbor_12
    # - Edge (v2, point) is internal to triangle_2
    triangulation.triangle_neighbors[new_triangle_1_idx] = [neighbor_12, new_triangle_3_idx, new_triangle_2_idx]

    # Triangle 2 (point, v2, v3): neighbors are [neighbor_23, triangle_1, triangle_3]
    # - Edge (point, v2) is internal to triangle_1
    # - Edge (v2, v3) borders the original neighbor_23
    # - Edge (v3, point) is internal to triangle_3
    triangulation.triangle_neighbors[new_triangle_2_idx] = [new_triangle_1_idx, neighbor_23, new_triangle_3_idx]

    # Triangle 3 (point, v3, v1): neighbors are [neighbor_31, triangle_2, triangle_1]
    # - Edge (point, v3) is internal to triangle_2
    # - Edge (v3, v1) borders the original neighbor_31
    # - Edge (v1, point) is internal to triangle_1
    triangulation.triangle_neighbors[new_triangle_3_idx] = [new_triangle_2_idx, neighbor_31, new_triangle_1_idx]

    # Update the original neighbors to point to the correct new triangles
    # and place them on the stack used for Lawson swapping
    stack = []

    if neighbor_12 >= 0:
        stack.append(neighbor_12)
        # Find which edge of neighbor_12 was connected to the original triangle
        neighbor_edges = triangulation.triangle_neighbors[neighbor_12]
        for i, neighbor_ref in enumerate(neighbor_edges):
            if neighbor_ref == containing_idx:
                # Update this neighbor to point to new_triangle_1 instead
                triangulation.triangle_neighbors[neighbor_12, i] = new_triangle_1_idx
                break

    if neighbor_23 >= 0:
        stack.append(neighbor_23)
        neighbor_edges = triangulation.triangle_neighbors[neighbor_23]
        for i, neighbor_ref in enumerate(neighbor_edges):
            if neighbor_ref == containing_idx:
                # Update this neighbor to point to new_triangle_2 instead
                triangulation.triangle_neighbors[neighbor_23, i] = new_triangle_2_idx
                break

    if neighbor_31 >= 0:
        stack.append(neighbor_31)
        neighbor_edges = triangulation.triangle_neighbors[neighbor_31]
        for i, neighbor_ref in enumerate(neighbor_edges):
            if neighbor_ref == containing_idx:
                # Update this neighbor to point to new_triangle_3 instead
                triangulation.triangle_neighbors[neighbor_31, i] = new_triangle_3_idx
                break

    # Update last_triangle_idx to one of the new triangles
    triangulation.last_triangle_idx = new_triangle_1_idx

    # Restore Delaunay triangulation (edge flipping)
    lawson_swapping(point_idx, stack, triangulation)

    if debug:
        triangulation.plot()

    triangulation.last_triangle_idx = triangle_count - 1
    return triangulation


def remove_super_triangle_triangles(
    triangulation: Triangulation,
    points,
    sorted_indices,
    n_original_points
):
    """
    Remove triangles that contain vertices of the super triangle.

    :param triangulation:
    :param points: Original input points
    :param sorted_indices: Indices mapping sorted points back to original points
    :param n_original_points: Number of original points
    :return: List of final triangles
    """
    triangle_vertices = triangulation.triangle_vertices
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


def triangulate(points: NDArray[np.floating]):
    """
    Implement Delaunay triangulation using the incremental algorithm with efficient
    adjacency tracking.

    :param points: Input points to triangulate
    :return: List of triangles forming the Delaunay triangulation
    """
    # Sort points for efficient insertion
    sorted_points, sorted_indices = get_sorted_points(points)

    # Initialize triangulation with super triangle
    triangulation = initialize_triangulation(sorted_points)

    n_original_points = len(sorted_points)

    # Loop over each point and insert into triangulation
    for point_idx, point in enumerate(sorted_points):
        # Insert point and update triangulation
        triangulation = insert_point(
            point_idx=point_idx,
            point=point,
            triangulation=triangulation,
            debug=False,
        )

    # Remove triangles that contain vertices of the super triangle
    final_triangles = remove_super_triangle_triangles(
        triangulation, points, sorted_indices, n_original_points
    )

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
    # print(arr.shape)
    # sorted_points, sorted_indices = get_sorted_points(arr, debug=True)
    # all_points, triangle_vertices, triangle_neighbors, last_triangle_idx = initialize_triangulation(
    #     sorted_points,
    #     margin=5.0,
    #     debug=True
    # )
    # insert_point(
    #     point_idx=0,
    #     point=sorted_points[0],
    #     all_points=all_points,
    #     triangle_vertices=triangle_vertices,
    #     triangle_neighbors=triangle_neighbors,
    #     last_triangle_idx=last_triangle_idx,
    #     debug=True,
    # )

    triangulate(arr)
