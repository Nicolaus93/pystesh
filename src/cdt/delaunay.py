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

    det1 = ab[0] * ap[1] - ab[1] * ap[0]
    det2 = bc[0] * bp[1] - bc[1] * bp[0]
    det3 = ca[0] * cp[1] - ca[1] * cp[0]
    determinants = np.array([det1, det2, det3])

    # Check if all determinants have the same sign (inside the triangle)
    if np.all(determinants > 0) or np.all(determinants < 0):
        return True
    else:
        return False


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
                if adjacent_idx != -1 and adjacent_idx not in visited:
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
    triangle_neighbors = np.array([-1, -1, -1], dtype=int).reshape(1, -1)

    return Triangulation(
        all_points=all_points,
        triangle_vertices=triangle_vertices,
        triangle_neighbors=triangle_neighbors,
    )


# def get_triangle_relation(
#     t1_idx: int,
#     t2_idx: int,
#     triangle_vertices: NDArray[np.integer]
# ) -> tuple[int, int, list[int]]:
#     """
#     Find the shared edges between two triangles.
#     """
#     t1 = sorted(triangle_vertices[t1_idx])
#     t1_edges = {(t1[0], t1[1]), (t1[1], t1[2]), (t1[2], t1[0])}
#
#     t2 = sorted(triangle_vertices[t2_idx])
#     t2_edges = {(t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0])}
#
#     shared = list(t1_edges & t2_edges)
#     unique_t1 = list(t1_edges - t2_edges)
#     unique_t2 = list(t2_edges - t1_edges)
#     if len(shared) != 2 or len(unique_t1) != 1 or len(unique_t2) != 1:
#         raise RuntimeError(f"Invalid triangle pair: {t1_idx}, {t2_idx}")
#     return unique_t1[0], unique_t2[0], shared
#
#
# def find_edge_index(triangle: NDArray[np.integer], v1: int, v2: int) -> int:
#     """Return index of edge opposite to vertex not in (v1, v2)."""
#     for i in range(3):
#         a, b = triangle[i], triangle[(i + 1) % 3]
#         if {a, b} == {v1, v2}:
#             return (i + 2) % 3  # edge opposite the remaining vertex
#     raise ValueError("Edge not found")
#
#
# def replace_neighbor(triangulation: Triangulation, tri_idx: int, old_neighbor_idx: int, new_neighbor_idx: int) -> None:
#     """Replace one neighbor triangle with another."""
#     for i in range(3):
#         if triangulation.triangle_neighbors[tri_idx, i] == old_neighbor_idx:
#             triangulation.triangle_neighbors[tri_idx, i] = new_neighbor_idx
#             return
#     raise RuntimeError(f"Neighbor {old_neighbor_idx} not found in triangle {tri_idx}")
#
#
#
# def lawson_swapping(
#     point_idx: int,
#     stack: list[tuple[int, int]],
#     triangulation: Triangulation,
# ) -> None:
#     """
#     Restore the Delaunay condition by flipping edges as necessary.
#     After the insertion of P, all the triangles which are now opposite P are placed on a stack. Each triangle is then
#     removed from the stack, one at a time, and a check is made to see if P lies inside its circumcircle. If this is the
#     case, then the two triangles which share the edge opposite P violate the Delaunay condition and form a convex
#     quadrilateral with the diagonal drawn in the wrong direction. To satisfy the Delaunay constraint that each triangle
#     has an empty circumcircle, the diagonal of the quadrilateral is simply swapped, and any triangles which are now
#     opposite P are placed on the stack.
#
#     This function performs Lawson's algorithm by using a stack to iteratively check
#     whether a newly inserted point lies within the circumcircle of neighboring triangles.
#     If it does, the shared edge is flipped and new candidate edges are added to the stack.
#
#     :param point_idx: Index of the newly inserted point
#     :param stack: list of triangles to check for legality
#     :param triangulation: Triangulation structure containing geometry and topology
#     """
#     vertices = triangulation.triangle_vertices  # assuming they're counterclockwise ordered
#     neighbors = triangulation.triangle_neighbors  # assuming neighbor[t_idx, i] shares vertices[t_idx, i], vertices[t_idx, i + 1] with its neighbor
#     points = triangulation.all_points
#
#     p = points[point_idx]
#
#     while stack:
#         t_neigh_idx, c_idx = stack.pop()
#         t_neigh_vertices = vertices[t_neigh_idx]
#         t_vertices = vertices[c_idx]
#         t_points = points[t_neigh_vertices]
#
#         if not in_circumcircle(t_points, p):
#             continue
#
#         logger.info(f"Point {p} lies in circumcircle of triangle {t_neigh_idx}; flipping edge to restore Delaunay.")
#
#         shared_vertices = list(set(t_vertices) & set(t_neigh_vertices))
#         if len(shared_vertices) != 2:
#             raise RuntimeError(f"Triangles {c_idx} and {t_neigh_idx} must share an edge. Shared: {shared_vertices}")
#
#         # TODO: counterclockwise order
#         a, b = shared_vertices
#         c = next(v for v in t_neigh_vertices if v not in shared_vertices)
#
#         old_vertices_1 = vertices[t_neigh_idx]
#         old_vertices_2 = vertices[c_idx]
#
#         # get old neighbor triangles of (p, b, a)
#         for i in range(3):
#             match old_vertices_1[i], old_vertices_1[(i + 1) % 3]:
#                 case (a, point_idx):
#                     t5 = neighbors[t_neigh_idx, i]
#                 case (point_idx, b):
#                     t6 = neighbors[t_neigh_idx, i]
#                 case (b, a):
#                     temp = neighbors[t_neigh_idx, i]
#                     assert temp == t_neigh_idx
#                 case _:
#                     raise RuntimeError
#
#         # get old neighbor triangles of (c, a, b)
#         for i in range(3):
#             match old_vertices_2[i], old_vertices_2[(i + 1) % 3]:
#                 case (c, a):
#                     t8 = neighbors[c_idx, i]
#                 case (a, b):
#                     temp = neighbors[c_idx, i]
#                     assert temp == c_idx
#                 case (b, c):
#                     t7 = neighbors[c_idx, i]
#                 case _:
#                     raise RuntimeError
#
#         # Update triangles
#         # TODO: counterclockwise order
#         vertices[t_neigh_idx] = np.array([point_idx, c, a])
#         # (t_neigh_idx: [point_idx, c, a])
#         neighbors[t_neigh_idx] = [
#             c_idx,  # edge (point_idx, c)
#             t8,  # edge (c, a)
#             t5,  # edge (a, point_idx)
#         ]
#
#         # TODO: counterclockwise order
#         vertices[c_idx] = np.array([point_idx, b, c])
#         # (c_idx: [point_idx, b, c])
#         neighbors[c_idx] = [
#             t6,  # (point_idx, b)
#             t7,  # (b, c)
#             t_neigh_idx,  # edge (point_idx, c)
#         ]
#
#         # Update neighbor triangles that reference old triangles
#         # t6 needs to update c -> t
#         # t8 needs to update t -> c
#         neighbors[t6, np.where(neighbors[t6] == c_idx)] = t_neigh_idx
#         neighbors[t8, np.where(neighbors[t8] == t_neigh_idx)] = c_idx
#
#         # Push the adjacent triangles into the stack to continue flipping if needed
#         if t8 != -1:
#             stack.append((t8, c_idx))
#         if t7 != -1:
#             stack.append((t7, t_neigh_idx))


def orient2d(pa: NDArray, pb: NDArray, pc: NDArray) -> float:
    """
    Shewchuk's robust 2D orientation predicate.
    Returns > 0 if points are in counterclockwise order
    Returns < 0 if points are in clockwise order
    Returns = 0 if points are collinear

    This is a simplified version - for full robustness, use Shewchuk's exact arithmetic
    """
    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1])
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0])
    det = detleft - detright

    # For full robustness, this should use adaptive precision arithmetic
    # This is a simplified version for demonstration
    return det


def ensure_ccw_triangle(vertices: NDArray, points: NDArray) -> NDArray:
    """Ensure triangle vertices are in counterclockwise order"""
    p0, p1, p2 = points[vertices]
    if orient2d(p0, p1, p2) < 0:
        # Swap vertices to make counterclockwise
        return np.array([vertices[0], vertices[2], vertices[1]])
    return vertices


def find_neighbor_edge_index(triangle_neighbors: NDArray, triangle_idx: int, neighbor_idx: int) -> int:
    """
    Find which edge index (0, 1, or 2) connects to the given neighbor.
    Returns the index i such that triangle_neighbors[triangle_idx, i] == neighbor_idx
    """
    neighbors = triangle_neighbors[triangle_idx]
    for i, neighbor in enumerate(neighbors):
        if neighbor == neighbor_idx:
            return i
    raise RuntimeError(f"Triangle {triangle_idx} is not a neighbor of triangle {neighbor_idx}")


def find_shared_edge(tri1_vertices: NDArray, tri2_vertices: NDArray) -> tuple[int, int, int, int]:
    """
    Find the shared edge between two triangles.
    Returns (v1, v2, opposite1, opposite2) where:
    - v1, v2 are the shared vertices
    - opposite1 is the vertex in tri1 not on the shared edge
    - opposite2 is the vertex in tri2 not on the shared edge
    """
    shared = list(set(tri1_vertices) & set(tri2_vertices))
    if len(shared) != 2:
        raise RuntimeError(f"Triangles must share exactly one edge. Shared vertices: {shared}")

    v1, v2 = shared
    opposite1 = next(v for v in tri1_vertices if v not in shared)
    opposite2 = next(v for v in tri2_vertices if v not in shared)

    return v1, v2, opposite1, opposite2


def find_vertex_position(triangle_vertices: NDArray, vertex: int) -> int:
    """Find the position (0, 1, or 2) of a vertex in a triangle"""
    return next(i for i in range(3) if triangle_vertices[i] == vertex)


def lawson_swapping(
    point_idx: int,
    stack: list[tuple[int, int]],
    triangulation: Triangulation,
) -> None:
    """
    Restore the Delaunay condition by flipping edges as necessary.

    This implementation uses robust geometric predicates to ensure proper
    orientation handling and maintains counterclockwise vertex ordering.

    :param point_idx: Index of the newly inserted point
    :param stack: List of (triangle_idx, candidate_triangle_idx) pairs to check
    :param triangulation: Triangulation structure containing geometry and topology
    """
    vertices = triangulation.triangle_vertices
    neighbors = triangulation.triangle_neighbors
    points = triangulation.all_points

    p = points[point_idx]

    logger.info(f"Lawson swapping phase")
    while stack:
        logger.debug(f"Stack -> {stack}")
        t3_idx, t4_idx = stack.pop()

        # Skip if either triangle is invalid
        if t3_idx == -1 or t4_idx == -1:
            continue

        t3 = vertices[t3_idx]
        t4 = vertices[t4_idx]
        t3_points = points[t3]

        # Check if point lies in circumcircle of the neighboring triangle
        if not incircle_test(t3_points, p):
            continue

        logger.warning(f"Point {point_idx} lies in circumcircle of triangle {t3_idx}; flipping edge")

        # Find shared edge and opposite vertices
        a, b, temp, c = find_shared_edge(t4, t3)

        # The candidate triangle should contain the newly inserted point
        assert temp in t4
        assert temp == point_idx

        # Create new triangles after edge flip
        new_t3_vertices = np.array([point_idx, c, a])
        new_t4_vertices = np.array([point_idx, b, c])

        # Ensure counterclockwise orientation
        new_t3_vertices = ensure_ccw_triangle(new_t3_vertices, points)
        new_t4_vertices = ensure_ccw_triangle(new_t4_vertices, points)

        # Find neighbor triangles before updating
        # For triangle t4_idx, find neighbors opposite to each vertex
        old_t4_neighbors = neighbors[t4_idx].copy()
        old_t3_neighbors = neighbors[t3_idx].copy()

        # Find which edges correspond to which neighbors in the old triangles
        # For t4_idx (contains point_idx)
        # Neighbor opposite to a in t4
        pos_a = find_vertex_position(t4, a)
        t6 = old_t4_neighbors[pos_a]
        # Neighbor opposite to b in candidate triangle
        pos_b = find_vertex_position(t4, b)
        t5 = old_t4_neighbors[pos_b]

        # For t3_idx
        # Neighbor opposite to a in neigh triangle
        pos_a_neigh = find_vertex_position(t3, a)
        t7 = old_t3_neighbors[pos_a_neigh]
        # Neighbor opposite to b in neigh triangle
        pos_b_neigh = find_vertex_position(t3, b)
        t8 = old_t3_neighbors[pos_b_neigh]

        # Update triangle vertices
        vertices[t3_idx] = new_t3_vertices
        vertices[t4_idx] = new_t4_vertices

        # Update neighbors for new triangles
        # We need to determine neighbors based on the actual vertex ordering after CCW correction

        # For new_t3: determine neighbors for each edge
        t3_neighbors = np.array([-1, -1, -1])
        for i in range(3):
            curr_v = new_t3_vertices[i]
            next_v = new_t3_vertices[(i + 1) % 3]

            # Edge from curr_v to next_v
            if {curr_v, next_v} == {point_idx, c}:
                t3_neighbors[i] = t4_idx
            elif {curr_v, next_v} == {c, a}:
                t3_neighbors[i] = t8
            elif {curr_v, next_v} == {a, point_idx}:
                t3_neighbors[i] = t5

        # For new_t4: determine neighbors for each edge
        t4_neighbors = np.array([-1, -1, -1])
        for i in range(3):
            curr_v = new_t4_vertices[i]
            next_v = new_t4_vertices[(i + 1) % 3]

            # Edge from curr_v to next_v
            if {curr_v, next_v} == {point_idx, b}:
                t4_neighbors[i] = t6
            elif {curr_v, next_v} == {b, c}:
                t4_neighbors[i] = t7
            elif {curr_v, next_v} == {c, point_idx}:
                t4_neighbors[i] = t3_idx

        neighbors[t3_idx] = t3_neighbors
        neighbors[t4_idx] = t4_neighbors

        # Update references in neighboring triangles
        def update_neighbor_reference(neighbor_idx: int, old_triangle: int, new_triangle: int):
            if neighbor_idx != -1:
                mask = neighbors[neighbor_idx] == old_triangle
                if np.any(mask):
                    neighbors[neighbor_idx][mask] = new_triangle

        # Update all affected neighbor references
        update_neighbor_reference(t5, t4_idx, t3_idx)
        update_neighbor_reference(t7, t3_idx, t4_idx)

        # Add new potentially illegal edges to stack
        # Check edge opposite to point_idx in both new triangles
        if t8 != -1:
            stack.append((t8, t3_idx))
        if t7 != -1:
            stack.append((t7, t4_idx))


def incircle_test(triangle_points: NDArray[np.floating], point: NDArray[[np.floating]]) -> bool:
    """
    Test if a point lies inside the circumcircle of a triangle.
    This should use Shewchuk's robust incircle predicate for full robustness.
    """
    # This is a simplified version - for full robustness, use Shewchuk's exact arithmetic
    p1, p2, p3 = triangle_points

    # Translate so that p3 is at origin
    p1_rel = p1 - p3
    p2_rel = p2 - p3
    p_rel = point - p3

    # Compute the determinant
    det = np.linalg.det([
        [p1_rel[0], p1_rel[1], p1_rel[0] ** 2 + p1_rel[1] ** 2],
        [p2_rel[0], p2_rel[1], p2_rel[0] ** 2 + p2_rel[1] ** 2],
        [p_rel[0], p_rel[1], p_rel[0] ** 2 + p_rel[1] ** 2]
    ])

    return det > 0


def reorder_neighbors_for_triangle(
    original_vertices: np.ndarray,
    final_vertices: np.ndarray,
    original_neighbors: list
) -> list:
    """Reorder neighbor indices to match reordered vertices"""
    if np.array_equal(original_vertices, final_vertices):
        return original_neighbors

    # Create new neighbor array based on final vertex ordering
    final_neighbors = [None] * 3
    for idx in range(3):
        final_vertex = final_vertices[idx]
        # Find where this vertex was in the original ordering
        original_pos = np.where(original_vertices == final_vertex)[0][0]
        # The neighbor opposite to this vertex in the final triangle
        # is the same as the neighbor opposite to it in the original triangle
        final_neighbors[idx] = original_neighbors[original_pos]

    return final_neighbors


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
    logger.debug(f"Triangle {containing_idx} with vertices {v1_idx}, {v2_idx}, {v3_idx} contains point {point_idx}")

    # Get the original neighbors before we modify anything
    orig_neighbors = triangulation.triangle_neighbors[containing_idx].copy()
    neighbor_opp_v1 = orig_neighbors[0]  # neighbor opposite to v1 (across edge v2-v3)
    neighbor_opp_v2 = orig_neighbors[1]  # neighbor opposite to v2 (across edge v3-v1)
    neighbor_opp_v3 = orig_neighbors[2]  # neighbor opposite to v3 (across edge v1-v2)
    logger.debug(f"Neighbours: opposite v3={neighbor_opp_v3}, opposite v1={neighbor_opp_v1}, opposite v2={neighbor_opp_v2}")

    # Create three new triangles by connecting point to each vertex
    # Note that we need to maintain CCW ordering for each triangle

    # Triangle 1: point + edge (v1, v2)
    original_t12_vertices = np.array([point_idx, v1_idx, v2_idx])
    new_triangle_12 = ensure_ccw_triangle(original_t12_vertices, triangulation.all_points)

    # Triangle 2: point + edge (v2, v3)
    original_t23_vertices = np.array([point_idx, v2_idx, v3_idx])
    new_triangle_23 = ensure_ccw_triangle(original_t23_vertices, triangulation.all_points)

    # Triangle 3: point + edge (v3, v1)
    original_t31_vertices = np.array([point_idx, v3_idx, v1_idx])
    new_triangle_31 = ensure_ccw_triangle(original_t31_vertices, triangulation.all_points)

    # Update triangulation data structure
    # first, update containing_idx
    triangulation.triangle_vertices[containing_idx] = new_triangle_12

    # Then add two more triangles
    triangulation.triangle_vertices = np.vstack(
        (
            triangulation.triangle_vertices,
            [new_triangle_23],
            [new_triangle_31],
        )
    )

    # Get indices for the new triangles
    triangle_count = len(triangulation.triangle_vertices)
    new_triangle_12_idx = containing_idx  # reusing the original index
    new_triangle_23_idx = triangle_count - 2
    new_triangle_31_idx = triangle_count - 1

    # Update neighbors array to accommodate new triangles
    # Add rows for the two new triangles
    triangulation.triangle_neighbors = np.vstack(
        (
            triangulation.triangle_neighbors,
            np.full((2, 3), -1, dtype=int)  # Initialize with -1 for no neighbor
        )
    )

    # Set up neighbor relationships for the three new triangles
    # for every triangle [v1, v2, v3] we define the neighbor as:
    # [t_sharing_edge_opposite_of_v1, t_sharing_edge_opposite_of_v2, t_sharing_edge_opposite_of_v3]

    # Triangle 1 (point, v1, v2)
    original_neighs_t12 = [
        neighbor_opp_v3,      # Edge (v1, v2) -> original neighbor opposite to v3
        new_triangle_23_idx,  # Edge (v2, point) -> triangle_23
        new_triangle_31_idx,  # Edge (point, v1) -> triangle_31
    ]
    triangulation.triangle_neighbors[new_triangle_12_idx] = reorder_neighbors_for_triangle(
        original_t12_vertices, new_triangle_12, original_neighs_t12,
    )

    # Triangle 2 (point, v2, v3)
    original_neighs_t23 = [
        neighbor_opp_v1,      # Edge (v2, v3) -> original neighbor opposite to v1
        new_triangle_31_idx,  # Edge (v3, point) -> triangle_31
        new_triangle_12_idx,  # Edge (point, v2) -> triangle_12
    ]
    triangulation.triangle_neighbors[new_triangle_23_idx] = reorder_neighbors_for_triangle(
        original_t23_vertices, new_triangle_23, original_neighs_t23,
    )

    # Triangle 3 (point, v3, v1)
    original_neighs_t31 = [
        neighbor_opp_v2,      # Edge (v3, v1) -> original neighbor opposite to v2
        new_triangle_12_idx,  # Edge (v1, point) -> triangle_12
        new_triangle_23_idx,  # Edge (point, v3) -> triangle_23
    ]
    triangulation.triangle_neighbors[new_triangle_31_idx] = reorder_neighbors_for_triangle(
        original_t31_vertices, new_triangle_31, original_neighs_t31,
    )

    # Update the original neighbors to point to the correct new triangles
    # and place them on the stack used for Lawson swapping
    stack = []

    def update_external_neighbor(neighbor_idx: int, new_triangle_idx: int):
        if neighbor_idx >= 0:
            stack.append((neighbor_idx, new_triangle_idx))
            neighbor_refs = triangulation.triangle_neighbors[neighbor_idx]
            for i, ref in enumerate(neighbor_refs):
                if ref == containing_idx:
                    triangulation.triangle_neighbors[neighbor_idx, i] = new_triangle_idx
                    break

    # Update each external neighbor to point to the correct new triangle
    update_external_neighbor(neighbor_opp_v1, new_triangle_23_idx)  # v2-v3 edge
    update_external_neighbor(neighbor_opp_v2, new_triangle_31_idx)  # v3-v1 edge
    update_external_neighbor(neighbor_opp_v3, new_triangle_12_idx)  # v1-v2 edge

    # Restore Delaunay triangulation (edge flipping)
    if stack:
        lawson_swapping(point_idx, stack, triangulation)

    if debug:
        triangulation.plot()

    # Update last_triangle_idx to one of the new triangles
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
