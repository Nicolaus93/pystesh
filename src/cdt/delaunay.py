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


def find_containing_triangle(triangles, point, last_triangle_idx=None):
    """
    Implementation of Lawson's algorithm to find the triangle containing a point.
    Starts from the most recently added triangle (if provided) and "walks" towards the point.
    """
    if not triangles:
        return None

    # Start from the last added triangle if provided, otherwise start from the first triangle
    current_idx = last_triangle_idx if last_triangle_idx is not None else 0

    # Keep track of visited triangles to avoid cycles
    visited = {current_idx}

    it = 0
    while True:
        print(it)
        triangle = triangles[current_idx]

        # Check if the point is inside this triangle
        if point_inside_triangle(triangle, point):
            return current_idx

        # If not inside, find which edge to cross
        a, b, c = triangle
        edges = [(a, b), (b, c), (c, a)]
        next_idx = None

        for i, (v1, v2) in enumerate(edges):
            # Vector from edge to point
            edge_vector = v2 - v1
            point_vector = point - v1

            # If cross product is negative, the point is on the "outside" of this edge
            if np.cross(edge_vector, point_vector) < 0:
                # Find the adjacent triangle that shares this edge
                adjacent_idx = find_adjacent_triangle(triangles, current_idx, (v1, v2))

                if adjacent_idx is not None and adjacent_idx not in visited:
                    next_idx = adjacent_idx
                    visited.add(adjacent_idx)
                    break
        it += 1

        if next_idx is None:
            # If we can't find a next triangle, return the closest one we've found
            return current_idx

        current_idx = next_idx


def find_adjacent_triangle(triangles, triangle_idx, edge):
    """Find the index of the triangle adjacent to the given triangle across the specified edge."""
    v1, v2 = edge
    for i, triangle in enumerate(triangles):
        if i == triangle_idx:
            continue

        # Check if the triangle contains the edge (in either direction)
        edges = [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ]
        for e1, e2 in edges:
            if (np.array_equal(e1, v1) and np.array_equal(e2, v2)) or (
                np.array_equal(e1, v2) and np.array_equal(e2, v1)
            ):
                return i

    return None


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
    return det > 0


def triangulate(points: NDArray[np.floating]):
    """
    Implement Delaunay triangulation using the incremental algorithm.
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
    sorted_points = points[sorted_indices]

    # Step 3: Establish the supertriangle
    # Create a triangle that encompasses all points
    margin = 10  # Extra margin to ensure all points are inside
    supertriangle = np.array(
        [
            [min_vals[0] - margin, min_vals[1] - margin],
            [max_vals[0] + margin, min_vals[1] - margin],
            [(min_vals[0] + max_vals[0]) / 2, max_vals[1] + margin],
        ]
    )

    triangles = [supertriangle]
    last_triangle_idx = 0

    # Step 4-7: Loop over each point and insert into triangulation
    for point in sorted_points:
        # Step 5: Find the triangle containing the point
        containing_idx = find_containing_triangle(triangles, point, last_triangle_idx)

        if containing_idx is None:
            continue  # Skip if no containing triangle found (shouldn't happen with proper supertriangle)

        containing_triangle = triangles[containing_idx]

        # Delete the containing triangle
        triangles.pop(containing_idx)

        # Get vertices of the containing triangle
        v1, v2, v3 = containing_triangle

        # Create three new triangles by connecting point to each vertex
        new_triangles = [
            np.array([point, v1, v2]),
            np.array([point, v2, v3]),
            np.array([point, v3, v1]),
        ]

        # Add the new triangles
        triangles.extend(new_triangles)

        # Step 6: Initialize stack for edge flipping
        # Find triangles adjacent to edges opposite to the new point
        stack = []

        # For each new triangle, find its opposite triangle (if any)
        for i in range(len(triangles) - 3, len(triangles)):
            new_triangle = triangles[i]

            # Find the edge opposite to the new point
            for j in range(3):
                if not np.array_equal(new_triangle[j], point):
                    edge = (new_triangle[j], new_triangle[(j + 1) % 3])

                    # Find the adjacent triangle
                    for k in range(len(triangles)):
                        if k != i:
                            adjacent_triangle = triangles[k]

                            # Check if the adjacent triangle contains this edge
                            for m in range(3):
                                edge_adj = (
                                    adjacent_triangle[m],
                                    adjacent_triangle[(m + 1) % 3],
                                )
                                if (
                                    np.array_equal(edge[0], edge_adj[1])
                                    and np.array_equal(edge[1], edge_adj[0])
                                ) or (
                                    np.array_equal(edge[0], edge_adj[0])
                                    and np.array_equal(edge[1], edge_adj[1])
                                ):
                                    stack.append((i, k))
                                    break

        # Step 7: Restore Delaunay triangulation (edge flipping)
        while stack:
            t1_idx, t2_idx = stack.pop()

            # Get the triangles
            t1 = triangles[t1_idx]
            t2 = triangles[t2_idx]

            # Find the shared edge
            shared_vertices = []
            for v1 in t1:
                for v2 in t2:
                    if np.array_equal(v1, v2):
                        shared_vertices.append(v1)

            if len(shared_vertices) != 2:
                continue  # Not exactly 2 shared vertices, skip

            # Find the non-shared vertices
            v1 = None
            for v in t1:
                if not any(np.array_equal(v, sv) for sv in shared_vertices):
                    v1 = v
                    break

            v2 = None
            for v in t2:
                if not any(np.array_equal(v, sv) for sv in shared_vertices):
                    v2 = v
                    break

            # Check if edge flip is needed (Delaunay condition)
            if in_circumcircle(t2, v1):
                # Swap the diagonal
                new_t1 = np.array([v1, v2, shared_vertices[0]])
                new_t2 = np.array([v1, v2, shared_vertices[1]])

                # Replace the old triangles
                triangles[t1_idx] = new_t1
                triangles[t2_idx] = new_t2

                # Add potentially affected triangles to the stack
                for i, triangle in enumerate(triangles):
                    if i != t1_idx and i != t2_idx:
                        # Check if this triangle shares an edge with new_t1 or new_t2
                        edges1 = [
                            (new_t1[0], new_t1[1]),
                            (new_t1[1], new_t1[2]),
                            (new_t1[2], new_t1[0]),
                        ]
                        edges2 = [
                            (new_t2[0], new_t2[1]),
                            (new_t2[1], new_t2[2]),
                            (new_t2[2], new_t2[0]),
                        ]

                        for edge in edges1 + edges2:
                            tri_edges = [
                                (triangle[0], triangle[1]),
                                (triangle[1], triangle[2]),
                                (triangle[2], triangle[0]),
                            ]

                            for tri_edge in tri_edges:
                                if (
                                    np.array_equal(edge[0], tri_edge[0])
                                    and np.array_equal(edge[1], tri_edge[1])
                                ) or (
                                    np.array_equal(edge[0], tri_edge[1])
                                    and np.array_equal(edge[1], tri_edge[0])
                                ):
                                    if (i, t1_idx) not in stack and (
                                        t1_idx,
                                        i,
                                    ) not in stack:
                                        stack.append((i, t1_idx))
                                    if (i, t2_idx) not in stack and (
                                        t2_idx,
                                        i,
                                    ) not in stack:
                                        stack.append((i, t2_idx))

        # Update the last triangle index
        last_triangle_idx = len(triangles) - 1

    # Remove triangles that contain vertices of the supertriangle
    final_triangles = []
    for triangle in triangles:
        has_super_vertex = False
        for vertex in triangle:
            if any(np.array_equal(vertex, sv) for sv in supertriangle):
                has_super_vertex = True
                break

        if not has_super_vertex:
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
