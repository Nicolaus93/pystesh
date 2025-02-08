import matplotlib.pyplot as plt
import numpy as np
import triangle


def create_constrained_triangulation(points, edges, quality: bool = False):
    """
    Create a constrained Delaunay triangulation from points and edges.

    Parameters:
    points: array-like, shape (n, 2)
        List of 2D points
    edges: array-like, shape (m, 2)
        List of edges specified by point indices

    Returns:
    dict: Triangle dictionary containing triangulation results
    """
    # Convert points to numpy array if not already
    vertices = np.asarray(points, dtype=np.float64)

    # Convert edges to numpy array if not already
    segments = np.asarray(edges, dtype=np.int32)

    # Verify data integrity
    if vertices.shape[1] != 2:
        raise ValueError("Points must be 2D (shape: Nx2)")

    if segments.shape[1] != 2:
        raise ValueError("Edges must be pairs of indices (shape: Mx2)")

    if np.max(segments) >= len(vertices):
        raise ValueError("Edge indices must be less than number of vertices")

    if np.min(segments) < 0:
        raise ValueError("Edge indices must be non-negative")

    # Create the input dictionary for triangle
    tri_data = {
        "vertices": vertices,
        "segments": segments,
        # Add a marker for each vertex
        "vertex_markers": np.ones(len(vertices), dtype=np.int32),
    }

    # Create the triangulation with constraints
    triangulation = triangle.triangulate(tri_data, "p")  # Start with just 'p'

    # If you need quality constraints, can add them in a second pass
    if triangulation and quality:
        triangulation = triangle.triangulate(triangulation, "q")

    return triangulation


def plot_triangulation(points, edges, triangulation):
    """
    Plot the triangulation results.

    Parameters:
    points: array-like, shape (n, 2)
        Original input points
    edges: array-like, shape (m, 2)
        Original constrained edges
    triangulation: dict
        Triangle output dictionary
    """
    plt.figure(figsize=(10, 10))

    # Convert inputs to numpy arrays
    points = np.asarray(points)
    edges = np.asarray(edges)

    # Plot original points
    plt.plot(points[:, 0], points[:, 1], "ko", label="Vertices")

    # Plot constrained edges
    for edge in edges:
        start, end = edge
        plt.plot(
            [points[start, 0], points[end, 0]],
            [points[start, 1], points[end, 1]],
            "r-",
            linewidth=2,
        )

    # Plot triangulation
    if "triangles" in triangulation:
        for triangle_indices in triangulation["triangles"]:
            triangle_points = triangulation["vertices"][triangle_indices]
            # Close the triangle by repeating the first point
            plot_points = np.vstack((triangle_points, triangle_points[0]))
            plt.plot(plot_points[:, 0], plot_points[:, 1], "b-", linewidth=0.5)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.savefig("triangulation.png")


# Example usage:
if __name__ == "__main__":
    # Example points and edges
    pts = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64
    )

    edges = np.array(
        [
            [0, 1],  # bottom edge
            [1, 2],  # right edge
            [2, 3],  # top edge
            [3, 0],  # left edge
        ],
        dtype=np.int32,
    )

    try:
        # Create triangulation
        result = create_constrained_triangulation(pts, edges)

        # Plot results
        if result:
            plot_triangulation(pts, edges, result)
        else:
            print("Triangulation failed")

    except Exception as e:
        print(f"Error during triangulation: {e}")
