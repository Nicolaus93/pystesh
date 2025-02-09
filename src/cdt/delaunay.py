import numpy as np


def triangulate(points):
    """
    1. (Normalize coordinates of points.) Scale the coordinates of the points so that they all lie between
    0 and 1. This scaling should be uniform so that the relative positions of the points are unchanged.

    2. (Sort points into bins.) Cover the region to be triangulated by a rectangular grid so that each
    rectangle (or bin) contains roughly sqrt(N) points. Label the bins so that consecutive bins are adjacent to one
    another, for example by using column-by-column or row-by-row ordering, and then allocate each point to
    its appropriate bin. Sort the list of points in ascending sequence of their bin numbers so that consecutive
    points are grouped together in the x-y plane.

    3. (Establish the supertriangle.) Select three dummy points to form a supertriangle that completely encompasses all
    of the points to be triangulated. This supertriangle initially defines a Delaunay triangulation which is comprised
    of a single triangle. Its vertices are defined in terms of normalized coordinates and are usually located at a
    considerable distance from the window which encloses the set of points.

    # 4. (Loop over each point.) For each point P in the
    # list of sorted points, do steps 5-7.
    # 5. (Insert new point in triangulation.)
    # Find an
    # existing triangle which encloses P. Delete this triangle
    # and form three new triangles by connecting P to each
    # of its vertices. The net gain in the total number of
    # triangles after this stage is two. The searching algor-
    # ithm of Lawson [I] may be used to locate the triangle
    # containing P efficiently. Because of the bin sorting
    # phase, only a few triangles need to be examined if the
    # search is initiated in the triangle which has been
    # formed most recently.
    # 6. (Initialize stack.) Place all triangles which are
    # adjacent to the edges opposite P on a last-in-first-out
    # stack. There is a maximum of three such triangles.
    # 7. (Restore Delaunay t~angulation.)
    # While the
    # stack of triangles is not empty, execute Lawson’s
    # swapping scheme, as defined by steps 7.1-7.3.
    # 7.1. Remove a triangle which is opposite P
    # from the top of the stack.
    # 7.2. If P is outside (or on) the circumcircle for
    # this triangle, return to step 7.1. Else, the
    # triangle containing P as a vertex and the
    # unstacked triangle form a convex quadri-
    # lateral whose diagonal is drawn in the
    # wrong direction. Swap this diagonal so that
    # two old triangles are replaced by two new
    # triangles and the structure of the Delaunay
    # triangulation is locally restored.
    # 7.3. Place any triangles which are now opposite
    # P on the stack.

    """
    # Find the min and max along each axis (x and y)
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Normalize the points to [0, 1] range
    normalized_points = (points - min_vals) / (max_vals - min_vals)

    # Step 2: sort the points into bins
    grid_size = 4
    bins = np.linspace(0, 1, grid_size + 1)  # Bin edges along x-axis
    x_idxs = np.digitize(normalized_points[:, 0], bins) - 1
    y_idxs = np.digitize(normalized_points[:, 1], bins) - 1
    bin_numbers = y_idxs * grid_size + x_idxs

    # Step 5: Sort the points by their bin numbers
    sorted_indices = np.argsort(bin_numbers)
    sorted_bin_numbers = bin_numbers[sorted_indices]

    print(sorted_indices)
    print(normalized_points[sorted_indices])
    print(sorted_bin_numbers)


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
