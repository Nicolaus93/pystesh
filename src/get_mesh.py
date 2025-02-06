from argparse import ArgumentParser

from OCP.OCP.IFSelect import IFSelect_ReturnStatus
from OCP.OCP.STEPControl import STEPControl_Reader

from src.get_edges import Edge, get_edge_loops, get_edges
import numpy as np
import polyscope as ps
from src.utils import project_points_to_2d


def get_visualization(face_edge_map: dict[int, list[Edge]], face_idx: int) -> None:
    if face_idx not in face_edge_map:
        raise IndexError(f"Face index {face_idx} not in face edge map.")

    edges = face_edge_map[face_idx]
    edge_loops = get_edge_loops(edges)

    # visualize
    loop_points = []
    for loop in edge_loops:
        # for edge, reverse in loop:
        #     if reverse:
        #         print(edge.last_p, edge.first_p)
        #     else:
        #         print(edge.first_p, edge.last_p)

        all_points = []
        for edge, reverse in loop:
            points = edge.sample_points(10)
            if reverse:
                points = points[::-1]
            all_points.append(points[:-1])  # last point is first of the next edge
        loop_points.append(np.vstack(all_points))
        break

    ps.init()
    for idx, points in enumerate(loop_points):
        edges = np.array(
            [(i, i + 1) for i in range(len(points) - 1)] + [(len(points) - 1, 0)]
        )
        ps.register_curve_network(f"loop_{idx}", points, edges)
    ps.show()

    projected, to_2d, normal, centroid = project_points_to_2d(loop_points)

    return


def run(step_file: str) -> None:
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError("Error reading file")

    reader.TransferRoots()
    shape = reader.OneShape()
    face_edge_map = get_edges(shape)
    print(f"Found {len(face_edge_map)} faces.")
    get_visualization(face_edge_map, 3)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filename", help="input step file")
    args = parser.parse_args()
    run(args.filename)
