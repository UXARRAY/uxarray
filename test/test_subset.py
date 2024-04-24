import uxarray as ux
import os

import pytest

from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

GRID_PATHS = [
    current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc',
    current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc",
    current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
]

DATA_PATHS = [
    current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc',
    current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc",
    current_path / "meshfiles" / "ugrid" / "outCSne30" / "var2.nc"
]


def test_grid_face_isel():
    for grid_path in GRID_PATHS:
        grid = ux.open_grid(grid_path)

        face_indices = [0, 1, 2, 3, 4]
        for n_max_faces in range(1, len(face_indices)):
            grid_subset = grid.isel(n_face=face_indices[:n_max_faces])
            assert grid_subset.n_face == n_max_faces

        face_indices = [0, 1, 2, grid.n_face]
        with pytest.raises(IndexError):
            grid_subset = grid.isel(n_face=face_indices)


def test_grid_node_isel():
    for grid_path in GRID_PATHS:
        grid = ux.open_grid(grid_path)

        node_indices = [0, 1, 2, 3, 4]
        for n_max_nodes in range(1, len(node_indices)):
            grid_subset = grid.isel(n_node=node_indices[:n_max_nodes])
            assert grid_subset.n_node >= n_max_nodes

        face_indices = [0, 1, 2, grid.n_node]
        with pytest.raises(IndexError):
            grid_subset = grid.isel(n_face=face_indices)


def test_grid_nn_subset():
    coord_locs = [[0, 0], [-180, 0], [180, 0], [0, 90], [0, -90]]

    for grid_path in GRID_PATHS:
        grid = ux.open_grid(grid_path)

        # corner-nodes
        ks = [1, 2, grid.n_node - 1]
        for coord in coord_locs:
            for k in ks:
                grid_subset = grid.subset.nearest_neighbor(coord,
                                                           k,
                                                           element="nodes")
                assert grid_subset.n_node >= k

        # face-centers
        ks = [1, 2, grid.n_face - 1]
        for coord in coord_locs:
            for k in ks:
                grid_subset = grid.subset.nearest_neighbor(
                    coord, k, "face centers")

                assert grid_subset.n_face == k
                assert isinstance(grid_subset, ux.Grid)


def test_grid_bounding_circle_subset():
    coord_locs = [[0, 0], [-180, 0], [180, 0], [0, 90], [0, -90]]
    rs = [45, 90, 180]

    for grid_path in GRID_PATHS:
        grid = ux.open_grid(grid_path)
        for element in ["nodes", "face centers"]:
            for coord in coord_locs:
                for r in rs:
                    grid_subset = grid.subset.bounding_circle(coord, r, element)

                    assert isinstance(grid_subset, ux.Grid)


def test_grid_bounding_box_subset():
    bbox = [(-10, 10), (-10, 10)]
    bbox_antimeridian = [(-170, 170), (-45, 45)]

    for element in ["nodes", "face centers"]:
        for grid_path in GRID_PATHS:
            grid = ux.open_grid(grid_path)

            grid_subset = grid.subset.bounding_box(bbox[0],
                                                   bbox[1],
                                                   element=element)

            grid_subset_antimeridian = grid.subset.bounding_box(
                bbox_antimeridian[0], bbox_antimeridian[1], element=element)
