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
                                                           tree_type="nodes")
                assert grid_subset.n_node >= k

        # face-centers
        ks = [1, 2, grid.n_face - 1]
        for coord in coord_locs:
            for k in ks:
                grid_subset = grid.subset.nearest_neighbor(
                    coord, k, tree_type="face centers")
                assert grid_subset.n_face == k
