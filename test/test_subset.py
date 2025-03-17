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

quad_hex_grid_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'grid.nc'
quad_hex_data_path = current_path / 'meshfiles' / "ugrid" / "quad-hexagon" / 'data.nc'

def test_repr():
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)

    # grid repr
    grid_repr = uxds.uxgrid.subset.__repr__()
    assert "bounding_box" in grid_repr
    assert "bounding_circle" in grid_repr
    assert "nearest_neighbor" in grid_repr

    # data array repr
    da_repr = uxds['t2m'].subset.__repr__()
    assert "bounding_box" in da_repr
    assert "bounding_circle" in da_repr
    assert "nearest_neighbor" in da_repr


def test_grid_face_isel():
    for grid_path in GRID_PATHS:
        grid = ux.open_grid(grid_path)

        grid_contains_edge_node_conn = "edge_node_connectivity" in grid._ds

        face_indices = [0, 1, 2, 3, 4]
        for n_max_faces in range(1, len(face_indices)):
            grid_subset = grid.isel(n_face=face_indices[:n_max_faces])
            assert grid_subset.n_face == n_max_faces
            if not grid_contains_edge_node_conn:
                assert "edge_node_connectivity" not in grid_subset._ds

        face_indices = [0, 1, 2, grid.n_face]
        with pytest.raises(IndexError):
            grid_subset = grid.isel(n_face=face_indices)
            if not grid_contains_edge_node_conn:
                assert "edge_node_connectivity" not in grid_subset._ds


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
                                                   bbox[1],)

            grid_subset_antimeridian = grid.subset.bounding_box(
                bbox_antimeridian[0], bbox_antimeridian[1])


def test_uxda_isel():
    uxds = ux.open_dataset(GRID_PATHS[0], DATA_PATHS[0])

    sub = uxds['bottomDepth'].isel(n_face=[1, 2, 3])

    assert len(sub) == 3


def test_uxda_isel_with_coords():
    uxds = ux.open_dataset(GRID_PATHS[0], DATA_PATHS[0])
    uxds = uxds.assign_coords({"lon_face": uxds.uxgrid.face_lon})
    sub = uxds['bottomDepth'].isel(n_face=[1, 2, 3])

    assert "lon_face" in sub.coords
    assert len(sub.coords['lon_face']) == 3


def test_inverse_indices():
    grid = ux.open_grid(GRID_PATHS[0])

    # Test nearest neighbor subsetting
    coord = [0, 0]
    subset = grid.subset.nearest_neighbor(coord, k=1, element="face centers", inverse_indices=True)

    assert subset.inverse_indices is not None

    # Test bounding box subsetting
    box = [(-10, 10), (-10, 10)]
    subset = grid.subset.bounding_box(box[0], box[1], inverse_indices=True)

    assert subset.inverse_indices is not None

    # Test bounding circle subsetting
    center_coord = [0, 0]
    subset = grid.subset.bounding_circle(center_coord, r=10, element="face centers", inverse_indices=True)

    assert subset.inverse_indices is not None

    # Ensure code raises exceptions when the element is edges or nodes or inverse_indices is incorrect
    assert pytest.raises(Exception, grid.subset.bounding_circle, center_coord, r=10, element="edge centers", inverse_indices=True)
    assert pytest.raises(Exception, grid.subset.bounding_circle, center_coord, r=10, element="nodes", inverse_indices=True)
    assert pytest.raises(ValueError, grid.subset.bounding_circle, center_coord, r=10, element="face center", inverse_indices=(['not right'], True))

    # Test isel directly
    subset = grid.isel(n_face=[1], inverse_indices=True)
    assert subset.inverse_indices.face.values == 1


def test_da_subset():
    uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)

    res1 = uxds['t2m'].subset.bounding_box(lon_bounds=(-10, 10), lat_bounds=(-10, 10))
    res2 = uxds['t2m'].subset.bounding_circle(center_coord=(0,0), r=10)
    res3 = uxds['t2m'].subset.nearest_neighbor(center_coord=(0, 0), k=4)

    assert len(res1) == len(res2) == len(res3) == 4
