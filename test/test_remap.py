import os
import numpy as np
import numpy.testing as nt
import pytest
from pathlib import Path

import uxarray as ux
from uxarray.core.dataset import UxDataset
from uxarray.core.dataarray import UxDataArray
from uxarray.remap.inverse_distance_weighted import _inverse_distance_weighted_remap
from uxarray.remap.nearest_neighbor import _nearest_neighbor

current_path = Path(os.path.dirname(os.path.realpath(__file__)))
gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
gridfile_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
dsfile_vortex_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"
gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
dsfile_v1_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"
dsfile_v2_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v2.nc"
dsfile_v3_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v3.nc"
mpasfile_QU = current_path / "meshfiles" / "mpas" / "QU" / "mesh.QU.1920km.151026.nc"


def test_remap_to_same_grid_corner_nodes():
    """Test remapping to the same dummy 3-vertex grid. Corner nodes case."""
    source_verts = np.array([(0.0, 90.0), (-180, 0.0), (0.0, -90)])
    source_data_single_dim = [1.0, 2.0, 3.0]
    source_grid = ux.open_grid(source_verts)
    destination_grid = ux.open_grid(source_verts)

    destination_single_data = _nearest_neighbor(source_grid,
                                                destination_grid,
                                                source_data_single_dim,
                                                remap_to="nodes")

    source_data_multi_dim = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                                       [7.0, 8.0, 9.0]])

    destination_multi_data = _nearest_neighbor(source_grid,
                                               destination_grid,
                                               source_data_multi_dim,
                                               remap_to="nodes")

    nt.assert_array_equal(source_data_single_dim, destination_single_data)
    nt.assert_array_equal(source_data_multi_dim, destination_multi_data)


def test_remap_to_corner_nodes_cartesian():
    """Test remapping to the same dummy 3-vertex grid, using cartesian coordinates. Corner nodes case."""
    source_verts = np.array([(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    source_data_single_dim = [1.0, 2.0, 3.0]
    source_grid = ux.open_grid(source_verts)
    destination_grid = ux.open_grid(source_verts)

    destination_data = _nearest_neighbor(source_grid,
                                         destination_grid,
                                         source_data_single_dim,
                                         remap_to="nodes",
                                         coord_type="cartesian")

    nt.assert_array_equal(source_data_single_dim, destination_data)


def test_nn_remap():
    """Test nearest neighbor remapping."""
    uxds = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)
    uxgrid = ux.open_grid(gridfile_ne30)
    uxda = uxds['v1']
    out_da = uxda.remap.nearest_neighbor(destination_grid=uxgrid, remap_to="nodes")

    assert len(out_da) != 0


def test_remap_return_types():
    """Tests the return type of the `UxDataset` and `UxDataArray` implementations of Nearest Neighbor Remapping."""
    source_data_paths = [dsfile_v1_geoflow, dsfile_v2_geoflow, dsfile_v3_geoflow]
    source_uxds = ux.open_mfdataset(gridfile_geoflow, source_data_paths)
    destination_grid = ux.open_grid(gridfile_CSne30)

    remap_uxda_to_grid = source_uxds['v1'].remap.nearest_neighbor(destination_grid)

    assert isinstance(remap_uxda_to_grid, UxDataArray)

    remap_uxds_to_grid = source_uxds.remap.nearest_neighbor(destination_grid)

    assert isinstance(remap_uxds_to_grid, UxDataset)
    assert len(remap_uxds_to_grid.data_vars) == 3


def test_edge_centers_remapping():
    """Tests the ability to remap on edge centers using Nearest Neighbor Remapping."""
    source_grid = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)
    destination_grid = ux.open_grid(mpasfile_QU)

    remap_to_edge_centers_spherical = source_grid['v1'].remap.nearest_neighbor(destination_grid=destination_grid,
                                                                         remap_to="edge centers", coord_type='spherical')

    remap_to_edge_centers_cartesian = source_grid['v1'].remap.nearest_neighbor(destination_grid=destination_grid,
                                                                         remap_to="edge centers", coord_type='cartesian')

    assert remap_to_edge_centers_spherical._edge_centered()
    assert remap_to_edge_centers_cartesian._edge_centered()


def test_overwrite():
    """Tests that the remapping no longer overwrites the dataset."""
    source_grid = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)
    destination_dataset = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)

    remap_to_edge_centers = source_grid['v1'].remap.nearest_neighbor(destination_grid=destination_dataset.uxgrid,
                                                                     remap_to="face centers", coord_type='cartesian')

    assert not np.array_equal(destination_dataset['v1'], remap_to_edge_centers)


def test_source_data_remap():
    """Test the remapping of all source data positions."""
    source_uxds = ux.open_dataset(mpasfile_QU, mpasfile_QU)
    destination_grid = ux.open_grid(gridfile_geoflow)

    face_centers = source_uxds['latCell'].remap.nearest_neighbor(destination_grid=destination_grid, remap_to="nodes")
    nodes = source_uxds['latVertex'].remap.nearest_neighbor(destination_grid=destination_grid, remap_to="nodes")
    edges = source_uxds['angleEdge'].remap.nearest_neighbor(destination_grid=destination_grid, remap_to="nodes")

    assert len(face_centers.values) != 0
    assert len(nodes.values) != 0
    assert len(edges.values) != 0


def test_value_errors():
    """Tests the raising of value errors and warnings in the function."""
    source_uxds = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)
    source_uxds_2 = ux.open_dataset(mpasfile_QU, mpasfile_QU)
    destination_grid = ux.open_grid(gridfile_geoflow)

    with nt.assert_raises(ValueError):
        source_uxds['v1'].remap.nearest_neighbor(destination_grid=destination_grid, remap_to="test", coord_type='spherical')
    with nt.assert_raises(ValueError):
        source_uxds['v1'].remap.nearest_neighbor(destination_grid=destination_grid, remap_to="test", coord_type="cartesian")
    with nt.assert_raises(ValueError):
        source_uxds['v1'].remap.nearest_neighbor(destination_grid=destination_grid, remap_to="nodes", coord_type="test")
    with nt.assert_raises(ValueError):
        source_uxds_2['cellsOnCell'].remap.nearest_neighbor(destination_grid=destination_grid, remap_to="nodes")

def test_preserve_coordinates():
    """Tests if coordinates are preserved after remapping."""
    source_uxds = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)
    destination_grid = ux.open_grid(mpasfile_QU)

    res = source_uxds.remap.nearest_neighbor(destination_grid=destination_grid)

    assert "time" in res.coords


# Inverse Distance Weighted Remapping Tests
def test_remap_center_nodes():
    """Test remapping to center nodes."""
    dataset = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)
    destination_grid = ux.open_grid(gridfile_geoflow)

    data_on_face_centers = dataset['v1'].remap.inverse_distance_weighted(destination_grid, remap_to="face centers", power=6)

    assert not np.array_equal(dataset['v1'], data_on_face_centers)


def test_remap_nodes():
    """Test remapping to nodes."""
    dataset = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)
    destination_grid = ux.open_grid(gridfile_geoflow)

    data_on_nodes = dataset['v1'].remap.inverse_distance_weighted(destination_grid, remap_to="nodes")

    assert not np.array_equal(dataset['v1'], data_on_nodes)


def test_cartesian_remap_to_nodes():
    """Test remapping using cartesian coordinates."""
    source_verts = np.array([(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    source_data = [1.0, 2.0, 3.0]
    source_grid = ux.open_grid(source_verts)
    destination_grid = ux.open_grid(source_verts)

    destination_data_neighbors_2 = _inverse_distance_weighted_remap(source_grid, destination_grid, source_data, remap_to="nodes", coord_type="cartesian", k=3)
    destination_data_neighbors_1 = _inverse_distance_weighted_remap(source_grid, destination_grid, source_data, remap_to="nodes", coord_type="cartesian", k=2)

    assert not np.array_equal(destination_data_neighbors_1, destination_data_neighbors_2)


def test_remap_return_types_idw():
    """Tests the return type of the `UxDataset` and `UxDataArray` implementations of Inverse Distance Weighted."""
    source_data_paths = [dsfile_v1_geoflow, dsfile_v2_geoflow, dsfile_v3_geoflow]
    source_uxds = ux.open_mfdataset(gridfile_geoflow, source_data_paths)
    destination_grid = ux.open_grid(gridfile_CSne30)

    remap_uxda_to_grid = source_uxds['v1'].remap.inverse_distance_weighted(destination_grid, power=3, k=10)

    assert isinstance(remap_uxda_to_grid, UxDataArray)
    assert len(remap_uxda_to_grid) == 1

    remap_uxds_to_grid = source_uxds.remap.inverse_distance_weighted(destination_grid)

    assert isinstance(remap_uxds_to_grid, UxDataset)
    assert len(remap_uxds_to_grid.data_vars) == 3


def test_edge_remapping():
    """Tests the ability to remap on edge centers using Inverse Distance Weighted Remapping."""
    source_grid = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)
    destination_grid = ux.open_grid(mpasfile_QU)

    remap_to_edge_centers_spherical = source_grid['v1'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="edge centers", coord_type='spherical')
    remap_to_edge_centers_cartesian = source_grid['v1'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="edge centers", coord_type='cartesian')

    assert remap_to_edge_centers_spherical._edge_centered()
    assert remap_to_edge_centers_cartesian._edge_centered()


def test_overwrite_idw():
    """Tests that the remapping no longer overwrites the dataset."""
    source_grid = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)
    destination_dataset = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)

    remap_to_edge_centers = source_grid['v1'].remap.inverse_distance_weighted(destination_grid=destination_dataset.uxgrid, remap_to="face centers", coord_type='cartesian')

    assert not np.array_equal(destination_dataset['v1'], remap_to_edge_centers)


def test_source_data_remap_idw():
    """Test the remapping of all source data positions."""
    source_uxds = ux.open_dataset(mpasfile_QU, mpasfile_QU)
    destination_grid = ux.open_grid(gridfile_geoflow)

    face_centers = source_uxds['latCell'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="nodes")
    nodes = source_uxds['latVertex'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="nodes")
    edges = source_uxds['angleEdge'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="nodes")

    assert len(face_centers.values) != 0
    assert len(nodes.values) != 0
    assert len(edges.values) != 0


def test_value_errors_idw():
    """Tests the raising of value errors and warnings in the function."""
    source_uxds = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)
    source_uxds_2 = ux.open_dataset(mpasfile_QU, mpasfile_QU)
    destination_grid = ux.open_grid(gridfile_geoflow)

    with nt.assert_raises(ValueError):
        source_uxds['v1'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="nodes", k=1)

    with nt.assert_raises(ValueError):
        source_uxds['v1'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="nodes", k=source_uxds.uxgrid.n_node + 1)

    with nt.assert_raises(ValueError):
        source_uxds['v1'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="test", k=2, coord_type='spherical')
    with nt.assert_raises(ValueError):
        source_uxds['v1'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="test", k=2, coord_type="cartesian")

    with nt.assert_raises(ValueError):
        source_uxds['v1'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="nodes", k=2, coord_type="test")

    with nt.assert_raises(ValueError):
        source_uxds_2['cellsOnCell'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="nodes")

    with nt.assert_warns(UserWarning):
        source_uxds['v1'].remap.inverse_distance_weighted(destination_grid=destination_grid, remap_to="nodes", power=6)
