import os
import numpy as np

from unittest import TestCase
from pathlib import Path

import uxarray as ux

from uxarray.core.dataset import UxDataset
from uxarray.core.dataarray import UxDataArray

from uxarray.remap import _nearest_neighbor

current_path = Path(os.path.dirname(os.path.realpath(__file__)))
gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
gridfile_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
dsfile_vortex_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"
gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
dsfile_v1_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"
dsfile_v2_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v2.nc"
dsfile_v3_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v3.nc"


class TestNearestNeighborRemap(TestCase):
    """Tests for nearest neighbor remapping."""

    def test_remap_to_same_grid_corner_nodes(self):
        """Test remapping to the same dummy 3-vertex grid.

        Corner nodes case.
        """
        # single triangle with point on antimeridian
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

        assert np.array_equal(source_data_single_dim, destination_single_data)
        assert np.array_equal(source_data_multi_dim, destination_multi_data)

    def test_remap_to_corner_nodes_cartesian(self):
        """Test remapping to the same dummy 3-vertex grid, using cartesian
        coordinates.

        Corner nodes case.
        """

        # single triangle
        source_verts = np.array([(0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
                                 (1.0, 0.0, 0.0)])
        source_data_single_dim = [1.0, 2.0, 3.0]

        # open the source and destination grids
        source_grid = ux.open_grid(source_verts)
        destination_grid = ux.open_grid(source_verts)

        # create the destination data using the nearest neighbor function
        destination_data = _nearest_neighbor(source_grid,
                                             destination_grid,
                                             source_data_single_dim,
                                             remap_to="nodes",
                                             coord_type="cartesian")

        # assert that the source and destination data are the same
        assert np.array_equal(source_data_single_dim, destination_data)

    def test_nn_remap(self):
        """Test nearest neighbor remapping.

        Steps:
        1. Open a grid and a dataset,
        2. Open the grid to remap dataset in 1
        3. Remap the dataset in 1 to the grid in 2
        """
        # TODO; write better test
        uxds = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)

        uxgrid = ux.open_grid(gridfile_ne30)

        uxda = uxds['v1']
        out_da = uxda.nearest_neighbor_remap(destination_obj=uxgrid)
        pass

    def test_remap_return_types(self):
        """Tests the return type of the `UxDataset` and `UxDataArray`
        implementations of Nearest Neighbor Remapping."""
        source_data_paths = [
            dsfile_v1_geoflow, dsfile_v2_geoflow, dsfile_v3_geoflow
        ]
        source_uxds = ux.open_mfdataset(gridfile_geoflow, source_data_paths)
        destination_uxds = ux.open_dataset(gridfile_CSne30,
                                           dsfile_vortex_CSne30)

        remap_uxda_to_grid = source_uxds['v1'].nearest_neighbor_remap(
            destination_uxds.uxgrid)

        assert isinstance(remap_uxda_to_grid, UxDataArray)

        remap_uxda_to_uxda = source_uxds['v1'].nearest_neighbor_remap(
            destination_uxds['psi'])

        # Dataset with two vars: original "psi" and remapped "v1"
        assert isinstance(remap_uxda_to_uxda, UxDataset)
        assert len(remap_uxda_to_uxda.data_vars) == 2

        remap_uxda_to_uxds = source_uxds['v1'].nearest_neighbor_remap(
            destination_uxds)

        # Dataset with two vars: original "psi" and remapped "v1"
        assert isinstance(remap_uxda_to_uxds, UxDataset)
        assert len(remap_uxda_to_uxds.data_vars) == 2

        remap_uxds_to_grid = source_uxds.nearest_neighbor_remap(
            destination_uxds.uxgrid)

        # Dataset with three vars: remapped "v1, v2, v3"
        assert isinstance(remap_uxds_to_grid, UxDataset)
        assert len(remap_uxds_to_grid.data_vars) == 3

        remap_uxds_to_uxda = source_uxds.nearest_neighbor_remap(
            destination_uxds['psi'])

        # Dataset with four vars: original "psi" and remapped "v1, v2, v3"
        assert isinstance(remap_uxds_to_uxda, UxDataset)
        assert len(remap_uxds_to_uxda.data_vars) == 4

        remap_uxds_to_uxds = source_uxds.nearest_neighbor_remap(
            destination_uxds)

        # Dataset with four vars: original "psi" and remapped "v1, v2, v3"
        assert isinstance(remap_uxds_to_uxds, UxDataset)
        assert len(remap_uxds_to_uxds.data_vars) == 4
