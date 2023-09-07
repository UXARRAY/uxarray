import os
import numpy as np
import numpy.testing as nt
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

from uxarray.remap import nearest_neighbor

current_path = Path(os.path.dirname(os.path.realpath(__file__)))
gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
dsfile_v1_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"

class TestNearestNeighborRemap(TestCase):
""" Tests for nearest neighbor remapping. """

    def test_remap_to_same_grid_corner_nodes(self):
        """ Test remapping to the same grid. Corner nodes case. """
        # single triangle with point on antimeridian
        source_verts = np.array([(0.0, 90.0), (-180, 0.0), (0.0, -90)])
        source_data_single_dim = [1.0, 2.0, 3.0]
        source_grid = ux.open_grid(source_verts)
        destination_grid = ux.open_grid(source_verts)

        destination_data = nearest_neighbor(source_grid,
                                            destination_grid,
                                            source_data_single_dim,
                                            destination_data_mapping="nodes")

        np.array_equal(source_data_single_dim, destination_data)

        source_data_multi_dim = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                                          [7.0, 8.0, 9.0]])

        destination_data = nearest_neighbor(source_grid,
                                            destination_grid,
                                            source_data_multi_dim,
                                            destination_data_mapping="nodes")

        np.array_equal(source_data_multi_dim, destination_data)

        pass


    def test_nn_remap(self):
        """ Test nearest neighbor remapping. 
        Steps: 
        1. Open a grid and a dataset, 
        2. Open the grid to remap dataset in 1
        3. Remap the dataset in 1 to the grid in 2"""
        # TODO; write better test
        uxds = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)

        uxgrid = ux.open_grid(gridfile_ne30)

        uxda = uxds['v1']
        out_da = uxda.nearest_neighbor_remap(destination_obj=uxgrid)
        pass
