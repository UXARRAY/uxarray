import os
from unittest import TestCase
from pathlib import Path
import numpy.testing as nt

import uxarray as ux
import xarray as xr
import numpy as np

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestAPI(TestCase):
    geoflow_data_path = current_path / "meshfiles" / "ugrid" / "geoflow-small"
    gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
    geoflow_data_v1 = geoflow_data_path / "v1.nc"
    geoflow_data_v2 = geoflow_data_path / "v2.nc"
    geoflow_data_v3 = geoflow_data_path / "v3.nc"

    gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"
    dsfiles_mf_ne30 = str(
        current_path) + "/meshfiles/ugrid/outCSne30/outCSne30_*.nc"

    def test_open_geoflow_dataset(self):
        """Loads a single dataset with its grid topology file using uxarray's
        open_dataset call."""

        # Paths to Data Variable files
        data_paths = [
            self.geoflow_data_v1, self.geoflow_data_v2, self.geoflow_data_v3
        ]

        uxds_v1 = ux.open_dataset(self.gridfile_geoflow, data_paths[0])

        # Ideally uxds_v1.uxgrid should NOT be None
        with self.assertRaises(AssertionError):
            nt.assert_equal(uxds_v1.uxgrid, None)

    def test_open_dataset(self):
        """Loads a single dataset with its grid topology file using uxarray's
        open_dataset call."""

        uxds_var2_ne30 = ux.open_dataset(self.gridfile_ne30,
                                         self.dsfile_var2_ne30)

        nt.assert_equal(uxds_var2_ne30.uxgrid.node_lon.size,
                        constants.NNODES_outCSne30)
        nt.assert_equal(len(uxds_var2_ne30.uxgrid._ds.data_vars),
                        constants.DATAVARS_outCSne30)
        nt.assert_equal(uxds_var2_ne30.source_datasets,
                        str(self.dsfile_var2_ne30))

    def test_open_mf_dataset(self):
        """Loads multiple datasets with their grid topology file using
        uxarray's open_dataset call."""

        uxds_mf_ne30 = ux.open_mfdataset(self.gridfile_ne30,
                                         self.dsfiles_mf_ne30)

        nt.assert_equal(uxds_mf_ne30.uxgrid.node_lon.size,
                        constants.NNODES_outCSne30)
        nt.assert_equal(len(uxds_mf_ne30.uxgrid._ds.data_vars),
                        constants.DATAVARS_outCSne30)

        nt.assert_equal(uxds_mf_ne30.source_datasets, self.dsfiles_mf_ne30)

    def test_open_grid(self):
        """Loads only a grid topology file using uxarray's open_grid call."""
        uxgrid = ux.open_grid(self.gridfile_geoflow)

        nt.assert_almost_equal(uxgrid.calculate_total_face_area(),
                               constants.MESH30_AREA,
                               decimal=3)

    def test_copy_dataset(self):
        """Loads a single dataset with its grid topology file using uxarray's
        open_dataset call and make a copy of the object."""

        uxds_var2_ne30 = ux.open_dataset(self.gridfile_ne30,
                                         self.dsfile_var2_ne30)

        # make a shallow and deep copy of the dataset object
        uxds_var2_ne30_copy_deep = uxds_var2_ne30.copy(deep=True)
        uxds_var2_ne30_copy = uxds_var2_ne30.copy(deep=False)

        # Ideally uxds_var2_ne30_copy.uxgrid should NOT be None
        with self.assertRaises(AssertionError):
            nt.assert_equal(uxds_var2_ne30_copy.uxgrid, None)

        # Check that the copy is a shallow copy
        assert (uxds_var2_ne30_copy.uxgrid is uxds_var2_ne30.uxgrid)
        assert (uxds_var2_ne30_copy.uxgrid == uxds_var2_ne30.uxgrid)

        # Check that the deep copy is a deep copy
        assert (uxds_var2_ne30_copy_deep.uxgrid == uxds_var2_ne30.uxgrid)
        assert (uxds_var2_ne30_copy_deep.uxgrid is not uxds_var2_ne30.uxgrid)

    def test_copy_dataarray(self):
        """Loads an unstructured grid and data using uxarray's open_dataset
        call and make a copy of the dataarray object."""

        # Paths to Data Variable files
        data_paths = [
            self.geoflow_data_v1, self.geoflow_data_v2, self.geoflow_data_v3
        ]

        uxds_v1 = ux.open_dataset(self.gridfile_geoflow, data_paths[0])

        # get the uxdataarray object
        v1_uxdata_array = uxds_v1['v1']

        # make a shallow and deep copy of the dataarray object
        v1_uxdata_array_copy_deep = v1_uxdata_array.copy(deep=True)
        v1_uxdata_array_copy = v1_uxdata_array.copy(deep=False)

        # Check that the copy is a shallow copy
        assert (v1_uxdata_array_copy.uxgrid is v1_uxdata_array.uxgrid)
        assert (v1_uxdata_array_copy.uxgrid == v1_uxdata_array.uxgrid)

        # Check that the deep copy is a deep copy
        assert (v1_uxdata_array_copy_deep.uxgrid == v1_uxdata_array.uxgrid)
        assert (v1_uxdata_array_copy_deep.uxgrid is not v1_uxdata_array.uxgrid)

    def test_open_dataset_grid_kwargs(self):
        """Drops ``Mesh2_face_nodes`` from the inputted grid file using
        ``grid_kwargs``"""

        with self.assertRaises(ValueError):
            # attempt to open a dataset after dropping face nodes should raise a KeyError
            uxds = ux.open_dataset(
                self.gridfile_ne30,
                self.dsfile_var2_ne30,
                grid_kwargs={"drop_variables": "Mesh2_face_nodes"})
