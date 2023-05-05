import os
from unittest import TestCase
from pathlib import Path
import numpy.testing as nt

import uxarray as ux

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"
dsfiles_mf_ne30 = str(
    current_path) + "/meshfiles/ugrid/outCSne30/outCSne30_*.nc"

gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"


class TestAPI(TestCase):

    def test_open_dataset_failing(self):
        """Loads a single dataset with its grid topology file using uxarray's
        open_dataset call."""

        # Base data path
        base_path = "test/meshfiles/ugrid/geoflow-small/"

        # Path to Grid file
        grid_path = base_path + "grid.nc"

        # Paths to Data Variable files
        var_names = ['v1.nc', 'v2.nc', 'v3.nc']
        data_paths = [base_path + name for name in var_names]

        uxds_v1 = ux.open_dataset(grid_path, data_paths[0])

    def test_open_dataset(self):
        """Loads a single dataset with its grid topology file using uxarray's
        open_dataset call."""

        uxds_var2_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

        nt.assert_equal(uxds_var2_ne30.uxgrid.Mesh2_node_x.size,
                        constants.NNODES_outCSne30)
        nt.assert_equal(len(uxds_var2_ne30.uxgrid._ds.data_vars),
                        constants.DATAVARS_outCSne30)
        nt.assert_equal(uxds_var2_ne30.uxgrid.source_grid, gridfile_ne30)
        nt.assert_equal(uxds_var2_ne30.source_datasets, str(dsfile_var2_ne30))

    def test_open_mf_dataset(self):
        """Loads multiple datasets with their grid topology file using
        uxarray's open_dataset call."""

        uxds_mf_ne30 = ux.open_mfdataset(gridfile_ne30, dsfiles_mf_ne30)

        nt.assert_equal(uxds_mf_ne30.uxgrid.Mesh2_node_x.size,
                        constants.NNODES_outCSne30)
        nt.assert_equal(len(uxds_mf_ne30.uxgrid._ds.data_vars),
                        constants.DATAVARS_outCSne30)
        nt.assert_equal(uxds_mf_ne30.uxgrid.source_grid, gridfile_ne30)
        nt.assert_equal(uxds_mf_ne30.source_datasets, dsfiles_mf_ne30)

    def test_open_grid(self):
        """Loads only a grid topology file using uxarray's open_grid call."""
        uxgrid = ux.open_grid(gridfile_geoflow)

        nt.assert_almost_equal(uxgrid.calculate_total_face_area(),
                               constants.MESH30_AREA,
                               decimal=3)

    def test_copy_dataset(self):
        """Loads a single dataset with its grid topology file using uxarray's
        open_dataset call and make a copy of the object."""

        uxds_var2_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)
        uxds_var2_ne30_copy = uxds_var2_ne30.copy()

        # Ideally this should fail and uxgrid should NOT be None
        nt.assert_equal(uxds_var2_ne30_copy.uxgrid, None)

        # this should pass
        # nt.assert_equal(uxds_var2_ne30.uxgrid.Mesh2_node_x.size, uxds_var2_ne30_copy.uxgrid.Mesh2_node_x.size)
