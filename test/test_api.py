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

grid_path = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
grid_var2_path = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"
grid_mf_paths = str(current_path) + "/meshfiles/ugrid/outCSne30/outCSne30_*.nc"

mesh_path = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"


class TestAPI(TestCase):

    def test_open_dataset(self):
        """Loads a single dataset with its grid topology file
        using uxarray's open_dataset call."""

        ux_ds3 = ux.open_dataset(grid_path, grid_var2_path)

        nt.assert_equal(ux_ds3.uxgrid.Mesh2_node_x.size,
                        constants.NNODES_outCSne30)
        nt.assert_equal(len(ux_ds3.uxgrid._ds.data_vars),
                        constants.DATAVARS_outCSne30)
        nt.assert_equal(ux_ds3.uxgrid.source_grid,
                        grid_path)
        nt.assert_equal(ux_ds3.source_datasets,
                        str(grid_var2_path))

    def test_open_mf_dataset(self):
        """Loads multiple datasets with their grid topology file
        using uxarray's open_dataset call."""

        uds3 = ux.open_mfdataset(grid_path, grid_mf_paths)

        nt.assert_equal(uds3.uxgrid.Mesh2_node_x.size,
                        constants.NNODES_outCSne30)
        nt.assert_equal(len(uds3.uxgrid._ds.data_vars),
                        constants.DATAVARS_outCSne30)
        nt.assert_equal(uds3.uxgrid.source_grid,
                        grid_path)
        nt.assert_equal(uds3.source_datasets,
                        grid_mf_paths)

    def test_open_grid(self):
        """Loads only a grid topology file using uxarray's open_grid call."""
        uxgrid = ux.open_grid(mesh_path)

        nt.assert_almost_equal(uxgrid.calculate_total_face_area(), constants.MESH30_AREA, decimal=3)
