import os
from unittest import TestCase
from pathlib import Path

import uxarray as ux

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

grid1_path = current_path / "meshfiles" / "ov_RLL10deg_CSne4.ug"
grid2_path = current_path / "meshfiles" / "outCSne8.g"
grid3_path = current_path / "meshfiles" / "outCSne30.ug"
grid3_var1_path = current_path / "meshfiles" / "outCSne30_vortex.nc"
grid3_var2_path = current_path / "meshfiles" / "outCSne30_var2.ug"

mesh_path = current_path / "meshfiles" / "mesh.nc"
outmesh_path = current_path / "meshfiles" / "mesh_out.ug"


class TestDataset(TestCase):

    def test_open_grid_only(self):
        """Loads grid files of different formats using uxarray's open_dataset
        call."""

        ux_ds1 = ux.open_dataset(uds1_name)
        ux_ds2 = ux.open_dataset(uds2_name)
        ux_ds3 = ux.open_dataset(uds3_name)

        assert (ux_ds1.Mesh2_node_x.size == constants.NNODES_ov_RLL10deg_CSne4)
        assert (ux_ds2.Mesh2_node_x.size == constants.NNODES_outCSne8)
        assert (ux_ds3.Mesh2_node_x.size == constants.NNODES_outCSne30)

        assert (len(ux_ds3.ds.data_vars) == constants.DATAVARS_outCSne30)

        assert (ux_ds1.source_grid == uds1_name)
        assert (ux_ds1.source_datasets is None)

    def test_open_single_dataset(self):
        """Loads one grid and data file using uxarray's open_dataset call."""

        ux_ds3 = ux.open_dataset(grid3_var2_path, grid3_path)

        assert (ux_ds3.uxgrid.Mesh2_node_x.size == constants.NNODES_outCSne30)

        # Todo: Figure the failure with this
        # assert (len(ux_ds3.uxgrid.ds.data_vars) == constants.DATAVARS_outCSne30 + 1)

        # Todo: Figure how to handle such func calls
        # ux_ds3.uxgrid.integrate(ux_ds3.var2)
        #
        # ux_ds3.uxgrid.integrate()
        #
        # ux.integrate(ux_ds3)
        #
        # ux_ds3.var2.integrate()

        # Todo: Figure how to handle these
        # assert (ux_ds3.uxgrid.source_grid == grid3_path)
        # assert (len(ux_ds3.uxgrid.source_datasets) == 1)
        # assert (ux_ds3.uxgrid.source_datasets[0] == grid3_var1_path)

    def test_open_multiple_dataset(self):
        """Loads a grid file and two data files of different formats using
        uxarray's open_dataset call."""

        uds3 = ux.open_dataset(grid3_path, grid3_var1_path, grid3_var2_path)

        assert (uds3.Mesh2_node_x.size == constants.NNODES_outCSne30)
        assert (len(uds3.ds.data_vars) == constants.DATAVARS_outCSne30 + 2)

        assert (uds3.source_grid == grid3_path)
        assert (len(uds3.source_datasets) == 2)
        assert (uds3.source_datasets[0] == grid3_var1_path)
        assert (uds3.source_datasets[1] == grid3_var2_path)

    def test_open_non_mesh2_write_exodus(self):
        """Loads grid files of different formats using uxarray's open_dataset
        call."""
        grid = ux.open_dataset(mesh_path)
        grid.write(outmesh_path, "exodus")
