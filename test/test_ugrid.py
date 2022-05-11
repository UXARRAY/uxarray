import os
import numpy as np
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux
from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestUgrid(TestCase):

    def test_read_ugrid(self):
        """Reads a ugrid file."""

        ug_filename1 = current_path / "meshfiles" / "outCSne30.ug"
        ug_filename2 = current_path / "meshfiles" / "outRLL1deg.ug"
        ug_filename3 = current_path / "meshfiles" / "ov_RLL10deg_CSne4.ug"

        ux_grid1 = ux.Grid(str(ug_filename1))
        ux_grid2 = ux.Grid(str(ug_filename2))
        ux_grid3 = ux.Grid(str(ug_filename3))

        # num_node_var = ux.ugrid_vars["nMesh2_node"]
        ux_grid1_node_x_var = ux_grid1.ds_var_names["Mesh2_node_x"]
        ux_grid2_node_x_var = ux_grid2.ds_var_names["Mesh2_node_x"]
        ux_grid3_node_x_var = ux_grid3.ds_var_names["Mesh2_node_x"]

        assert (
            ux_grid1.ds[ux_grid1_node_x_var].size == constants.NNODES_outCSne30)
        assert (ux_grid2.ds[ux_grid2_node_x_var].size ==
                constants.NNODES_outRLL1deg)
        assert (ux_grid3.ds[ux_grid3_node_x_var].size ==
                constants.NNODES_ov_RLL10deg_CSne4)

    def test_write_ugrid(self):
        """Read an exodus file and writes a ugrid file."""

        exo2_filename = current_path / "meshfiles" / "outCSne8.g"
        ux_grid = ux.Grid(str(exo2_filename))
        outfile = current_path / "write_test_outCSne8.ug"
        ux_grid.write(str(outfile))
