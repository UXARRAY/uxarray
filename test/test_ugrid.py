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

        tgrid1 = ux.Grid(str(ug_filename1))
        tgrid2 = ux.Grid(str(ug_filename2))
        tgrid3 = ux.Grid(str(ug_filename3))

        # num_node_var = ux.ugrid_vars["nMesh2_node"]
        n1 = tgrid1.var_names_dict["Mesh2_node_x"]
        n2 = tgrid2.var_names_dict["Mesh2_node_x"]
        n3 = tgrid3.var_names_dict["Mesh2_node_x"]

        assert (tgrid1.ds[n1].size == constants.NNODES_outCSne30)
        assert (tgrid2.ds[n2].size == constants.NNODES_outRLL1deg)
        assert (tgrid3.ds[n3].size == constants.NNODES_ov_RLL10deg_CSne4)

    def test_write_ugrid(self):
        """Read an exodus file and writes a ugrid file."""

        exo2_filename = current_path / "meshfiles" / "outCSne8.g"
        tgrid = ux.Grid(str(exo2_filename))
        outfile = current_path / "write_test_outCSne8.ug"
        tgrid.write(str(outfile))
