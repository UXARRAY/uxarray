import os
from unittest import TestCase
from pathlib import Path

import uxarray as ux
from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

uds1_name = current_path / "meshfiles" / "ov_RLL10deg_CSne4.ug"
uds2_name = current_path / "meshfiles" / "outCSne8.g"
uds3_name = current_path / "meshfiles" / "outCSne30.ug"
uds3_data_name1 = current_path / "meshfiles" / "outCSne30_vortex.nc"
uds3_data_name2 = current_path / "meshfiles" / "outCSne30_var2.ug"

mesh_name = current_path / "meshfiles" / "mesh.nc"
outmesh = current_path / "meshfiles" / "mesh_out.ug"


class TestDataset(TestCase):

    def test_open_grid_only(self):
        """Loads grid files of different formats using uxarray's open_dataset
        call."""

        ux_ds1 = ux.open_dataset(uds1_name)
        ux_ds2 = ux.open_dataset(uds2_name)
        ux_ds3 = ux.open_dataset(uds3_name)

        # get node names for each grid object
        ux_ds1_node_x_var = ux_ds1.ds_var_names["Mesh2_node_x"]
        ux_ds2_node_x_var = ux_ds2.ds_var_names["Mesh2_node_x"]
        ux_ds3_node_x_var = ux_ds3.ds_var_names["Mesh2_node_x"]

        assert (ux_ds1.ds[ux_ds1_node_x_var].size ==
                constants.NNODES_ov_RLL10deg_CSne4)
        assert (ux_ds2.ds[ux_ds2_node_x_var].size == constants.NNODES_outCSne8)
        assert (ux_ds3.ds[ux_ds3_node_x_var].size == constants.NNODES_outCSne30)

        assert (len(ux_ds3.ds.data_vars) == constants.DATAVARS_outCSne30)

    def test_open_single_dataset(self):
        """Loads one grid and data file using uxarray's open_dataset call."""

        uds3 = ux.open_dataset(uds3_name, uds3_data_name1)

        n3 = uds3.ds_var_names["Mesh2_node_x"]

        assert (uds3.ds[n3].size == constants.NNODES_outCSne30)
        assert (len(uds3.ds.data_vars) == constants.DATAVARS_outCSne30 + 1)

    def test_open_multiple_dataset(self):
        """Loads a grid file and two data files of different formats using
        uxarray's open_dataset call."""

        uds3 = ux.open_dataset(uds3_name, uds3_data_name1, uds3_data_name2)

        n3 = uds3.ds_var_names["Mesh2_node_x"]

        assert (uds3.ds[n3].size == constants.NNODES_outCSne30)
        assert (len(uds3.ds.data_vars) == constants.DATAVARS_outCSne30 + 2)

    def test_open_non_mesh2_write_exodus(self):
        """Loads grid files of different formats using uxarray's open_dataset
        call."""
        grid = ux.open_dataset(mesh_name)
        grid.write(outmesh)
