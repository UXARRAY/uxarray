import os
import xarray as xr

from unittest import TestCase
from pathlib import Path
import warnings

import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestUgrid(TestCase):

    def test_read_ugrid(self):
        """Reads a ugrid file."""

        ug_filename1 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
        ug_filename2 = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
        ug_filename3 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"

        xr_grid1 = xr.open_dataset(str(ug_filename1))
        xr_grid2 = xr.open_dataset(str(ug_filename2))
        xr_grid3 = xr.open_dataset(str(ug_filename3))

        ux_grid1 = ux.Grid(xr_grid1)
        ux_grid2 = ux.Grid(xr_grid2)
        ux_grid3 = ux.Grid(xr_grid3)

        assert (ux_grid1.Mesh2_node_x.size == constants.NNODES_outCSne30)
        assert (ux_grid2.Mesh2_node_x.size == constants.NNODES_outRLL1deg)
        assert (
            ux_grid3.Mesh2_node_x.size == constants.NNODES_ov_RLL10deg_CSne4)

    def test_read_ugrid_opendap(self):
        """Read an ugrid model from an OPeNDAP URL."""

        try:
            # make sure we can read the ugrid file from the OPeNDAP URL
            url = "http://test.opendap.org:8080/opendap/ugrid/NECOFS_GOM3_FORECAST.nc"
            xr_grid = xr.open_dataset(url, drop_variables="siglay")

        except OSError:
            # print warning and pass if we can't connect to the OPeNDAP server
            warnings.warn(f'Could not connect to OPeNDAP server: {url}')
            pass

        else:
            ugrid = ux.Grid(xr_grid)
            assert isinstance(getattr(ugrid, "Mesh2_node_x"), xr.DataArray)
            assert isinstance(getattr(ugrid, "Mesh2_node_y"), xr.DataArray)
            assert isinstance(getattr(ugrid, "Mesh2_face_nodes"), xr.DataArray)

    def test_encode_ugrid(self):
        """Read an Exodus dataset and encode that as a UGRID format."""

        exo2_filename = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
        xr_grid = xr.open_dataset(str(exo2_filename))
        ux_grid = ux.Grid(xr_grid)
        ux_grid.encode_as("ugrid")

    def test_standardized_dtype_and_fill(self):
        """Test to see if Mesh2_Face_Nodes uses the expected integer datatype
        and expected fill value as set in constants.py."""

        ug_filename1 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
        ug_filename2 = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
        ug_filename3 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"

        xr_grid1 = xr.open_dataset(str(ug_filename1))
        xr_grid2 = xr.open_dataset(str(ug_filename2))
        xr_grid3 = xr.open_dataset(str(ug_filename3))

        ux_grid1 = ux.Grid(xr_grid1)
        ux_grid2 = ux.Grid(xr_grid2)
        ux_grid3 = ux.Grid(xr_grid3)

        # check for correct dtype and fill value
        grids_with_fill = [ux_grid2]
        for grid in grids_with_fill:
            assert grid.Mesh2_face_nodes.dtype == INT_DTYPE
            assert grid.Mesh2_face_nodes._FillValue == INT_FILL_VALUE
            assert INT_FILL_VALUE in grid.Mesh2_face_nodes.values

        grids_without_fill = [ux_grid1, ux_grid3]
        for grid in grids_without_fill:
            assert grid.Mesh2_face_nodes.dtype == INT_DTYPE
            assert grid.Mesh2_face_nodes._FillValue == INT_FILL_VALUE

    def test_standardized_dtype_and_fill_dask(self):
        """Test to see if Mesh2_Face_Nodes uses the expected integer datatype
        and expected fill value as set in constants.py.

        with dask chunking
        """
        ug_filename = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
        xr_grid = xr.open_dataset(str(ug_filename), chunks={'nMesh2_node': 100})
        ux_grid = ux.Grid(xr_grid)
        assert ux_grid.Mesh2_face_nodes.dtype == INT_DTYPE
        assert ux_grid.Mesh2_face_nodes._FillValue == INT_FILL_VALUE
        assert INT_FILL_VALUE in ux_grid.Mesh2_face_nodes.values
