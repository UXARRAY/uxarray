import os
from unittest import TestCase
from pathlib import Path
import numpy.testing as nt
import xarray as xr

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
dsfile_v1_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"

mpas_ds_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'


class TestUxDataset(TestCase):

    def test_uxgrid_setget(self):
        """Load a dataset with its grid topology file using uxarray's
        open_dataset call and check its grid object."""

        uxds_var2_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

        uxgrid_var2_ne30 = ux.open_grid(gridfile_ne30)
        assert (uxds_var2_ne30.uxgrid == uxgrid_var2_ne30)

    def test_integrate(self):
        """Load a dataset and calculate integrate()."""

        uxds_var2_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

        integrate_var2 = uxds_var2_ne30.integrate()

        nt.assert_almost_equal(integrate_var2, constants.VAR2_INTG, decimal=3)

    def test_info(self):
        """Tests custom info containing grid information."""
        uxds_var2_geoflow = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)

        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            try:
                uxds_var2_geoflow.info(show_attrs=True)
            except Exception as exc:
                assert False, f"'uxds_var2_geoflow.info()' raised an exception: {exc}"

    def test_ugrid_dim_names(self):
        """Tests the remapping of dimensions to the UGRID conventions."""

        ugrid_dims = ["n_face", "n_node", "n_edge"]

        uxds_remap = ux.open_dataset(mpas_ds_path, mpas_ds_path)

        for dim in ugrid_dims:
            assert dim in uxds_remap.dims

    def test_read_from_https(self):
        """Tests reading a dataset from a HTTPS link."""
        import requests

        small_file_480km = requests.get(
            "https://web.lcrc.anl.gov/public/e3sm/inputdata/share/meshes/mpas/ocean/oQU480.230422.nc"
        ).content

        ds_small_480km = ux.open_dataset(small_file_480km, small_file_480km)
        assert isinstance(ds_small_480km, ux.core.dataset.UxDataset)
