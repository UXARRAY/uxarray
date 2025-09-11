import numpy.testing as nt
import xarray as xr
import uxarray as ux
from uxarray import UxDataset
import pytest

import numpy as np





@pytest.fixture()
def healpix_sample_ds():
    uxgrid = ux.Grid.from_healpix(zoom=1)
    fc_var = ux.UxDataArray(data=np.ones((3, uxgrid.n_face)), dims=['time', 'n_face'], uxgrid=uxgrid)
    nc_var = ux.UxDataArray(data=np.ones((3, uxgrid.n_node)), dims=['time', 'n_node'], uxgrid=uxgrid)
    return ux.UxDataset({"fc": fc_var, "nc": nc_var}, uxgrid=uxgrid)




@pytest.fixture()
def healpix_sample_ds():
    uxgrid = ux.Grid.from_healpix(zoom=1)
    fc_var = ux.UxDataArray(data=np.ones((3, uxgrid.n_face)), dims=['time', 'n_face'], uxgrid=uxgrid)
    nc_var = ux.UxDataArray(data=np.ones((3, uxgrid.n_node)), dims=['time', 'n_node'], uxgrid=uxgrid)
    return ux.UxDataset({"fc": fc_var, "nc": nc_var}, uxgrid=uxgrid)

def test_uxgrid_setget(gridpath, datasetpath):
    """Load a dataset with its grid topology file using uxarray's
    open_dataset call and check its grid object."""
    uxds_var2_ne30 = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc"))
    uxgrid_var2_ne30 = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    assert (uxds_var2_ne30.uxgrid == uxgrid_var2_ne30)

def test_integrate(gridpath, datasetpath, mesh_constants):
    """Load a dataset and calculate integrate()."""
    uxds_var2_ne30 = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc"))
    integrate_var2 = uxds_var2_ne30.integrate()
    nt.assert_almost_equal(integrate_var2, mesh_constants['VAR2_INTG'], decimal=3)

def test_info(gridpath, datasetpath):
    """Tests custom info containing grid information."""
    uxds_var2_geoflow = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        try:
            uxds_var2_geoflow.info(show_attrs=True)
        except Exception as exc:
            assert False, f"'uxds_var2_geoflow.info()' raised an exception: {exc}"

def test_ugrid_dim_names(gridpath):
    """Tests the remapping of dimensions to the UGRID conventions."""
    ugrid_dims = ["n_face", "n_node", "n_edge"]
    uxds_remap = ux.open_dataset(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    for dim in ugrid_dims:
        assert dim in uxds_remap.dims

def test_get_dual(gridpath, datasetpath):
    """Tests the creation of the dual mesh on a data set."""
    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc"))
    dual = uxds.get_dual()

    assert isinstance(dual, UxDataset)
    assert len(uxds.data_vars) == len(dual.data_vars)
