import numpy.testing as nt
import xarray as xr
import uxarray as ux
import uxarray.errors
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
    # integrate() now returns a UxDataset with the integral of each variable
    assert isinstance(integrate_var2, UxDataset)
    nt.assert_almost_equal(integrate_var2["var2"].values, mesh_constants['VAR2_INTG'], decimal=3)


def test_integrate_multiple_data_arrays(gridpath, datasetpath, mesh_constants):
    """integrate() integrates every data variable into a new UxDataset."""
    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc"))

    # Add a second face-centered variable: doubling the data doubles the integral
    uxds["var2_doubled"] = uxds["var2"] * 2.0

    result = uxds.integrate()
    assert isinstance(result, UxDataset)
    assert set(result.data_vars) == {"var2", "var2_doubled"}

    nt.assert_almost_equal(result["var2"].values, mesh_constants['VAR2_INTG'], decimal=3)
    nt.assert_almost_equal(
        result["var2_doubled"].values, 2.0 * result["var2"].values, decimal=10
    )


def test_integrate_skips_non_grid_variables(gridpath, datasetpath):
    """Variables not mapped to the grid are skipped with a warning."""
    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc"))

    # A variable whose final dimension does not map to the grid
    uxds["not_on_grid"] = xr.DataArray(np.arange(5.0), dims=["other_dim"])

    with pytest.warns(UserWarning, match="skipped during integration"):
        result = uxds.integrate()

    assert "not_on_grid" not in result.data_vars
    assert "var2" in result.data_vars

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


def _load_sel_subset(gridpath, datasetpath):
    grid_file = gridpath("ugrid", "outCSne30", "outCSne30.ug")
    data_file = datasetpath("ugrid", "outCSne30", "outCSne30_sel_timeseries.nc")
    uxds = ux.open_dataset(grid_file, data_file)
    base_time = np.datetime64("2018-04-28T00:00:00")
    offsets = np.arange(uxds.sizes["time"], dtype="timedelta64[h]")
    uxds = uxds.assign_coords(time=(base_time + offsets).astype("datetime64[ns]"))
    return uxds


def test_sel_time_slice(gridpath, datasetpath):
    uxds = _load_sel_subset(gridpath, datasetpath)

    times = uxds["time"].values
    sliced = uxds.sel(time=slice(times[0], times[2]))

    assert sliced.dims["time"] == 3
    np.testing.assert_array_equal(sliced["time"].values, times[:3])


def test_sel_method_forwarded(gridpath, datasetpath):
    uxds = _load_sel_subset(gridpath, datasetpath)

    target = np.datetime64("2018-04-28T02:20:00")
    nearest = uxds.sel(time=target, method="nearest")

    np.testing.assert_array_equal(
        nearest["time"].values,
        np.array(uxds["time"].values[2], dtype="datetime64[ns]"),
    )

def test_isel_ignore_grid():
    """ensure UxDataset.isel(..., ignore_grid=True) still attaches result.uxgrid.
    Regression test for issue #1683.
    """
    uxds = ux.tutorial.open_dataset("outCSne30-timeseries")
    result = uxds.isel(time=0, ignore_grid=True)
    result.uxgrid  # (will cause crash if uxgrid not properly attached to result)
    assert result.uxgrid == uxds.uxgrid

    result = uxds.isel(n_face=0, ignore_grid=True)
    result.uxgrid  # (will cause crash if uxgrid not properly attached to result)
    assert result.uxgrid == uxds.uxgrid   # ignore_grid means grid never gets sliced here


def test_uxdataset_init_from_xarray_dataset():
    ds = xr.Dataset(
        data_vars={"a": ("x", [1, 2])},
        coords={"x": [10, 20]},
        attrs={"source": "testing"},
    )

    uxds = ux.UxDataset(ds)

    assert "a" in uxds.data_vars
    assert "x" in uxds.coords
    assert uxds.attrs["source"] == "testing"

def test_uxdataset_to_array():
    """Tests UxDataset.to_array(), ensuring `dim` and `name` kwargs work too."""
    uxds = UxDataset(
        data_vars={
            "a": ("x", [1, 2]),
            "b": ("x", [3, 4]),
            "c": ("y", [-1, -2, -3, -4]),
        },
        coords={"x": [10, 20], "y": [-10, -20, -30, -40]},
        attrs={"source": "testing"},
    )
    # first check basic functionality without worrying about kwargs
    arr = uxds.to_array()
    assert isinstance(arr, ux.UxDataArray)
    assert arr.sizes == {"variable": 3, "x": 2, "y": 4}
    assert arr.attrs["source"] == "testing"
    for k, c in arr.coords.items():
        assert k in arr.coords and c.equals(arr.coords[k])
    # next check that dim & name args/kwargs work as expected.
    arr1 = uxds.to_array('custom_dim')
    assert arr1.sizes == {"custom_dim": 3, "x": 2, "y": 4}
    assert arr1.name is None
    arr2 = uxds.to_array(dim='custom_dim', name='custom_name')
    assert arr2.name == 'custom_name'


class TestNeighborhood:
    """Tests for ``UxDataset.neighborhood`` and the reductions on it."""

    def test_face_centered(self, gridpath, datasetpath):
        """Ensures the dataset-level reduction matches the per-variable
        ``UxDataArray.neighborhood`` results."""
        uxds = ux.open_dataset(
            gridpath("ugrid", "outCSne30", "outCSne30.ug"),
            datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"),
        )

        filtered_ds = uxds.neighborhood(r=5.0).mean()
        filtered_da = uxds["psi"].neighborhood(r=5.0).mean()

        assert isinstance(filtered_ds, UxDataset)
        nt.assert_allclose(filtered_ds["psi"].values, filtered_da.values)

    def test_non_grid_variable_skipped(self):
        """Data variables without a grid dimension should be left
        untouched."""
        uxgrid = ux.Grid.from_healpix(zoom=1)

        uxds = UxDataset(
            data_vars={
                "face_var": ("n_face", np.arange(uxgrid.n_face, dtype=float)),
                "scalar_var": ("other_dim", np.array([1.0, 2.0, 3.0])),
            },
            uxgrid=uxgrid,
        )

        filtered = uxds.neighborhood(r=0.0).mean()

        nt.assert_allclose(filtered["face_var"].values, uxds["face_var"].values)
        nt.assert_allclose(filtered["scalar_var"].values, uxds["scalar_var"].values)

    def test_one_query_per_grid_location(self):
        """Variables sharing a grid location must share one neighbor query.

        The query dominates the cost of a reduction, so rebuilding it per
        variable would make a dataset reduction scale with the number of
        variables. Counting calls is the only way to see that from outside.
        """
        from unittest.mock import patch

        import uxarray.grid.neighbors as neighbors

        uxgrid = ux.Grid.from_healpix(zoom=2)
        # touch both locations first: a HEALPix grid cannot populate node
        # coordinates lazily from inside the tree build
        n_node, n_face = uxgrid.n_node, uxgrid.n_face
        rng = np.random.default_rng(0)
        uxds = UxDataset(
            data_vars={
                "face_a": ("n_face", rng.random(n_face)),
                "face_b": ("n_face", rng.random(n_face)),
                "face_c": ("n_face", rng.random(n_face)),
                "node_a": ("n_node", rng.random(n_node)),
            },
            uxgrid=uxgrid,
        )

        real = neighbors._csr_neighbors
        with patch.object(neighbors, "_csr_neighbors", side_effect=real) as spy:
            filtered = uxds.neighborhood(r=20.0).percentile(90)

        assert spy.call_count == 2, (
            f"expected one query per grid location (faces, nodes), got "
            f"{spy.call_count}"
        )
        # and the reduction, with its parameter, reached every variable
        for name in ("face_a", "node_a"):
            nt.assert_allclose(
                filtered[name].values,
                uxds[name].neighborhood(r=20.0).percentile(90).values,
            )

    def test_one_query_reused_across_reductions(self):
        """A DatasetNeighborhood holds its queries, so a second reduction on
        the same object must not rebuild them."""
        from unittest.mock import patch

        import uxarray.grid.neighbors as neighbors

        uxgrid = ux.Grid.from_healpix(zoom=2)
        rng = np.random.default_rng(0)
        uxds = UxDataset(
            data_vars={"face_a": ("n_face", rng.random(uxgrid.n_face))},
            uxgrid=uxgrid,
        )

        nb = uxds.neighborhood(r=20.0)
        real = neighbors._csr_neighbors
        with patch.object(neighbors, "_csr_neighbors", side_effect=real) as spy:
            smooth, spread = nb.mean(), nb.std(ddof=1)

        assert spy.call_count == 1, (
            f"expected the query to be built once and reused, got {spy.call_count}"
        )
        assert smooth["face_a"].shape == spread["face_a"].shape

    def test_callable_escape_hatch(self):
        """``reduce`` applies a user's own function to every grid-mapped
        variable."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        rng = np.random.default_rng(0)
        uxds = UxDataset(
            data_vars={"face_a": ("n_face", rng.random(uxgrid.n_face))},
            uxgrid=uxgrid,
        )

        def rms(values, axis):
            return np.sqrt(np.mean(values**2, axis=axis))

        filtered = uxds.neighborhood(r=20.0).reduce(rms)
        assert np.all(filtered["face_a"].values >= 0)


def test_uxgrid_None_is_invalid_in_uxdataset():
    """Ensures GridInvalidError gets raised if uxgrid=None when getting UxDataset.uxgrid.
    Regression test for #1620.
    """
    # construct array without uxgrid. Ideally this line would crash, but allowing uxgrid=None
    # is an important workaround for subclassing from xarray. see #1620 for more details.
    ds = ux.UxDataset({'arr0': xr.DataArray([1,2,3], dims=['n_face'])})
    # ensure getting arr.uxgrid crashes with GridInvalidError (it is None...)
    with pytest.raises(uxarray.errors.GridInvalidError):
        ds.uxgrid

    # trying to set uxgrid to a non-Grid should raise TypeError:
    with pytest.raises(TypeError):
        ds.uxgrid = "not a grid"
    with pytest.raises(TypeError):
        ds.uxgrid = 123
    # this remains true even for None, outside of __init__:
    with pytest.raises(TypeError):
        ds.uxgrid = None
    # it also applies (for non-None non-Grid objects) during __init__:
    with pytest.raises(TypeError):
        ux.UxDataset({'arr1': xr.DataArray([4,5], dims=['n_face'])}, uxgrid=[1,2])
