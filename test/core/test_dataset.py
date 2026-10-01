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


class TestUxDatasetMimicsUxDataArrayMethods:
    """Testing behavior of UxDataset methods which simply apply UxDataArray methods iteratively,
    such as UxDataset.zonal_mean().
    """
    def _uxds_face_with_just_psi(self):
        uxds = ux.tutorial.open_dataset("outCSne30-timeseries")
        assert set(uxds.data_vars) == {'psi'}
        assert 'n_face' in uxds.dims
        return uxds

    def _uxds_face_with_more_vars_and_coords(self):
        uxds = ux.tutorial.open_dataset("outCSne30-timeseries")
        assert set(uxds.data_vars) == {'psi'}
        uxds = uxds.assign(psi2 = uxds['psi']*2)
        uxds = uxds.assign(unrelated_time_var = xr.DataArray(10*np.arange(uxds.sizes['time']), dims=['time']))
        uxds = uxds.assign(unrelated_scalar_var = 7)
        uxds = uxds.assign_coords(unrelated_scalar_coord = 70)
        # (will also want to check what happens to coords which are not used by any data_vars!)
        uxds = uxds.assign_coords(unused_dim_coord = xr.DataArray([1,2,3], dims='unused_dim'))
        assert set(uxds.data_vars) == {'psi', 'psi2', 'unrelated_time_var', 'unrelated_scalar_var'}
        return uxds

    def _uxds_hex_face(self):
        uxds = ux.tutorial.open_dataset('quad-hexagon-random-face')
        assert set(uxds.data_vars) == {'random_data_face'}
        assert 'n_face' in uxds.dims
        return uxds

    def _uxds_hex_edge(self):
        uxds = ux.tutorial.open_dataset('quad-hexagon-random-edge')
        assert set(uxds.data_vars) == {'random_data_edge'}
        assert 'n_edge' in uxds.dims
        return uxds

    def _uxds_hex_node(self):
        uxds = ux.tutorial.open_dataset('quad-hexagon-random-node')
        assert set(uxds.data_vars) == {'random_data_node'}
        assert 'n_node' in uxds.dims
        return uxds

    def _uxds_hex_face_and_node(self):
        arr_face = self._uxds_hex_face()['random_data_face']
        arr_node = self._uxds_hex_node()['random_data_node']
        uxds = ux.UxDataset({'face_data': arr_face, 'node_data': arr_node}, uxgrid=arr_face.uxgrid)
        return uxds

    def _uxds_hex_face_and_edge(self):
        arr_face = self._uxds_hex_face()['random_data_face']
        arr_edge = self._uxds_hex_edge()['random_data_edge']
        uxds = ux.UxDataset({'face_data': arr_face, 'edge_data': arr_edge}, uxgrid=arr_face.uxgrid)
        return uxds

    def _uxds_hex_node_and_edge(self):
        arr_node = self._uxds_hex_node()['random_data_node']
        arr_edge = self._uxds_hex_edge()['random_data_edge']
        uxds = ux.UxDataset({'node_data': arr_node, 'edge_data': arr_edge}, uxgrid=arr_node.uxgrid)
        return uxds

    def _uxds_hex_face_and_node_and_edge(self):
        arr_face = self._uxds_hex_face()['random_data_face']
        arr_node = self._uxds_hex_node()['random_data_node']
        arr_edge = self._uxds_hex_edge()['random_data_edge']
        uxds = ux.UxDataset({'face_data': arr_face, 'node_data': arr_node, 'edge_data': arr_edge}, uxgrid=arr_face.uxgrid)
        return uxds

    def test_uxds_mimics_uxda_zonal_mean(self):
        """Ensure UxDataset.zonal_mean() mimics UxDataArray.zonal_mean() for each variable."""
        ds = self._uxds_face_with_just_psi()
        psi_result = ds['psi'].zonal_mean()
        assert ds.zonal_mean()['psi'].equals(psi_result)
        assert isinstance(psi_result, xr.DataArray)
        assert isinstance(ds.zonal_mean(), xr.Dataset)

        # quick sanity check: zonal_mean and zonal_average are aliases.
        # (doing this here instead of making a separate test for zonal_average...)
        assert ds['psi'].zonal_average().equals(ds['psi'].zonal_mean())
        assert ds.zonal_average().equals(ds.zonal_mean())

        ds = self._uxds_face_with_more_vars_and_coords()
        psi_result = ds['psi'].zonal_mean()
        psi2_result = ds['psi2'].zonal_mean()
        ds_result = ds.zonal_mean()
        assert ds_result['psi'].equals(psi_result)
        assert ds_result['psi2'].equals(psi2_result)
        assert 'unrelated_scalar_var' not in ds['psi'] and 'unrelated_scalar_var' not in psi_result.coords
        assert ds_result['unrelated_scalar_var'].equals(ds['unrelated_scalar_var'])
        assert 'unrelated_time_var' not in ds['psi'] and 'unrelated_time_var' not in psi_result.coords
        assert ds_result['unrelated_time_var'].equals(ds['unrelated_time_var'])
        assert ds_result.coords['unrelated_scalar_coord'].equals(ds.coords['unrelated_scalar_coord'])
        assert ds_result.coords['unused_dim_coord'].equals(ds.coords['unused_dim_coord'])

        # zonal mean doesn't support edge-centered data
        ds = self._uxds_hex_edge()
        arr = ds['random_data_edge']
        with pytest.raises(uxarray.errors.DataCenteringError):
            arr.zonal_mean()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.zonal_mean()

        # zonal mean doesn't support node-centered data
        ds = self._uxds_hex_node()
        arr = ds['random_data_node']
        with pytest.raises(uxarray.errors.DataCenteringError):
            arr.zonal_mean()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.zonal_mean()

        # also should crash if any data_vars contain unsupported centering,
        # even if some data_vars are centered at supported location (faces).
        ds = self._uxds_hex_face_and_node()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.zonal_mean()
        ds = self._uxds_hex_face_and_edge()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.zonal_mean()

    def test_uxds_mimics_uxda_zonal_anomaly(self):
        """Ensure UxDataset.zonal_anomaly() mimics UxDataArray.zonal_anomaly() for each variable."""
        ds = self._uxds_face_with_just_psi()
        psi_result = ds['psi'].zonal_anomaly()
        assert ds.zonal_anomaly()['psi'].equals(psi_result)
        assert isinstance(psi_result, xr.DataArray)
        assert isinstance(ds.zonal_anomaly(), xr.Dataset)

        ds = self._uxds_face_with_more_vars_and_coords()
        psi_result = ds['psi'].zonal_anomaly()
        psi2_result = ds['psi2'].zonal_anomaly()
        ds_result = ds.zonal_anomaly()
        assert ds_result['psi'].equals(psi_result)
        assert ds_result['psi2'].equals(psi2_result)
        assert 'unrelated_scalar_var' not in ds['psi'] and 'unrelated_scalar_var' not in psi_result.coords
        assert ds_result['unrelated_scalar_var'].equals(ds['unrelated_scalar_var'])
        assert 'unrelated_time_var' not in ds['psi'] and 'unrelated_time_var' not in psi_result.coords
        assert ds_result['unrelated_time_var'].equals(ds['unrelated_time_var'])
        assert ds_result.coords['unrelated_scalar_coord'].equals(ds.coords['unrelated_scalar_coord'])
        assert ds_result.coords['unused_dim_coord'].equals(ds.coords['unused_dim_coord'])

        # zonal anomaly doesn't support edge-centered data
        ds = self._uxds_hex_edge()
        arr = ds['random_data_edge']
        with pytest.raises(uxarray.errors.DataCenteringError):
            arr.zonal_anomaly()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.zonal_anomaly()

        # zonal anomaly doesn't support node-centered data
        ds = self._uxds_hex_node()
        arr = ds['random_data_node']
        with pytest.raises(uxarray.errors.DataCenteringError):
            arr.zonal_anomaly()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.zonal_anomaly()

        # also should crash if any data_vars contain unsupported centering,
        # even if some data_vars are centered at supported location (faces).
        ds = self._uxds_hex_face_and_node()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.zonal_anomaly()
        ds = self._uxds_hex_face_and_edge()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.zonal_anomaly()

    def test_uxds_mimics_uxda_azimuthal_mean(self):
        """Ensure UxDataset.azimuthal_mean() mimics UxDataArray.azimuthal_mean() for each variable."""
        kw_psi = dict(center_coord=(45, 0), outer_radius=50, radius_step=10)
        kw_hex = dict(center_coord=(0, 0), outer_radius=0.3, radius_step=0.1)

        ds = self._uxds_face_with_just_psi()
        psi_result = ds['psi'].azimuthal_mean(**kw_psi)
        assert ds.azimuthal_mean(**kw_psi)['psi'].equals(psi_result)
        assert isinstance(psi_result, xr.DataArray)
        assert isinstance(ds.azimuthal_mean(**kw_psi), xr.Dataset)

        # quick sanity check: azimuthal_mean and azimuthal_average are aliases.
        # (doing this here instead of making a separate test for azimuthal_average...)
        assert ds['psi'].azimuthal_average(**kw_psi).equals(ds['psi'].azimuthal_mean(**kw_psi))
        assert ds.azimuthal_average(**kw_psi).equals(ds.azimuthal_mean(**kw_psi))

        ds = self._uxds_face_with_more_vars_and_coords()
        psi_result = ds['psi'].azimuthal_mean(**kw_psi)
        psi2_result = ds['psi2'].azimuthal_mean(**kw_psi)
        ds_result = ds.azimuthal_mean(**kw_psi)
        assert ds_result['psi'].equals(psi_result)
        assert ds_result['psi2'].equals(psi2_result)
        assert 'unrelated_scalar_var' not in ds['psi'] and 'unrelated_scalar_var' not in psi_result.coords
        assert ds_result['unrelated_scalar_var'].equals(ds['unrelated_scalar_var'])
        assert 'unrelated_time_var' not in ds['psi'] and 'unrelated_time_var' not in psi_result.coords
        assert ds_result['unrelated_time_var'].equals(ds['unrelated_time_var'])
        assert ds_result.coords['unrelated_scalar_coord'].equals(ds.coords['unrelated_scalar_coord'])
        assert ds_result.coords['unused_dim_coord'].equals(ds.coords['unused_dim_coord'])

        # azimuthal mean doesn't support edge-centered data
        ds = self._uxds_hex_edge()
        arr = ds['random_data_edge']
        with pytest.raises(uxarray.errors.DataCenteringError):
            arr.azimuthal_mean(**kw_hex)
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.azimuthal_mean(**kw_hex)

        # azimuthal mean doesn't support node-centered data
        ds = self._uxds_hex_node()
        arr = ds['random_data_node']
        with pytest.raises(uxarray.errors.DataCenteringError):
            arr.azimuthal_mean(**kw_hex)
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.azimuthal_mean(**kw_hex)

        # also should crash if any data_vars contain unsupported centering,
        # even if some data_vars are centered at supported location (faces).
        ds = self._uxds_hex_face_and_node()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.azimuthal_mean(**kw_hex)
        ds = self._uxds_hex_face_and_edge()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.azimuthal_mean(**kw_hex)

    def test_uxds_mimics_uxda_weighted_mean(self):
        """Ensure UxDataset.weighted_mean() mimics UxDataArray.weighted_mean() for each variable."""
        ds = self._uxds_face_with_just_psi()
        psi_result = ds['psi'].weighted_mean()
        assert ds.weighted_mean()['psi'].equals(psi_result)
        assert isinstance(psi_result, ux.UxDataArray)
        assert isinstance(ds.weighted_mean(), ux.UxDataset)

        ds = self._uxds_face_with_more_vars_and_coords()
        psi_result = ds['psi'].weighted_mean()
        psi2_result = ds['psi2'].weighted_mean()
        ds_result = ds.weighted_mean()
        assert ds_result['psi'].equals(psi_result)
        assert ds_result['psi2'].equals(psi2_result)
        assert 'unrelated_scalar_var' not in ds['psi'] and 'unrelated_scalar_var' not in psi_result.coords
        assert ds_result['unrelated_scalar_var'].equals(ds['unrelated_scalar_var'])
        assert 'unrelated_time_var' not in ds['psi'] and 'unrelated_time_var' not in psi_result.coords
        assert ds_result['unrelated_time_var'].equals(ds['unrelated_time_var'])
        assert ds_result.coords['unrelated_scalar_coord'].equals(ds.coords['unrelated_scalar_coord'])
        assert ds_result.coords['unused_dim_coord'].equals(ds.coords['unused_dim_coord'])

        # weighted_mean does support edge-centered data
        ds = self._uxds_hex_edge()
        arr = ds['random_data_edge']
        arr_result = arr.weighted_mean()
        ds_result = ds.weighted_mean()
        assert ds_result['random_data_edge'].equals(arr_result)

        # weighted_mean also supports ds with both edge & face data,
        # but only if weights not provided
        ds = self._uxds_hex_face_and_edge()
        arr_face_result = ds['face_data'].weighted_mean()
        arr_edge_result = ds['edge_data'].weighted_mean()
        ds_result = ds.weighted_mean()
        assert ds_result['face_data'].equals(arr_face_result)
        assert ds_result['edge_data'].equals(arr_edge_result)
        # when weights provided, usually make DataCenteringError:
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.weighted_mean(weights=np.arange(ds.sizes['n_face']))
        # but, if weights are a 1D xr.DataArrray with dim='n_face' or 'n_edge',
        # raise NotImplementedError instead (a clear unambiguous implementation is
        # possible in these case, but just isn't implemented yet).
        weights_face = xr.DataArray(np.arange(ds.sizes['n_face']), dims=['n_face'])
        weights_edge = xr.DataArray(np.arange(ds.sizes['n_edge']), dims=['n_edge'])
        with pytest.raises(NotImplementedError):
            ds.weighted_mean(weights=weights_face)
        with pytest.raises(NotImplementedError):
            ds.weighted_mean(weights=weights_edge)

        # weighted_mean doesn't support node-centered data
        ds = self._uxds_hex_node()
        arr = ds['random_data_node']
        with pytest.raises(uxarray.errors.DataCenteringError):
            arr.weighted_mean()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.weighted_mean()

        # also should crash if any data_vars contain unsupported centering,
        # even if some data_vars are centered at supported location (faces).
        ds = self._uxds_hex_face_and_node()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.weighted_mean()
        ds = self._uxds_hex_node_and_edge()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.weighted_mean()
        ds = self._uxds_hex_face_and_node_and_edge()
        with pytest.raises(uxarray.errors.DataCenteringError):
            ds.weighted_mean()
