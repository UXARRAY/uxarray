import uxarray as ux
import pytest
import numpy as np
import xarray as xr


@pytest.fixture()
def ds():
    """Basic dataset containing temporal and spatial variables on a HEALPix grid."""
    uxgrid = ux.Grid.from_healpix(zoom=1)
    t_var = ux.UxDataArray(data=np.ones((3,)), dims=['time'], uxgrid=uxgrid)
    fc_var = ux.UxDataArray(data=np.ones((3, uxgrid.n_face)), dims=['time', 'n_face'], uxgrid=uxgrid)
    nc_var = ux.UxDataArray(data=np.ones((3, uxgrid.n_node)), dims=['time', 'n_node'], uxgrid=uxgrid)

    uxds = ux.UxDataset({"fc": fc_var, "nc": nc_var, "t": t_var}, uxgrid=uxgrid)

    uxds["fc"] = uxds["fc"].assign_coords(face_id=("n_face", np.arange(uxgrid.n_face)))
    uxds["nc"] = uxds["nc"].assign_coords(node_id=("n_node", np.arange(uxgrid.n_node)))
    uxds["t"] = uxds["t"].assign_coords(time_id=("time", np.arange(uxds.dims["time"])))

    return uxds


class TestInheritedMethods:

    def test_where(self, ds):
        cond = (ds['face_id'] % 2) == 0
        result = ds.where(cond)
        assert isinstance(result, ux.UxDataset)
        assert hasattr(result, 'uxgrid') and result.uxgrid is ds.uxgrid
        # odd faces should be NaN after masking
        assert np.isnan(result['fc'].isel(n_face=1)).all()


    def test_assignment(self, ds):
        out = ds.assign(fc_doubled=ds['fc'] * 2)
        assert isinstance(out, ux.UxDataset)
        assert hasattr(out, 'uxgrid') and out.uxgrid is ds.uxgrid
        assert 'fc_doubled' in out

    def test_drop_vars(self, ds):
        ds_copy = ds.copy(deep=True)
        ds_copy['fc_copy'] = ds_copy['fc'].copy()
        out = ds_copy.drop_vars('fc_copy')
        assert isinstance(out, ux.UxDataset)
        assert hasattr(out, 'uxgrid') and out.uxgrid is ds_copy.uxgrid
        assert 'fc_copy' not in out

    def test_transpose(self, ds):
        dims = list(ds.dims)
        dims_rev = list(reversed(dims))
        out = ds.transpose(*dims_rev)
        assert isinstance(out, ux.UxDataset)
        assert hasattr(out, 'uxgrid') and out.uxgrid is ds.uxgrid


    def test_fillna(self, ds):
        ds_nan = ds.copy(deep=True)
        ds_nan['t'].values[0] = np.nan
        ds_nan['fc'].values[0, 0] = np.nan
        out = ds_nan.fillna(0)
        assert isinstance(out, ux.UxDataset)
        assert hasattr(out, 'uxgrid') and out.uxgrid is ds_nan.uxgrid
        assert not np.isnan(out['t'].values).any()
        assert not np.isnan(out['fc'].values).any()

    def test_rename(self, ds):
        out = ds.rename({'t': 't_renamed'})
        assert isinstance(out, ux.UxDataset)
        assert hasattr(out, 'uxgrid') and out.uxgrid is ds.uxgrid
        assert 't_renamed' in out and 't' not in out

    def test_to_array(self, ds):
        arr = ds.to_array()
        assert isinstance(arr, ux.UxDataArray)
        assert hasattr(arr, 'uxgrid') and arr.uxgrid is ds.uxgrid
        # variables dimension should include our three variables
        for v in ['fc', 'nc', 't']:
            assert v in arr['variable'].values

    def test_arithmetic_operations(self, ds):
        da = ds['fc'] + 1
        assert isinstance(da, ux.UxDataArray)
        assert hasattr(da, 'uxgrid')

        out = ds * 2
        assert isinstance(out, ux.UxDataset)
        assert hasattr(out, 'uxgrid')

        out2 = ds.copy(deep=True)
        out2['fc_squared'] = ds['fc'] ** 2
        assert isinstance(out2, ux.UxDataset)
        assert hasattr(out2, 'uxgrid')
        assert 'fc_squared' in out2

    def test_reduction_methods(self, ds):
        # Reduce over time -> keep spatial dims; result should stay a Dataset
        reduced = ds.mean(dim='time')
        assert isinstance(reduced, ux.UxDataset)
        assert hasattr(reduced, 'uxgrid')

        da_sum = ds['fc'].sum(dim='n_face')
        assert isinstance(da_sum, ux.UxDataArray)
        assert hasattr(da_sum, 'uxgrid')

    def test_groupby(self, ds):
        grouped = ds['fc'].groupby('face_id').mean()
        assert isinstance(grouped, ux.UxDataArray)
        assert hasattr(grouped, 'uxgrid')

    def test_assign_coords(self, ds):
        n = ds.dims['n_face']
        new_coord = xr.DataArray(np.arange(n) * 10, dims=['n_face'])
        out = ds.assign_coords(scaled_id=new_coord)
        assert isinstance(out, ux.UxDataset)
        assert hasattr(out, 'uxgrid') and out.uxgrid is ds.uxgrid
        assert 'scaled_id' in out.coords
        assert np.array_equal(out['scaled_id'].values, np.arange(n) * 10)

    def test_expand_dims(self, ds):
        out = ds.expand_dims({'member': 1})
        assert isinstance(out, ux.UxDataset)
        assert hasattr(out, 'uxgrid') and out.uxgrid is ds.uxgrid
        assert 'member' in out.dims and out.dims['member'] == 1
        assert 'member' in out['t'].dims and out['t'].sizes['member'] == 1

    def test_method_chaining(self, ds):
        out = (
            ds.assign(t_kelvin=ds['t'] + 273.15)
              .rename({'t': 't_celsius'})
              .fillna(0)
        )
        assert isinstance(out, ux.UxDataset)
        assert hasattr(out, 'uxgrid') and out.uxgrid is ds.uxgrid
        assert 't_celsius' in out and 't_kelvin' in out

    def test_stack_unstack(self, ds):
        # stack only over a subset with compatible dims
        ds_fc = ds[['fc']]
        stacked = ds_fc.stack(tf=('time', 'n_face'))
        assert isinstance(stacked, ux.UxDataset)
        assert hasattr(stacked, 'uxgrid')

        unstacked = stacked.unstack('tf')
        assert isinstance(unstacked, ux.UxDataset)
        assert hasattr(unstacked, 'uxgrid')
        assert unstacked['fc'].shape == ds_fc['fc'].shape

    def test_sortby(self, ds):
        n = ds.dims['n_face']
        ds_fc = ds[['fc']].assign_coords(reverse_id=('n_face', np.arange(n)[::-1]))
        out = ds_fc.sortby('reverse_id')
        assert isinstance(out, ux.UxDataset)
        assert hasattr(out, 'uxgrid')
        assert np.array_equal(out['face_id'].values, ds_fc['face_id'].values[::-1])

    def test_shift(self, ds):
        ds_fc = ds[['fc']]
        out = ds_fc.shift(n_face=1)
        assert isinstance(out, ux.UxDataset)
        assert hasattr(out, 'uxgrid')
        assert np.isnan(out['fc'].isel(n_face=0).values).all()


class TestDatasetSelection:

    def test_isel_face_dim(self, ds):
        ds_f_single = ds.isel(n_face=0)

        assert len(ds_f_single.coords) == 3

        assert ds_f_single.uxgrid != ds.uxgrid
        assert ds_f_single.sizes['n_face'] == 1
        assert ds_f_single.sizes['n_node'] >= 4

        ds_f_multi = ds.isel(n_face=[0, 1])

        assert len(ds_f_multi.coords) == 3

        assert ds_f_multi.uxgrid != ds.uxgrid
        assert ds_f_multi.sizes['n_face'] == 2
        assert ds_f_multi.sizes['n_node'] >= 4


    def test_isel_node_dim(self, ds):
        ds_n_single = ds.isel(n_node=0)

        assert len(ds_n_single.coords) == 3

        assert ds_n_single.uxgrid != ds.uxgrid
        assert ds_n_single.sizes['n_face'] >= 1

        ds_n_multi = ds.isel(n_node=[0, 1])

        assert len(ds_n_multi.coords) == 3

        assert ds_n_multi.uxgrid != ds.uxgrid
        assert ds_n_multi.uxgrid.sizes['n_face'] >= 1

    def test_isel_non_grid_dim(self, ds):
        ds_t_single = ds.isel(time=0)

        assert len(ds_t_single.coords) == 3

        assert ds_t_single.uxgrid == ds.uxgrid
        assert "time" not in ds_t_single.sizes

        ds_t_multi = ds.isel(time=[0, 1])

        assert len(ds_t_multi.coords) == 3

        assert ds_t_multi.uxgrid == ds.uxgrid
        assert ds_t_multi.sizes['time'] == 2
