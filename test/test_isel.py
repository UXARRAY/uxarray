import pytest
import uxarray as ux
import numpy as np

@pytest.fixture()
def ds():
    uxgrid = ux.Grid.from_healpix(zoom=1)
    fc_var = ux.UxDataArray(data=np.ones((3, uxgrid.n_face)), dims=['time', 'n_face'], uxgrid=uxgrid)
    nc_var = ux.UxDataArray(data=np.ones((3, uxgrid.n_node)), dims=['time', 'n_node'], uxgrid=uxgrid)
    return ux.UxDataset({"fc": fc_var, "nc": nc_var}, uxgrid=uxgrid)


class TestDataset:

    def test_isel_face_dim(self, ds):
        ds_f_single = ds.isel(n_face=0)

        assert ds_f_single.uxgrid != ds.uxgrid
        assert ds_f_single.sizes['n_face'] == 1
        assert ds_f_single.sizes['n_node'] >= 4

        ds_f_multi = ds.isel(n_face=[0, 1])

        assert ds_f_multi.uxgrid != ds.uxgrid
        assert ds_f_multi.sizes['n_face'] == 2
        assert ds_f_multi.sizes['n_node'] >= 4


    def test_isel_node_dim(self, ds):
        ds_n_single = ds.isel(n_node=0)

        assert ds_n_single.uxgrid != ds.uxgrid
        assert ds_n_single.sizes['n_face'] >= 1

        ds_n_multi = ds.isel(n_node=[0, 1])

        assert ds_n_multi.uxgrid != ds.uxgrid
        assert ds_n_multi.uxgrid.sizes['n_face'] >= 1

    def test_isel_non_grid_dim(self, ds):
        ds_t_single = ds.isel(time=0)

        assert ds_t_single.uxgrid == ds.uxgrid
        assert "time" not in ds_t_single.sizes

        ds_t_multi = ds.isel(time=[0, 1])
        assert ds_t_multi.uxgrid == ds.uxgrid
        assert ds_t_multi.sizes['time'] == 2
