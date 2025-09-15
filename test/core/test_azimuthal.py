import pytest
import uxarray as ux
import numpy as np


def test_gaussian(gridpath, datasetpath):
    uxds = ux.open_dataset(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
    )

    res = uxds['gaussian'].azimuthal_mean(center_coord=(45, 0), outer_radius=2, radius_step=0.5)

    # Expects decreasing values from center
    valid_vals = res[1:]

    np.testing.assert_array_less(
        valid_vals.diff("radius").values, 1e-12
    )



def test_inverse_gaussian(gridpath, datasetpath):

    uxds =  ux.open_dataset(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
    )

    res = uxds['inverse_gaussian'].azimuthal_mean(center_coord=(45, 0), outer_radius=2, radius_step=0.5)

    # Expects increasing values from center
    atol = 1e-12
    diffs = res[1:].diff("radius").values
    diffs = diffs[np.isfinite(diffs)]
    np.testing.assert_array_less(-atol, diffs)

def test_hit_counts(gridpath, datasetpath):
    uxds =  ux.open_dataset(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
    )

    res, hit_counts = uxds['inverse_gaussian'].azimuthal_mean(center_coord=(45, 0), outer_radius=2, radius_step=0.5, return_hit_counts=True)

    assert 'radius' in hit_counts.dims
    assert hit_counts.sizes['radius'] == res.sizes['radius']
