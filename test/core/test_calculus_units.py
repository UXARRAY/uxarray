"""Regression coverage for calculus units and unit-sphere scaling (#1856)."""

from contextlib import nullcontext

import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux


@pytest.mark.parametrize("method", ["gradient", "curl", "divergence"])
@pytest.mark.parametrize("radius", [None, 1.0, 1.0 + 1e-12, 2.0])
@pytest.mark.parametrize("scale_by_radius", [False, True])
@pytest.mark.parametrize(
    "units,base_units",
    [(None, "1"), (" \t ", "1"), ("  K s-1 ", "K s-1"), ("  m/s \n", "(m/s)")],
)
def test_calculus_units_and_radius_scaling(
    gridpath, datasetpath, method, radius, scale_by_radius, units, base_units
):
    uxds = ux.open_dataset(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc"),
    )
    if radius is None:
        uxds.uxgrid._ds.attrs.pop("sphere_radius", None)
    else:
        uxds.uxgrid.sphere_radius = radius

    u = uxds["gaussian"].copy()
    v = uxds["inverse_gaussian"].copy()
    for component in (u, v):
        component.attrs.pop("units", None)
        if units is not None:
            component.attrs["units"] = units
    original_attrs = u.attrs.copy()

    def compute(scale):
        if method == "gradient":
            result = u.gradient(scale_by_radius=scale)
            return [result["zonal_gradient"], result["meridional_gradient"]]
        return [getattr(u, method)(v, scale_by_radius=scale)]

    unit_results = compute(False)
    warning = (
        pytest.warns(UserWarning, match="result is left on the unit sphere")
        if scale_by_radius and radius is None
        else nullcontext()
    )
    with warning:
        results = compute(scale_by_radius)

    physically_scaled = scale_by_radius and radius is not None and radius != 1.0
    denominator = "m" if physically_scaled else "rad"
    divisor = radius if physically_scaled else 1.0
    for result, unit_result in zip(results, unit_results):
        assert result.attrs["units"] == f"{base_units}/{denominator}"
        assert unit_result.attrs["units"] == f"{base_units}/rad"
        finite = np.isfinite(unit_result.values)
        assert finite.any()
        assert np.any(np.abs(unit_result.values[finite]) > 0)
        nt.assert_allclose(result.values, unit_result.values / divisor, equal_nan=True)

    assert u.attrs == original_attrs
