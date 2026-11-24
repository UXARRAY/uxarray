import uxarray as ux

import pytest



def test_grid_repr(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))

    out = uxgrid._repr_html_()

    assert out is not None


def test_dataset_repr(gridpath, datasetpath):
    uxds = ux.open_dataset(
        gridpath("ugrid", "quad-hexagon", "grid.nc"),
        datasetpath("ugrid", "quad-hexagon", "data.nc")
    )

    out = uxds._repr_html_()

    assert out is not None


def test_dataarray_repr(gridpath, datasetpath):
    uxds = ux.open_dataset(
        gridpath("ugrid", "quad-hexagon", "grid.nc"),
        datasetpath("ugrid", "quad-hexagon", "data.nc")
    )

    out = uxds['t2m']._repr_html_()

    assert out is not None


def test_grid_repr_bounds_dims():
    """repr(grid) should not hide existing "lat_lon" and "min_max" dims."""
    grid = ux.tutorial.open_grid("quad-hexagon")
    grid.bounds  # populate bounds dims
    assert "lat_lon" in grid._ds.dims
    assert "min_max" in grid._ds.dims
    repr_str = repr(grid)
    assert "lat_lon" in repr_str
    assert "min_max" in repr_str
    # additionally, grid.dims should contain these dims, too:
    assert "lat_lon" in grid.dims
    assert "min_max" in grid.dims
