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
