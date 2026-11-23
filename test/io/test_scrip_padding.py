import numpy as np
import pytest
import xarray as xr

import uxarray as ux
from uxarray.constants import INT_FILL_VALUE


@pytest.mark.parametrize("chunks", [None, {}])
def test_scrip_decoded_padding_is_not_a_face_corner(tmp_path, chunks):
    fill = 1e20
    dataset = xr.Dataset(
        {
            "grid_center_lon": ("grid_size", [0.5, 1.33]),
            "grid_center_lat": ("grid_size", [0.5, 0.33]),
            "grid_corner_lon": (
                ("grid_size", "grid_corners"),
                [[0.0, 1, 1, 0], [1, 2, 1, fill]],
                {"units": "degrees", "_FillValue": fill},
            ),
            "grid_corner_lat": (
                ("grid_size", "grid_corners"),
                [[0.0, 0, 1, 1], [0, 0, 1, fill]],
                {"units": "degrees", "_FillValue": fill},
            ),
            "grid_imask": ("grid_size", np.array([1, 1], np.int32)),
            "grid_dims": ("grid_rank", np.array([2], np.int32)),
        }
    )
    path = tmp_path / "padded.nc"
    dataset.to_netcdf(path)
    grid = ux.open_grid(path, chunks=chunks)
    assert grid.face_node_connectivity.values[1, -1] == INT_FILL_VALUE
    np.testing.assert_array_equal(grid.n_nodes_per_face.values, [4, 3])
    assert np.isfinite(grid.face_areas.values).all()
    if chunks is not None:
        assert grid.face_node_connectivity.chunks is not None
