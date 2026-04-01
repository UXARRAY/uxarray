from pathlib import Path

import numpy as np
import numpy.testing as nt
import uxarray as ux
import xarray as xr


def _write_sparse_map(path: Path, source_size: int, destination_size: int) -> Path:
    rows = np.arange(1, destination_size + 1, dtype=np.int32)
    cols = np.arange(source_size, 0, -1, dtype=np.int32)
    values = np.ones(destination_size, dtype=np.float64)

    ds = xr.Dataset(
        data_vars={
            "row": (("n_s",), rows),
            "col": (("n_s",), cols),
            "S": (("n_s",), values),
        },
        coords={"n_s": np.arange(destination_size, dtype=np.int32)},
    )
    ds = ds.assign_coords(
        n_a=np.arange(source_size, dtype=np.int32),
        n_b=np.arange(destination_size, dtype=np.int32),
    )
    ds.to_netcdf(path)
    return path


def test_load_remap_weights_and_apply_vector(tmp_path, gridpath):
    grid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))
    weight_file = _write_sparse_map(
        tmp_path / "reverse_map.nc", grid.n_face, grid.n_face
    )

    weights = ux.load_remap_weights(weight_file)
    result = weights.apply(np.arange(grid.n_face, dtype=np.float64))

    nt.assert_equal(weights.source_size, grid.n_face)
    nt.assert_equal(weights.destination_size, grid.n_face)
    nt.assert_array_equal(result, np.arange(grid.n_face, dtype=np.float64)[::-1])
    assert isinstance(weights, ux.RemapWeights)


def test_apply_weights_to_uxdataarray(tmp_path, gridpath):
    grid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))
    weight_file = _write_sparse_map(
        tmp_path / "reverse_map.nc", grid.n_face, grid.n_face
    )

    source = ux.UxDataArray(
        xr.DataArray(
            np.arange(grid.n_face, dtype=np.float64),
            dims=["n_face"],
            name="temperature",
            attrs={"units": "K"},
        ),
        uxgrid=grid,
    )

    remapped = source.remap.apply_weights(weight_file, grid)

    nt.assert_array_equal(remapped.values, source.values[::-1])
    nt.assert_equal(remapped.attrs["units"], "K")
    nt.assert_equal(remapped.uxgrid, grid)


def test_apply_weights_reuses_loaded_operator(tmp_path, gridpath):
    grid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))
    weight_file = _write_sparse_map(
        tmp_path / "reverse_map.nc", grid.n_face, grid.n_face
    )
    weights = ux.load_remap_weights(weight_file)
    cached_weights = ux.load_remap_weights(weight_file)

    source = ux.UxDataset(
        xr.Dataset(
            data_vars={
                "a": (
                    ("time", "n_face"),
                    np.arange(2 * grid.n_face).reshape(2, grid.n_face),
                ),
                "flag": (("time",), np.array([1, 0], dtype=np.int32)),
            },
            coords={"time": np.array([0, 1], dtype=np.int32)},
        ),
        uxgrid=grid,
    )

    remapped = source.remap.apply_weights(weights, grid)
    remapped_again = source["a"].remap.apply_weights(weights, grid)

    assert cached_weights is weights
    nt.assert_array_equal(remapped["a"].values, source["a"].values[:, ::-1])
    nt.assert_array_equal(remapped["flag"].values, source["flag"].values)
    nt.assert_array_equal(remapped_again.values, source["a"].values[:, ::-1])
