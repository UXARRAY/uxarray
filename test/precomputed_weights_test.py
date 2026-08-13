from pathlib import Path

import dask.array as da
import numpy as np
import numpy.testing as nt
import pytest
import uxarray as ux
import xarray as xr

from uxarray.remap import RemapWeights, clear_remap_weights_cache, load_remap_weights
from uxarray.remap.weights import _WEIGHTS_CACHE, _WEIGHTS_CACHE_MAXSIZE, _normalize_indices


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


def _write_dense_map(
    path: Path, source_size: int, destination_size: int, nnz_per_row: int = 3
) -> Path:
    """A map with several non-trivial weights per destination row.

    ``_write_sparse_map`` builds a pure permutation with unit weights, so the
    sparse multiply never sums anything. Use this when the summation itself
    matters.
    """
    rng = np.random.default_rng(0)
    rows = np.repeat(np.arange(1, destination_size + 1), nnz_per_row).astype(np.int32)
    cols = (rng.integers(0, source_size, size=destination_size * nnz_per_row) + 1).astype(
        np.int32
    )
    values = rng.random(destination_size * nnz_per_row)

    ds = xr.Dataset(
        data_vars={
            "row": (("n_s",), rows),
            "col": (("n_s",), cols),
            "S": (("n_s",), values),
        },
        coords={"n_s": np.arange(rows.size, dtype=np.int32)},
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

    weights = load_remap_weights(weight_file)
    result = weights._apply(np.arange(grid.n_face, dtype=np.float64))

    nt.assert_equal(weights.source_size, grid.n_face)
    nt.assert_equal(weights.destination_size, grid.n_face)
    nt.assert_array_equal(result, np.arange(grid.n_face, dtype=np.float64)[::-1])
    assert isinstance(weights, RemapWeights)


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

    remapped = source.remap.apply_weights(grid, weight_file)

    nt.assert_array_equal(remapped.values, source.values[::-1])
    nt.assert_equal(remapped.attrs["units"], "K")
    nt.assert_equal(remapped.uxgrid, grid)


def test_apply_weights_reuses_loaded_operator(tmp_path, gridpath):
    grid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))
    weight_file = _write_sparse_map(
        tmp_path / "reverse_map.nc", grid.n_face, grid.n_face
    )
    weights = load_remap_weights(weight_file)
    cached_weights = load_remap_weights(weight_file)

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

    remapped = source.remap.apply_weights(grid, weights)
    remapped_again = source["a"].remap.apply_weights(grid, weights)

    assert cached_weights is weights
    nt.assert_array_equal(remapped["a"].values, source["a"].values[:, ::-1])
    nt.assert_array_equal(remapped["flag"].values, source["flag"].values)
    nt.assert_array_equal(remapped_again.values, source["a"].values[:, ::-1])


def test_normalize_indices_respects_start_index_attr():
    # 0-based array with an explicit start_index=0 attr — must not shift.
    arr = xr.DataArray(np.array([0, 1, 2], dtype=np.int32), attrs={"start_index": 0})
    nt.assert_array_equal(_normalize_indices(arr, 4, "Row"), np.array([0, 1, 2]))

    # 1-based array with explicit start_index=1 attr.
    arr1 = xr.DataArray(np.array([1, 2, 3], dtype=np.int32), attrs={"start_index": 1})
    nt.assert_array_equal(_normalize_indices(arr1, 3, "Row"), np.array([0, 1, 2]))


def test_normalize_indices_partial_zero_based_not_shifted():
    # 0-based partial coverage: min=1, max < size. Previous heuristic
    # would have wrongly shifted to -1; new heuristic keeps as 0-based.
    arr = np.array([1, 2, 3], dtype=np.int32)
    nt.assert_array_equal(_normalize_indices(arr, 10, "Row"), arr)


def test_normalize_indices_one_based_detected_by_max():
    arr = np.array([1, 2, 3, 4], dtype=np.int32)
    nt.assert_array_equal(
        _normalize_indices(arr, 4, "Row"), np.array([0, 1, 2, 3])
    )


def test_normalize_indices_out_of_bounds_raises():
    with pytest.raises(ValueError, match="out of bounds"):
        _normalize_indices(np.array([-1, 0, 1]), 4, "Row")


def test_apply_weights_rejects_non_spatial_source_dim(tmp_path, gridpath):
    grid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))
    weight_file = _write_sparse_map(
        tmp_path / "reverse_map.nc", grid.n_face, grid.n_face
    )

    source = ux.UxDataArray(
        xr.DataArray(
            np.arange(grid.n_face, dtype=np.float64),
            dims=["n_face"],
            name="t",
        ),
        uxgrid=grid,
    )

    with pytest.raises(ValueError, match="not a spatial dimension"):
        source.remap.apply_weights(grid, weight_file, source_dim="time")


def test_apply_weights_preserves_aux_coords(tmp_path, gridpath):
    grid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))
    weight_file = _write_sparse_map(
        tmp_path / "reverse_map.nc", grid.n_face, grid.n_face
    )

    nt_steps = 3
    da = xr.DataArray(
        np.arange(nt_steps * grid.n_face, dtype=np.float64).reshape(
            nt_steps, grid.n_face
        ),
        dims=("time", "n_face"),
        coords={
            "time": np.array([10, 20, 30], dtype=np.int64),
            "time_label": ("time", np.array(["a", "b", "c"])),
        },
        name="t",
    )
    source = ux.UxDataArray(da, uxgrid=grid)
    remapped = source.remap.apply_weights(grid, weight_file)
    assert "time_label" in remapped.coords
    nt.assert_array_equal(remapped["time_label"].values, np.array(["a", "b", "c"]))


def test_clear_remap_weights_cache(tmp_path, gridpath):
    grid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))
    weight_file = _write_sparse_map(
        tmp_path / "reverse_map.nc", grid.n_face, grid.n_face
    )
    load_remap_weights(weight_file)
    assert len(_WEIGHTS_CACHE) > 0
    clear_remap_weights_cache()
    assert len(_WEIGHTS_CACHE) == 0


def test_remap_weights_cache_is_lru_bounded(tmp_path, gridpath):
    grid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))
    clear_remap_weights_cache()
    for i in range(_WEIGHTS_CACHE_MAXSIZE + 5):
        path = tmp_path / f"map_{i}.nc"
        _write_sparse_map(path, grid.n_face, grid.n_face)
        load_remap_weights(path)
    assert len(_WEIGHTS_CACHE) == _WEIGHTS_CACHE_MAXSIZE


def test_apply_weights_dask_reproduces_numpy(tmp_path, gridpath):
    # the numpy (eager) and dask (blockwise) branches must agree. The source dim
    # is forced into a single chunk, so the sparse multiply is never split and
    # agreement must be exact.
    grid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    weight_file = _write_dense_map(tmp_path / "dense_map.nc", grid.n_face, grid.n_face)

    rng = np.random.default_rng(1)
    source = ux.UxDataArray(
        xr.DataArray(
            rng.random((6, grid.n_face)),
            dims=["time", "n_face"],
            name="temperature",
            attrs={"units": "K"},
        ),
        uxgrid=grid,
    )

    numpy_result = source.remap.apply_weights(grid, weight_file)
    assert isinstance(numpy_result.data, np.ndarray)

    # including chunkings of the source dim, which _apply_weights must rechunk away
    for chunks in ({"time": 2}, {"n_face": 50}, {"time": 2, "n_face": 50}):
        dask_result = source.chunk(chunks).remap.apply_weights(grid, weight_file)

        assert isinstance(dask_result.data, da.Array)

        assert numpy_result.dims == dask_result.dims
        assert numpy_result.dtype == dask_result.dtype
        nt.assert_array_equal(numpy_result.values, dask_result.values)
        nt.assert_equal(dask_result.attrs["units"], "K")
