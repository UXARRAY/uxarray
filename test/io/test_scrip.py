import os

import dask.array as da
import numpy as np
import numpy.testing as nt
import pytest
import xarray as xr

import uxarray as ux
from uxarray.constants import INT_DTYPE
from uxarray.errors import GridInvalidError
from uxarray.io._scrip import _detect_multigrid, _lookup_node_ids


def test_read_ugrid(gridpath, mesh_constants):
    """Reads a ugrid file."""
    uxgrid_ne30 = ux.open_grid(str(gridpath("ugrid", "outCSne30", "outCSne30.ug")))
    uxgrid_RLL1deg = ux.open_grid(str(gridpath("ugrid", "outRLL1deg", "outRLL1deg.ug")))
    uxgrid_RLL10deg_ne4 = ux.open_grid(str(gridpath("ugrid", "ov_RLL10deg_CSne4", "ov_RLL10deg_CSne4.ug")))

    nt.assert_equal(uxgrid_ne30.node_lon.size, mesh_constants['NNODES_outCSne30'])
    nt.assert_equal(uxgrid_RLL1deg.node_lon.size, mesh_constants['NNODES_outRLL1deg'])
    nt.assert_equal(uxgrid_RLL10deg_ne4.node_lon.size, mesh_constants['NNODES_ov_RLL10deg_CSne4'])

# TODO: UNCOMMENT
# def test_read_ugrid_opendap():
#     """Read an ugrid model from an OPeNDAP URL."""
#     try:
#         url = "http://test.opendap.org:8080/opendap/ugrid/NECOFS_GOM3_FORECAST.nc"
#         uxgrid_url = ux.open_grid(url, drop_variables="siglay")
#     except OSError:
#         warnings.warn(f'Could not connect to OPeNDAP server: {url}')
#         pass
#     else:
#         assert isinstance(getattr(uxgrid_url, "node_lon"), xr.DataArray)
#         assert isinstance(getattr(uxgrid_url, "node_lat"), xr.DataArray)
#         assert isinstance(getattr(uxgrid_url, "face_node_connectivity"), xr.DataArray)

def test_to_xarray_ugrid(gridpath):
    """Read an Exodus dataset and convert it to UGRID format using to_xarray."""
    ux_grid = ux.open_grid(gridpath("scrip", "outCSne8", "outCSne8.nc"))
    xr_obj = ux_grid.to_xarray("UGRID")
    xr_obj.to_netcdf("scrip_ugrid_csne8.nc")
    reloaded_grid = ux.open_grid("scrip_ugrid_csne8.nc")
    # Check that the grid topology is perfectly preserved
    nt.assert_array_equal(ux_grid.face_node_connectivity.values,
                          reloaded_grid.face_node_connectivity.values)

    # Check that node coordinates are numerically close
    nt.assert_allclose(ux_grid.node_lon.values, reloaded_grid.node_lon.values)
    nt.assert_allclose(ux_grid.node_lat.values, reloaded_grid.node_lat.values)

    # Cleanup
    reloaded_grid._ds.close()
    del reloaded_grid
    os.remove("scrip_ugrid_csne8.nc")


def test_oasis_multigrid_format_detection():
    """Detect OASIS-style multi-grid naming."""
    ds = xr.Dataset()
    ds["ocn.cla"] = xr.DataArray(np.random.rand(100, 4), dims=["nc_ocn", "nv_ocn"])
    ds["ocn.clo"] = xr.DataArray(np.random.rand(100, 4), dims=["nc_ocn", "nv_ocn"])
    ds["atm.cla"] = xr.DataArray(np.random.rand(200, 4), dims=["nc_atm", "nv_atm"])
    ds["atm.clo"] = xr.DataArray(np.random.rand(200, 4), dims=["nc_atm", "nv_atm"])

    format_type, grids = _detect_multigrid(ds)
    assert format_type == "multi_scrip"
    assert set(grids.keys()) == {"ocn", "atm"}


def test_open_multigrid_with_masks(gridpath):
    """Load OASIS multi-grids with masks applied."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    mask_file = gridpath("scrip", "oasis", "masks.nc")

    grids = ux.open_multigrid(grid_file, mask_filename=mask_file)
    assert grids["ocn"].n_face == 8
    assert grids["atm"].n_face == 20

    ocean_only = ux.open_multigrid(
        grid_file, gridnames=["ocn"], mask_filename=mask_file
    )
    assert set(ocean_only.keys()) == {"ocn"}
    assert ocean_only["ocn"].n_face == 8

    grid_names = ux.list_grid_names(grid_file)
    assert set(grid_names) == {"ocn", "atm"}


def test_open_multigrid_mask_active_value_default(gridpath):
    """Default mask semantics keep value==1 active for both grids."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    mask_file = gridpath("scrip", "oasis", "masks_no_atm.nc")

    grids = ux.open_multigrid(grid_file, mask_filename=mask_file)

    with xr.open_dataset(mask_file) as mask_ds:
        expected_ocn = int(mask_ds["ocn.msk"].values.sum())
        expected_atm = int(mask_ds["atm.msk"].values.sum())

    assert grids["ocn"].n_face == expected_ocn
    assert grids["atm"].n_face == expected_atm


def test_scrip_radians_units(gridpath):
    """SCRIP files with coordinates in radians are converted to degrees on load."""
    # scrip_radians.nc has a 2-cell grid whose lat/lon are stored in radians.
    # The expected degree values are: face_lon=[10, 20], face_lat=[30, 40].
    grid_file = gridpath("scrip", "scrip_radians", "scrip_radians_grid.nc")
    grid = ux.open_grid(grid_file)

    expected_face_lon = np.array([10.0, 20.0])
    expected_face_lat = np.array([30.0, 40.0])
    # 7 unique nodes: the two cells share only one corner point (15, 35)
    expected_node_lon = np.array([5., 5., 15., 15., 15., 25., 25.])
    expected_node_lat = np.array([25., 35., 25., 35., 45., 35., 45.])

    nt.assert_allclose(np.sort(grid.face_lon.values), np.sort(expected_face_lon), atol=1e-10)
    nt.assert_allclose(np.sort(grid.face_lat.values), np.sort(expected_face_lat), atol=1e-10)
    nt.assert_allclose(np.sort(grid.node_lon.values), np.sort(expected_node_lon), atol=1e-10)
    nt.assert_allclose(np.sort(grid.node_lat.values), np.sort(expected_node_lat), atol=1e-10)


@pytest.mark.parametrize(
    "chunks",
    [
        {"grid_size": 7},  # many tiny partitions
        {"grid_size": 10**6},  # one partition, larger than the grid
        "auto",  # what a caller (and the docstring) would reach for
        -1,
    ],
    ids=["tiny_chunks", "single_chunk", "auto", "minus_one"],
)
def test_scrip_dask_lazy_dedup_matches_eager(gridpath, chunks):
    """Opening a SCRIP grid with ``chunks=`` must produce the same mesh as
    the eager path, even though the underlying dedup algorithm differs
    (dask-native vs. Polars) once the corner arrays are dask-backed.

    The comparison is on corner coordinates *in winding order*, not sorted:
    the two paths are free to number nodes differently, but the order of
    corners within a face is what determines area sign and normal
    direction, so it must be preserved exactly.
    """
    grid_file = gridpath("scrip", "outCSne8", "outCSne8.nc")

    grid_eager = ux.open_grid(grid_file)
    grid_lazy = ux.open_grid(grid_file, chunks=chunks)

    assert isinstance(grid_lazy._ds["face_node_connectivity"].data, da.Array), (
        f"chunks={chunks!r} did not produce a dask-backed connectivity, so this "
        "test would silently compare the eager path against itself"
    )

    assert grid_eager.n_face == grid_lazy.n_face
    assert grid_eager.n_node == grid_lazy.n_node

    fnc_eager = grid_eager.face_node_connectivity.values
    fnc_lazy = grid_lazy.face_node_connectivity.values

    # Same set of unique nodes, independent of how each path numbered them.
    nodes_eager = np.unique(
        np.stack([grid_eager.node_lon.values, grid_eager.node_lat.values], 1), axis=0
    )
    nodes_lazy = np.unique(
        np.stack([grid_lazy.node_lon.values, grid_lazy.node_lat.values], 1), axis=0
    )
    nt.assert_allclose(nodes_eager, nodes_lazy, atol=1e-10)

    # Same corners per face, in the same order.
    nt.assert_allclose(
        grid_eager.node_lon.values[fnc_eager],
        grid_lazy.node_lon.values[fnc_lazy],
        atol=1e-10,
    )
    nt.assert_allclose(
        grid_eager.node_lat.values[fnc_eager],
        grid_lazy.node_lat.values[fnc_lazy],
        atol=1e-10,
    )


def test_scrip_dask_lazy_dedup_matches_eager_radians(gridpath):
    """The lazy path must also agree with the eager one when the file is in
    radians, i.e. when ``_values_in_degrees`` converts a dask array rather
    than returning it untouched."""
    grid_file = gridpath("scrip", "scrip_radians", "scrip_radians_grid.nc")

    grid_eager = ux.open_grid(grid_file)
    grid_lazy = ux.open_grid(grid_file, chunks={"grid_size": 3})

    assert isinstance(grid_lazy._ds["face_node_connectivity"].data, da.Array)
    assert grid_eager.n_node == grid_lazy.n_node

    fnc_eager = grid_eager.face_node_connectivity.values
    fnc_lazy = grid_lazy.face_node_connectivity.values
    nt.assert_allclose(
        grid_eager.node_lon.values[fnc_eager],
        grid_lazy.node_lon.values[fnc_lazy],
        atol=1e-10,
    )
    nt.assert_allclose(
        grid_eager.node_lat.values[fnc_eager],
        grid_lazy.node_lat.values[fnc_lazy],
        atol=1e-10,
    )


def test_lookup_node_ids_preserves_input_order():
    """``_lookup_node_ids`` must return ids positionally aligned with its
    inputs, and must find every corner.

    ``map_blocks`` splices each block's output back by position, so a lookup
    that reordered rows would build every face from the wrong corners -- a
    silently wrong mesh, not an error. The lookup is a binary search over the
    sorted unique-node arrays, so this also pins that the arrays it is handed
    really are sorted: an unsorted table would make searchsorted return
    nonsense and the guard inside would fire.
    """
    # sorted lexicographically by (lon, lat), as the dedup produces them
    unq_lon = np.array([10.0, 20.0, 30.0])
    unq_lat = np.array([1.0, 2.0, 3.0])

    # blocks repeat nodes, as a real corner table does, in neither sorted nor
    # lookup order
    lon_block = np.array([20.0, 30.0, 10.0, 20.0, 10.0])
    lat_block = np.array([2.0, 3.0, 1.0, 2.0, 1.0])

    ids = _lookup_node_ids(lon_block, lat_block, (unq_lon, unq_lat))

    nt.assert_array_equal(ids, np.array([1, 2, 0, 1, 0], dtype=INT_DTYPE))
    assert len(ids) == len(lon_block)

    # the property face_node_connectivity depends on: ids round-trip back to
    # the coordinates they came from
    nt.assert_allclose(unq_lon[ids], lon_block)
    nt.assert_allclose(unq_lat[ids], lat_block)


def test_lookup_node_ids_rejects_a_corner_it_cannot_find():
    """A corner absent from the unique table must raise, not guess.

    searchsorted returns an insertion point for a missing key rather than an
    error, so without the check a mismatch between the two halves of the
    dedup would map that corner to an arbitrary neighbouring node.
    """
    unq_lon = np.array([10.0, 20.0])
    unq_lat = np.array([1.0, 2.0])

    with pytest.raises(GridInvalidError, match="absent from the unique-node table"):
        _lookup_node_ids(np.array([15.0]), np.array([1.5]), (unq_lon, unq_lat))


def test_scrip_dask_dedup_does_not_materialize_corner_arrays(gridpath):
    """The point of the dask path is that the full corner table is never
    resident. Guard it: the connectivity must still be lazy after open, so
    a future refactor that quietly calls ``.compute()`` in the reader --
    reintroducing the OOM this path exists to avoid -- fails here rather
    than only on a multi-GB file nobody runs in CI.
    """
    grid = ux.open_grid(gridpath("scrip", "outCSne8", "outCSne8.nc"), chunks="auto")

    fnc = grid._ds["face_node_connectivity"].data
    assert isinstance(fnc, da.Array)
    assert fnc.npartitions >= 1


def test_open_multigrid_mask_active_value_per_grid_override(gridpath):
    """Per-grid override supports masks with different active values."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    mask_file = gridpath("scrip", "oasis", "masks_no_atm.nc")

    grids = ux.open_multigrid(
        grid_file,
        mask_filename=mask_file,
        mask_active_value={"atm": 0, "ocn": 1},
    )

    with xr.open_dataset(mask_file) as mask_ds:
        expected_ocn = int(mask_ds["ocn.msk"].values.sum())
        expected_atm = int((mask_ds["atm.msk"].values == 0).sum())

    assert grids["ocn"].n_face == expected_ocn
    assert grids["atm"].n_face == expected_atm
