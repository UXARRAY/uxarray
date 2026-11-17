"""Tests for dropping non-grid data during ``Grid`` construction (#1444, #1720)."""

import numpy as np
import pytest
import xarray as xr

import uxarray as ux
from uxarray.conventions.ugrid import DIM_NAMES
from uxarray.grid.utils import _drop_non_grid_data

TIMES = np.array(["2018-01-01", "2020-01-01"], dtype="datetime64[ns]")
STRAY_NAMES = {"time", "stray", "temp", "time_since_epoch", "special_index", "z"}

# the shapes listed in #1720
STRAY_CASES = {
    "scalar_coord": lambda ds: ds.assign_coords(time=np.datetime64("2016-01-01")),
    "dimension_coord": lambda ds: ds.assign_coords(time=("time", TIMES)),
    "bare_dimension": lambda ds: ds.assign(stray=("time", [0.0, 1.0])),
    "data_var_on_time_and_face": lambda ds: ds.assign(
        temp=(("time", "n_face"), np.ones((2, ds.sizes["n_face"])))
    ).assign_coords(time=("time", TIMES)),
    "non_dimension_coord": lambda ds: ds.assign_coords(
        time=("time", TIMES), time_since_epoch=("time", [0, 1])
    ),
    "coord_on_two_non_grid_dims": lambda ds: ds.assign_coords(
        special_index=(("time", "elevation"), np.arange(6).reshape(2, 3))
    ),
    "coord_mixing_grid_and_non_grid_dims": lambda ds: ds.assign_coords(
        z=(("n_node", "elevation"), np.zeros((ds.sizes["n_node"], 3)))
    ),
}


@pytest.fixture
def quad_hexagon_ds(gridpath):
    """Parsed quad-hexagon grid dataset (UGRID names and dims)."""
    return ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))._ds


@pytest.mark.parametrize("case", STRAY_CASES)
def test_stray_variables_are_dropped(quad_hexagon_ds, case):
    """Each stray ``time`` shape listed in #1720 should be gone after
    ``Grid.from_dataset``, with the grid itself unchanged."""
    uxgrid = ux.Grid.from_dataset(STRAY_CASES[case](quad_hexagon_ds))

    assert not STRAY_NAMES & set(uxgrid._ds.variables)
    assert set(uxgrid._ds.dims) <= set(DIM_NAMES)
    assert uxgrid.n_face == quad_hexagon_ds.sizes["n_face"]


def test_bounding_box_with_stray_grid_time(gridpath):
    """Subsetting must work when the grid was built from a source that carried a stray
    scalar ``time`` coordinate. Ensures issue #1444 has been fixed.
    """
    # grid built from a source carrying a stray scalar `time`
    grid_ds = xr.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"))
    uxgrid = ux.open_grid(grid_ds.assign_coords(time=np.datetime64("2016-01-01")))

    # the stray coordinate is dropped during grid construction
    assert "time" not in uxgrid._ds.coords

    # a face-centered variable carrying a *different* time coordinate
    times = np.array(["2018-01-01", "2020-01-01"], dtype="datetime64[ns]")
    uxda = ux.UxDataArray(
        data=np.ones((times.size, uxgrid.n_face)),
        dims=("time", "n_face"),
        coords={"time": times},
        uxgrid=uxgrid,
    )

    # previously raised: "IndexError: dimension coordinate 'time' conflicts ..."
    res = uxda.subset.bounding_box(lon_bounds=(-10, 10), lat_bounds=(-10, 10))

    assert isinstance(res, ux.UxDataArray)
    assert res.sizes["time"] == times.size
    assert "n_face" in res.dims


def test_scalar_time_data_var_does_not_block_subsetting(gridpath):
    """A scalar ``time`` stored as a data variable stays on the grid, but it isn't
    attached to the subset indexer, so subsetting keeps the data's own ``time``."""
    grid_ds = xr.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"))
    uxgrid = ux.open_grid(
        grid_ds.assign(time=xr.DataArray(np.datetime64("2016-01-01")))
    )
    uxda = ux.UxDataArray(
        np.ones((2, uxgrid.n_face)),
        dims=("time", "n_face"),
        coords={"time": TIMES},
        uxgrid=uxgrid,
    )

    res = uxda.subset.bounding_box((-10, 10), (-10, 10))

    np.testing.assert_array_equal(res["time"].values, TIMES)


def test_directly_built_grid_with_stray_time_can_be_subset(quad_hexagon_ds):
    """``ux.Grid(ds)`` doesn't filter, so a stray scalar ``time`` stays on the grid;
    subsetting data with a different ``time`` must still work."""
    ds = quad_hexagon_ds.assign_coords(time=np.datetime64("2016-01-01"))
    uxgrid = ux.Grid(ds, source_grid_spec="UGRID")
    uxda = ux.UxDataArray(
        np.ones((2, uxgrid.n_face)),
        dims=("time", "n_face"),
        coords={"time": TIMES},
        uxgrid=uxgrid,
    )

    res = uxda.subset.bounding_box((-10, 10), (-10, 10))

    np.testing.assert_array_equal(res["time"].values, TIMES)


def test_grid_coordinates_not_attached_to_data_subset(gridpath):
    """ICON's ``clon``/``clat`` are coordinates on the grid; slicing data shouldn't
    copy them onto the result."""
    uxgrid = ux.open_grid(gridpath("icon", "R02B04", "icon_grid_0010_R02B04_G.nc"))
    uxda = ux.UxDataArray(np.ones(uxgrid.n_face), dims=("n_face",), uxgrid=uxgrid)

    res = uxda.isel(n_face=[0, 1, 2])

    assert not {"clon", "clat"} & set(res.coords)


def test_single_file_with_grid_and_time_varying_data(gridpath, tmp_path):
    """A file holding the grid and data along ``time``: the grid gets neither."""
    raw = xr.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc")).load()
    single = raw.assign(
        temp=(("time", "n_face"), np.ones((TIMES.size, raw.sizes["n_face"])))
    ).assign_coords(time=("time", TIMES))
    path = tmp_path / "grid_and_data.nc"
    single.to_netcdf(path)

    uxds = ux.open_dataset(path)

    assert "temp" not in uxds.uxgrid._ds
    assert "time" not in uxds.uxgrid._ds.dims
    # the data keeps its own time axis
    assert "time" in uxds["temp"].dims

    sub = (
        uxds["temp"]
        .sel(time=TIMES[:1])
        .subset.bounding_box(lon_bounds=(-10, 10), lat_bounds=(-10, 10))
    )
    assert sub.sizes["time"] == 1
    uxds.close()


@pytest.mark.parametrize(
    "path, kept, dropped",
    [
        (("mpas", "QU", "oQU480.231010.nc"), ["boundary_node_indices"], []),
        (
            ("icon", "R02B04", "icon_grid_0010_R02B04_G.nc"),
            ["clon", "clat", "vlon", "vlat", "elon", "elat"],
            [],
        ),
        (
            ("ugrid", "fesom", "fesom.mesh.diag.nc"),
            ["edge_node_connectivity", "grid_topology"],
            ["nod_area", "nod_in_elem2D", "edge_cross_dxdy", "nz", "nz1"],
        ),
    ],
    ids=["mpas", "icon", "fesom"],
)
def test_grid_files(gridpath, path, kept, dropped):
    """Real grid files keep what uxarray reads from them (MPAS boundary nodes, ICON
    coordinates, FESOM2 connectivity); FESOM2's extras on non-grid dims are dropped."""
    uxgrid = ux.open_grid(gridpath(*path))

    assert set(kept) <= set(uxgrid._ds.variables)
    assert not set(dropped) & set(uxgrid._ds.variables)


def test_known_variable_on_misnamed_dim_kept(quad_hexagon_ds):
    """A known variable on a non-standard dim (the #1717 ``n2`` shape) is kept."""
    uxgrid = ux.Grid(quad_hexagon_ds, source_grid_spec="UGRID")
    ds = uxgrid._ds.assign(
        edge_node_connectivity=uxgrid.edge_node_connectivity.rename(two="n2")
    )

    cleaned = _drop_non_grid_data(ds)

    assert cleaned["edge_node_connectivity"].dims == ("n_edge", "n2")


def test_cached_values_kept_on_reread(gridpath):
    """Ensure cached data like ``bounds`` and ``max_face_radius`` survives writing the
    grid out with ``to_xarray()`` and reading it back, so it isn't recomputed."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    uxgrid.bounds
    uxgrid.face_bounds_lon
    uxgrid.face_bounds_lat
    uxgrid.max_face_radius
    cached = {"bounds", "face_bounds_lon", "face_bounds_lat", "max_face_radius"}

    reread = ux.open_grid(uxgrid.to_xarray())

    assert cached <= set(reread._ds.variables)


@pytest.mark.parametrize("as_coord", [False, True], ids=["data_var", "scalar_coord"])
def test_crs_variable_kept_for_projected_detection(gridpath, tmp_path, as_coord):
    """A CRS variable named by ``grid_mapping`` is kept, stored either as a scalar
    data variable (``crs``) or as a scalar coordinate (``spatial_ref``, as rioxarray
    writes it), so projected-grid detection still sees it."""
    crs_name = "spatial_ref" if as_coord else "crs"
    ds = xr.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc")).load()
    for name in ("node_lon", "node_lat"):
        ds[name].attrs = {"grid_mapping": crs_name}
    crs = xr.DataArray(0, attrs={"grid_mapping_name": "lambert_conformal_conic"})
    ds = ds.assign_coords({crs_name: crs}) if as_coord else ds.assign({crs_name: crs})
    path = tmp_path / "projected.nc"
    ds.to_netcdf(path)

    with pytest.warns(UserWarning, match="Projected"):
        uxgrid = ux.open_grid(path)

    assert crs_name in uxgrid._ds
    uxgrid._ds.close()
