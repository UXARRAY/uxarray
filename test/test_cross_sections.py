
import dask.array as da
import pytest
import numpy as np
import numpy.testing as nt
import xarray as xr
import uxarray as ux

from uxarray.constants import ERROR_TOLERANCE
from uxarray.grid.intersections import constant_lat_intersections_face_bounds


def test_constant_lat_subset_grid(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))

    grid_top_two = uxgrid.subset.constant_latitude(lat=0.1)
    assert grid_top_two.n_face == 2

    grid_bottom_two = uxgrid.subset.constant_latitude(lat=-0.1)
    assert grid_bottom_two.n_face == 2

    grid_all_four = uxgrid.subset.constant_latitude(lat=0.0)
    assert grid_all_four.n_face == 4

    with pytest.raises(ValueError):
        uxgrid.subset.constant_latitude(lat=10.0)

def test_constant_lon_subset_grid(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))

    grid_left_two = uxgrid.subset.constant_longitude(lon=-0.1)
    assert grid_left_two.n_face == 2

    grid_right_two = uxgrid.subset.constant_longitude(lon=0.2)
    assert grid_right_two.n_face == 2

    with pytest.raises(ValueError):
        uxgrid.subset.constant_longitude(lon=10.0)

def test_constant_lat_subset_uxds(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))
    uxds.uxgrid.normalize_cartesian_coordinates()

    da_top_two = uxds['t2m'].subset.constant_latitude(lat=0.1)
    np.testing.assert_array_equal(da_top_two.data, uxds['t2m'].isel(n_face=[1, 2]).data)

    da_bottom_two = uxds['t2m'].subset.constant_latitude(lat=-0.1)
    np.testing.assert_array_equal(da_bottom_two.data, uxds['t2m'].isel(n_face=[0, 3]).data)

    da_all_four = uxds['t2m'].subset.constant_latitude(lat=0.0)
    np.testing.assert_array_equal(da_all_four.data, uxds['t2m'].data)

    with pytest.raises(ValueError):
        uxds['t2m'].subset.constant_latitude(lat=10.0)

def test_constant_lon_subset_uxds(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))
    uxds.uxgrid.normalize_cartesian_coordinates()

    da_left_two = uxds['t2m'].subset.constant_longitude(lon=-0.1)
    np.testing.assert_array_equal(da_left_two.data, uxds['t2m'].isel(n_face=[0, 2]).data)

    da_right_two = uxds['t2m'].subset.constant_longitude(lon=0.2)
    np.testing.assert_array_equal(da_right_two.data, uxds['t2m'].isel(n_face=[1, 3]).data)

    with pytest.raises(ValueError):
        uxds['t2m'].subset.constant_longitude(lon=10.0)

def test_north_pole(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    lats = [89.85, 89.9, 89.95, 89.99]

    for lat in lats:
        cross_grid = uxgrid.subset.constant_latitude(lat=lat)
        assert cross_grid.n_face == 4

def test_south_pole(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    lats = [-89.85, -89.9, -89.95, -89.99]

    for lat in lats:
        cross_grid = uxgrid.subset.constant_latitude(lat=lat)
        assert cross_grid.n_face == 4

def test_constant_lat():
    bounds = np.array([
        [[-45, 45], [0, 360]],
        [[-90, -45], [0, 360]],
        [[45, 90], [0, 360]],
    ])
    bounds_rad = np.deg2rad(bounds)
    const_lat = 0

    candidate_faces = constant_lat_intersections_face_bounds(
        lat=const_lat,
        face_bounds_lat=bounds_rad[:, 0],
    )

    expected_faces = np.array([0])
    np.testing.assert_array_equal(candidate_faces, expected_faces)

def test_constant_lat_out_of_bounds():
    bounds = np.array([
        [[-45, 45], [0, 360]],
        [[-90, -45], [0, 360]],
        [[45, 90], [0, 360]],
    ])
    bounds_rad = np.deg2rad(bounds)
    const_lat = 100

    candidate_faces = constant_lat_intersections_face_bounds(
        lat=const_lat,
        face_bounds_lat=bounds_rad[:, 0],
    )

    assert len(candidate_faces) == 0


def test_edges_at_constant_latitude(gridpath):
    """``get_edges_at_constant_latitude`` returns exactly the edges whose endpoints straddle the latitude."""
    uxgrid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))

    edges = uxgrid.get_edges_at_constant_latitude(lat=0.0)

    # Derived from the node coordinates rather than from the screener: an edge
    # meets the equator when its two endpoints sit on opposite sides of z = 0,
    # or when both lie on it.
    edge_node_z = uxgrid.node_z.values[uxgrid.edge_node_connectivity.values]
    z0, z1 = edge_node_z[:, 0], edge_node_z[:, 1]
    expected = np.flatnonzero(
        (z0 * z1 < 0.0)
        | ((np.abs(z0) < ERROR_TOLERANCE) & (np.abs(z1) < ERROR_TOLERANCE))
    )

    assert len(expected) > 0
    nt.assert_array_equal(edges, expected)


def test_edges_at_constant_lat_lon_chunked_grid(gridpath):
    """Both edge queries run on a dask-backed grid and agree with the in-memory result."""
    da = pytest.importorskip("dask.array")
    gridfile = gridpath("ugrid", "quad-hexagon", "grid.nc")

    uxgrid = ux.open_grid(gridfile)
    expected_lat = uxgrid.get_edges_at_constant_latitude(lat=0.0)
    expected_lon = uxgrid.get_edges_at_constant_longitude(lon=0.0)

    chunked = ux.open_grid(gridfile)

    # Construct the derived variables first so that chunk() converts them too,
    # leaving both the connectivity and the node coordinates dask-backed.
    _ = chunked.edge_node_connectivity
    _ = chunked.node_x, chunked.node_y, chunked.node_z
    chunked.chunk(n_node=2, n_edge=4, n_face=2)

    assert isinstance(chunked.edge_node_connectivity.data, da.Array)
    for coord in (chunked.node_x, chunked.node_y, chunked.node_z):
        assert isinstance(coord.data, da.Array)

    nt.assert_array_equal(
        chunked.get_edges_at_constant_latitude(lat=0.0), expected_lat
    )
    nt.assert_array_equal(
        chunked.get_edges_at_constant_longitude(lon=0.0), expected_lon
    )


def test_const_lat_interval_da(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))
    uxds.uxgrid.normalize_cartesian_coordinates()

    res = uxds['t2m'].subset.constant_latitude_interval(lats=(-10, 10))

    assert len(res) == 4


def test_const_lat_interval_grid(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))

    res = uxgrid.subset.constant_latitude_interval(lats=(-10, 10))

    assert res.n_face == 4

    res, indices = uxgrid.subset.constant_latitude_interval(lats=(-10, 10), return_face_indices=True)

    assert len(indices) == 4

def test_const_lon_interva_da(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))
    uxds.uxgrid.normalize_cartesian_coordinates()

    res = uxds['t2m'].subset.constant_longitude_interval(lons=(-10, 10))

    assert len(res) == 4


def test_const_lon_interval_grid(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))

    res = uxgrid.subset.constant_longitude_interval(lons=(-10, 10))

    assert res.n_face == 4

    res, indices = uxgrid.subset.constant_longitude_interval(lons=(-10, 10), return_face_indices=True)

    assert len(indices) == 4


class TestArcs:
    def test_latitude_along_arc(self):
        node_lon = np.array([-40, -40, 40, 40])
        node_lat = np.array([-20, 20, 20, -20])
        face_node_connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)

        uxgrid = ux.Grid.from_topology(node_lon, node_lat, face_node_connectivity)

        # intersection at exactly 20 degrees latitude
        out1 = uxgrid.get_faces_at_constant_latitude(lat=20)

        # intersection at 25.41 degrees latitude (max along the great circle arc)
        out2 = uxgrid.get_faces_at_constant_latitude(lat=25.41)

        nt.assert_array_equal(out1, out2)



def test_double_subset(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))

    # construct edges
    sub_lat = uxgrid.subset.constant_latitude(0.0)

    sub_lat_lon = sub_lat.subset.constant_longitude(0.0)

    assert "n_edge" not in sub_lat_lon._ds.dims

    _ = uxgrid.face_edge_connectivity
    _ = uxgrid.edge_node_connectivity
    _ = uxgrid.edge_lon

    sub_lat = uxgrid.subset.constant_latitude(0.0)

    sub_lat_lon = sub_lat.subset.constant_longitude(0.0)

    assert "n_edge" in sub_lat_lon._ds.dims


def test_cross_section(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("scrip", "ne30pg2", "grid.nc"), datasetpath("scrip", "ne30pg2", "data.nc"))

    # Tributary GCA
    ss_gca = uxds['RELHUM'].cross_section(start=(-45, -45), end=(45, 45))
    assert isinstance(ss_gca, xr.DataArray)

    # Constant Latitude
    ss_clat = uxds['RELHUM'].cross_section(lat=45)
    assert isinstance(ss_clat, xr.DataArray)

    # Constant Longitude
    ss_clon = uxds['RELHUM'].cross_section(lon=45)
    assert isinstance(ss_clon, xr.DataArray)

    # Constant Longitude with increased samples
    ss_clon = uxds['RELHUM'].cross_section(lon=45, steps=3)
    assert isinstance(ss_clon, xr.DataArray)


    with pytest.raises(ValueError):
        _ = uxds['RELHUM'].cross_section(end=(45, 45))
        _ = uxds['RELHUM'].cross_section(start=(45, 45))
        _ = uxds['RELHUM'].cross_section(lon=45, end=(45, 45))
        _ = uxds['RELHUM'].cross_section()


CROSS_SECTION_MODES = [
    dict(start=(-45, -45), end=(45, 45)),
    dict(lat=45),
    dict(lon=45),
    dict(lon=45, steps=3),
]


def test_cross_section_dask_reproduces_numpy(gridpath, datasetpath):
    # the numpy (eager) and dask (lazy gather) branches must agree
    uxds = ux.open_dataset(gridpath("scrip", "ne30pg2", "grid.nc"), datasetpath("scrip", "ne30pg2", "data.nc"))
    uxda = uxds['RELHUM']  # ('lev', 'n_face'), so the face dim is not the leading one

    for kwargs in CROSS_SECTION_MODES:
        numpy_result = uxda.cross_section(**kwargs)

        # chunk the face dim as well: the gather must not rely on n_face being whole
        for chunks in ({"lev": 8}, {"n_face": 4000}, {"lev": 8, "n_face": 4000}):
            dask_result = uxda.chunk(chunks).cross_section(**kwargs)

            # the accessor no longer calls .compute(), so the result stays lazy
            assert isinstance(dask_result.data, da.Array)

            assert numpy_result.dims == dask_result.dims
            assert numpy_result.dtype == dask_result.dtype
            nt.assert_array_equal(numpy_result.values, dask_result.values)
            nt.assert_array_equal(numpy_result.lat.values, dask_result.lat.values)
            nt.assert_array_equal(numpy_result.lon.values, dask_result.lon.values)


def test_cross_section_dask_reproduces_numpy_partial_coverage(gridpath, datasetpath):
    # steps with no containing face become NaN; that fill path must match too
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"),
                           datasetpath("ugrid", "quad-hexagon", "multi_dim_data.nc"))
    uxda = uxds['multi_dim_data']  # ('time', 'lev', 'n_face') over a 4-face regional patch

    # arc starts inside the patch and leaves it, so the result mixes data with NaN
    kwargs = dict(start=(-0.043, -0.112), end=(5, 5))
    numpy_result = uxda.cross_section(**kwargs)
    nan_mask = np.isnan(numpy_result.values)
    assert nan_mask.any() and not nan_mask.all()

    for chunks in ({"time": 2}, {"time": 2, "lev": 3}, {"time": 2, "n_face": 1}):
        dask_result = uxda.chunk(chunks).cross_section(**kwargs)

        assert isinstance(dask_result.data, da.Array)

        assert numpy_result.dims == dask_result.dims
        assert numpy_result.dtype == dask_result.dtype
        nt.assert_array_equal(numpy_result.values, dask_result.values)

    # 'n_face' in a leading position: 'steps' must land in its place. This is the
    # case the replaced numpy path handled with an explicit moveaxis round-trip.
    leading = uxda.transpose("n_face", "time", "lev")
    leading_numpy = leading.cross_section(**kwargs)
    leading_dask = leading.chunk({"time": 2}).cross_section(**kwargs)

    assert leading_numpy.dims == ("steps", "time", "lev")
    assert leading_numpy.dims == leading_dask.dims
    nt.assert_array_equal(leading_numpy.values, leading_dask.values)
    nt.assert_array_equal(
        leading_numpy.transpose(*numpy_result.dims).values, numpy_result.values
    )
