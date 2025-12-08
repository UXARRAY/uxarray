import uxarray as ux
import pytest
import numpy as np
import xarray as xr

import numpy.testing as nt




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


def test_cross_section_cumulative_integrate(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("scrip", "ne30pg2", "grid.nc"), datasetpath("scrip", "ne30pg2", "data.nc"))
    cs = uxds['RELHUM'].cross_section(start=(-45, -45), end=(45, 45), steps=6)
    cs = cs.assign_coords(distance=("steps", np.linspace(0.0, 1.0, cs.sizes["steps"])))

    cs_ux = ux.UxDataArray(cs, uxgrid=uxds.uxgrid)

    result = cs_ux.cumulative_integrate(coord="distance")
    expected = cs.cumulative_integrate(coord="distance")

    assert isinstance(result, ux.UxDataArray)
    assert result.uxgrid == cs_ux.uxgrid
    xr.testing.assert_allclose(result.to_xarray(), expected)


def test_cumulative_integrate_requires_coord(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("scrip", "ne30pg2", "grid.nc"), datasetpath("scrip", "ne30pg2", "data.nc"))
    cs = uxds['RELHUM'].cross_section(start=(-45, -45), end=(45, 45), steps=3)
    cs_ux = ux.UxDataArray(cs, uxgrid=uxds.uxgrid)

    with pytest.raises(ValueError, match="Coordinate .* must be specified"):
        cs_ux.cumulative_integrate()
