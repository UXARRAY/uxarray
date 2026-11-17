import numpy as np
import pytest
import xarray as xr

import uxarray as ux


def _mesh():
    return xr.Dataset(
        {
            "mesh": ((), 0, {
                "cf_role": "mesh_topology", "topology_dimension": 2,
                "node_coordinates": "x y", "face_node_connectivity": "faces",
            }),
            "x": ("nodes", [0.0, 1.0, 0.0]),
            "y": ("nodes", [0.0, 0.0, 1.0]),
            "faces": (("cells", "corners"), [[0, 1, 2]], {"start_index": 0}),
        }
    )


@pytest.mark.parametrize("chunks", [None, {}])
def test_geographic_coordinates_selected_by_standard_name(tmp_path, chunks):
    ds = _mesh()
    ds.x.attrs["standard_name"] = "longitude"
    ds.y.attrs["standard_name"] = "latitude"
    ds.mesh.attrs["node_coordinates"] = "y x"
    ds["center_x"] = ("cells", [0.33], {"standard_name": "longitude"})
    ds["center_y"] = ("cells", [0.33], {"standard_name": "latitude"})
    ds.mesh.attrs["face_coordinates"] = "center_y center_x"
    path = tmp_path / "geographic.nc"
    ds.to_netcdf(path)
    grid = ux.open_grid(path, chunks=chunks)
    np.testing.assert_array_equal(grid.node_lon.values, ds.x.values)
    np.testing.assert_array_equal(grid.node_lat.values, ds.y.values)
    np.testing.assert_array_equal(grid.face_lon.values, ds.center_x.values)
    assert "x" in ds and "node_lon" not in ds
    if chunks is not None:
        assert grid.node_lon.chunks is not None


@pytest.mark.parametrize("metadata", ["standard_name", "m", "feet", "km", "mapping"])
def test_projected_nodes_load_with_warning(tmp_path, metadata):
    ds = _mesh()
    ds.x.values[:] = [500000, 500100, 500000]
    ds.y.values[:] = [4000000, 4000000, 4000100]
    if metadata == "standard_name":
        ds.x.attrs[metadata] = "projection_x_coordinate"
        ds.y.attrs[metadata] = "projection_y_coordinate"
    elif metadata == "mapping":
        ds["crs"] = ((), 0, {"grid_mapping_name": "lambert_conformal_conic"})
        ds.x.attrs["grid_mapping"] = ds.y.attrs["grid_mapping"] = "crs"
    else:
        ds.x.attrs["units"] = ds.y.attrs["units"] = metadata
    path = tmp_path / "utm.nc"
    ds.to_netcdf(path)
    for source in (ds, path):
        with pytest.warns(UserWarning, match="Projected"):
            grid = ux.open_grid(source)
            grid.node_lon  # noqa: B018
        np.testing.assert_array_equal(grid.node_lon.values, ds.x.values)


def test_projected_coordinates_retained_with_geographic_nodes():
    ds = _mesh()
    ds = ds.rename(x="node_x", y="node_y")
    ds.node_x.attrs = {"standard_name": "projection_x_coordinate", "units": "m"}
    ds.node_y.attrs = {"standard_name": "projection_y_coordinate", "units": "m"}
    ds["lon"] = ("nodes", [0.0, 1.0, 0.0], {"standard_name": "longitude"})
    ds["lat"] = ("nodes", [0.0, 0.0, 1.0], {"standard_name": "latitude"})
    ds.mesh.attrs["node_coordinates"] = "node_x node_y lat lon"
    ds["face_x"] = ("cells", [500000.0], {"standard_name": "projection_x_coordinate"})
    ds["face_y"] = ("cells", [4000000.0], {"standard_name": "projection_y_coordinate"})
    ds.mesh.attrs["face_coordinates"] = "face_x face_y"
    grid = ux.open_grid(ds)
    np.testing.assert_array_equal(grid._ds.node_x_projected.values, ds.node_x.values)
    np.testing.assert_array_equal(grid._ds.face_x_projected.values, ds.face_x.values)
    assert "node_x" not in grid._ds and "face_x" not in grid._ds
    assert np.isfinite(grid.face_areas.values).all()
    assert 0 < grid.node_x.values.max() <= 1
    assert 0 < grid.face_lon.values[0] < 1


def test_legacy_geographic_coordinates_without_metadata():
    ds = _mesh()
    grid = ux.open_grid(ds)
    np.testing.assert_array_equal(grid.node_lon.values, ds.x.values)
    np.testing.assert_array_equal(grid.node_lat.values, ds.y.values)


@pytest.mark.parametrize("units", ["deg", "degrees", "degree_east", "radians", "rad"])
@pytest.mark.parametrize("chunks", [None, {}])
def test_angular_units_are_accepted_and_radians_converted(units, chunks, tmp_path):
    ds = _mesh()
    ds.x.attrs["units"] = ds.y.attrs["units"] = units
    expected = ds.x.values.copy()
    if units in ("radians", "rad"):
        ds.x.values[:] = np.deg2rad(ds.x.values)
        ds.y.values[:] = np.deg2rad(ds.y.values)
    path = tmp_path / "angular.nc"
    ds.to_netcdf(path)
    grid = ux.open_grid(path, chunks=chunks)
    np.testing.assert_allclose(grid.node_lon.values, expected)
    np.testing.assert_allclose(grid.node_lat.values, ds.y.values if units not in ("radians", "rad") else np.rad2deg(ds.y.values))
    if chunks is not None:
        assert grid.node_lon.chunks is not None


def test_partial_geographic_metadata_uses_positional_fallback():
    ds = _mesh()
    ds.x.attrs["standard_name"] = "longitude"
    ds.y.attrs["units"] = "degrees_north"
    grid = ux.open_grid(ds)
    np.testing.assert_array_equal(grid.node_lon.values, ds.x.values)
    np.testing.assert_array_equal(grid.node_lat.values, ds.y.values)


def test_projected_face_coordinates_do_not_break_face_centers():
    ds = _mesh()
    ds["face_x"] = ("cells", [500000.0], {"standard_name": "projection_x_coordinate"})
    ds["face_y"] = ("cells", [4000000.0], {"standard_name": "projection_y_coordinate"})
    ds.mesh.attrs["face_coordinates"] = "face_x face_y"
    grid = ux.open_grid(ds)
    assert 0 < grid.face_lon.values[0] < 1


@pytest.mark.parametrize(
    "units, used", [("deg", True), ("degrees_east", True), ("m", False)]
)
def test_face_coordinates_used_only_when_geographic(units, used):
    ds = _mesh()
    ds["cx"] = ("cells", [0.3], {"units": units})
    ds["cy"] = ("cells", [0.3], {"units": units})
    ds.mesh.attrs["face_coordinates"] = "cx cy"
    grid = ux.open_grid(ds)
    assert (grid.face_lon.values[0] == 0.3) == used
