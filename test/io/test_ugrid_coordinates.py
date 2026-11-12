import numpy as np
import pytest
import xarray as xr

import uxarray as ux
from uxarray.errors import GridInvalidError


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


@pytest.mark.parametrize("metadata", ["standard_name", "m", "feet", "km"])
def test_projected_nodes_are_not_interpreted_as_degrees(tmp_path, metadata):
    ds = _mesh()
    ds.x.values[:] = [500000, 500100, 500000]
    ds.y.values[:] = [4000000, 4000000, 4000100]
    if metadata == "standard_name":
        ds.x.attrs[metadata] = "projection_x_coordinate"
        ds.y.attrs[metadata] = "projection_y_coordinate"
    else:
        ds.x.attrs["units"] = ds.y.attrs["units"] = metadata
    path = tmp_path / "utm.nc"
    ds.to_netcdf(path)
    for source in (ds, path):
        with pytest.raises(GridInvalidError, match="Projected coordinates.*provide geographic"):
            ux.open_grid(source)


def test_projected_coordinates_retained_with_geographic_nodes():
    ds = _mesh()
    ds.x.attrs = {"standard_name": "projection_x_coordinate", "units": "m"}
    ds.y.attrs = {"standard_name": "projection_y_coordinate", "units": "m"}
    ds["lon"] = ("nodes", [0.0, 1.0, 0.0], {"standard_name": "longitude"})
    ds["lat"] = ("nodes", [0.0, 0.0, 1.0], {"standard_name": "latitude"})
    ds.mesh.attrs["node_coordinates"] = "x y lat lon"
    ds["center_x"] = ("cells", [500000.0], {"standard_name": "projection_x_coordinate"})
    ds["center_y"] = ("cells", [4000000.0], {"standard_name": "projection_y_coordinate"})
    ds.mesh.attrs["face_coordinates"] = "center_x center_y"
    grid = ux.open_grid(ds)
    np.testing.assert_array_equal(grid._ds.x.values, ds.x.values)
    np.testing.assert_array_equal(grid._ds.center_x.values, ds.center_x.values)
    assert "face_lon" not in grid._ds
    assert "node_x" not in grid._ds
    assert np.isfinite(grid.face_areas.values).all()


def test_legacy_geographic_coordinates_without_metadata():
    ds = _mesh()
    grid = ux.open_grid(ds)
    np.testing.assert_array_equal(grid.node_lon.values, ds.x.values)
    np.testing.assert_array_equal(grid.node_lat.values, ds.y.values)
