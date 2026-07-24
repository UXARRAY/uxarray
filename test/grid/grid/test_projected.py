"""Tests for projected (non-spherical) coordinate detection and safe loading."""

import numpy as np
import pytest
import xarray as xr

import uxarray as ux


def _make_grid(stdname, units, values=None):
    if values is None:
        values = np.array([1_500_000.0, 1_500_100.0, 1_500_100.0, 1_500_000.0])
    node_lon = xr.DataArray(
        values, dims=["n_node"], attrs={"standard_name": stdname, "units": units}
    )
    node_lat = xr.DataArray(
        np.array([800_000.0, 800_000.0, 800_100.0, 800_100.0]), dims=["n_node"]
    )
    fnc = xr.DataArray(
        np.array([[0, 1, 2], [0, 2, 3]]),
        dims=["n_face", "n_max_face_nodes"],
        attrs={"cf_role": "face_node_connectivity", "start_index": 0, "_FillValue": -1},
    )
    return xr.Dataset(
        {"node_lon": node_lon, "node_lat": node_lat, "face_node_connectivity": fnc}
    )


def test_projected_coordinates_not_wrapped():
    """Meter-scale coordinates with standard_name=projection_x_coordinate must not be wrapped."""
    original = np.array([1_500_000.0, 1_500_100.0, 1_500_100.0, 1_500_000.0])
    ds = _make_grid("projection_x_coordinate", "m")
    with pytest.warns(UserWarning, match="Projected"):
        grid = ux.Grid(ds, source_grid_spec="UGRID")
    np.testing.assert_array_equal(grid.node_lon.values, original)


def test_projected_detected_from_units():
    """units='m' with no standard_name is sufficient to detect projected coords."""
    ds = _make_grid("", "m")
    ds["node_lon"].attrs.pop("standard_name", None)
    with pytest.warns(UserWarning, match="Projected"):
        ux.Grid(ds, source_grid_spec="UGRID")


def test_projected_detected_from_grid_mapping():
    """A grid_mapping variable with a non-latlon name signals projected coords."""
    ds = _make_grid("", "")
    ds["node_lon"].attrs = {"grid_mapping": "crs"}
    ds["crs"] = xr.DataArray(
        np.int32(0), attrs={"grid_mapping_name": "lambert_conformal_conic"}
    )
    with pytest.warns(UserWarning, match="Projected"):
        ux.Grid(ds, source_grid_spec="UGRID")


def test_geographic_coordinates_still_wrapped():
    """Geographic [0, 360] coords must still be normalized to [-180, 180]."""
    ds = _make_grid(
        "longitude",
        "degrees_east",
        values=np.array([270.0, 271.0, 271.0, 270.0]),
    )
    ds["node_lat"] = xr.DataArray(np.array([43.0, 43.0, 44.0, 44.0]), dims=["n_node"])
    grid = ux.Grid(ds, source_grid_spec="UGRID")
    assert grid.node_lon.values.max() <= 180.0
