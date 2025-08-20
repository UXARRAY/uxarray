import os
import numpy as np
import numpy.testing as nt
import xarray as xr
import pytest

from pathlib import Path

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.coordinates import _populate_node_latlon, _lonlat_rad_to_xyz, _normalize_xyz, _xyz_to_lonlat_rad, \
    _xyz_to_lonlat_deg, _xyz_to_lonlat_rad_scalar
from uxarray.grid.geometry import _pole_point_inside_polygon_cartesian

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parent

gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"


def test_antimeridian_crossing():
    verts = [[[-170, 40], [180, 30], [165, 25], [-170, 20]]]

    uxgrid = ux.open_grid(verts, latlon=True)

    gdf = uxgrid.to_geodataframe(periodic_elements='ignore')

    assert len(uxgrid.antimeridian_face_indices) == 1
    assert len(gdf['geometry']) == 1


def test_antimeridian_point_on():
    verts = [[[-170, 40], [180, 30], [-170, 20]]]

    uxgrid = ux.open_grid(verts, latlon=True)

    assert len(uxgrid.antimeridian_face_indices) == 1


def test_linecollection_execution():
    uxgrid = ux.open_grid(gridfile_CSne8)
    lines = uxgrid.to_linecollection()


def test_face_at_antimeridian():
    """Test for a face that crosses the antimeridian."""
    # Create a face that crosses the antimeridian
    verts = [[[170, 10], [-170, 10], [-170, -10], [170, -10]]]
    uxgrid = ux.open_grid(verts, latlon=True)

    # Test that antimeridian faces are detected
    assert len(uxgrid.antimeridian_face_indices) > 0
