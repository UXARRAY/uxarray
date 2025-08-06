"""
Shared pytest fixtures for uxarray tests.
"""

import pytest
from pathlib import Path
import os
import tempfile


@pytest.fixture(scope="session")
def test_data_paths():
    """Centralized test data paths for all formats."""
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    base_path = current_path / "meshfiles"
    return {
        "ugrid": {
            "ne30": base_path / "ugrid" / "outCSne30" / "outCSne30.ug",
            "rll1deg": base_path / "ugrid" / "outRLL1deg" / "outRLL1deg.ug",
            "rll10deg_ne4": base_path / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug",
            "geoflow": base_path / "ugrid" / "geoflow-small" / "grid.nc"
        },
        "exodus": {
            "ne8": base_path / "exodus" / "outCSne8" / "outCSne8.g",
            "mixed": base_path / "exodus" / "mixed" / "mixed.exo"
        },
        "esmf": {
            "ne30_grid": base_path / "esmf" / "ne30" / "ne30pg3.grid.nc",
            "ne30_data": base_path / "esmf" / "ne30" / "ne30pg3.data.nc"
        },
        "scrip": {
            "ne8": base_path / "scrip" / "outCSne8" / "outCSne8.nc"
        },
        "mpas": {
            "qu_grid": base_path / "mpas" / "QU" / "mesh.QU.1920km.151026.nc",
            "qu_ocean": base_path / "mpas" / "QU" / "oQU480.231010.nc"
        },
        "icon": {
            "r02b04": base_path / "icon" / "R02B04" / "icon_grid_0010_R02B04_G.nc"
        },
        "fesom": {
            "ugrid_diag": base_path / "ugrid" / "fesom" / "fesom.mesh.diag.nc",
            "ascii": base_path / "fesom" / "pi",
            "netcdf": base_path / "fesom" / "soufflet-netcdf" / "grid.nc"
        },
        "shp": {
            "us_nation": base_path / "shp" / "cb_2018_us_nation_20m" / "cb_2018_us_nation_20m.shp",
            "5poly": base_path / "shp" / "5poly" / "5poly.shp",
            "multipoly": base_path / "shp" / "multipoly" / "multipoly.shp"
        },
        "geojson": {
            "chicago_buildings": base_path / "geojson" / "sample_chicago_buildings.geojson"
        },
        "healpix": {
            "outCSne30": base_path / "healpix" / "outCSne30" / "data.nc"
        }
    }


@pytest.fixture
def temp_output_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
