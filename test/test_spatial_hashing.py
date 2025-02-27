import os
import numpy as np
import pytest
import xarray as xr
from pathlib import Path
import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

# Sample grid file paths
gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
gridfile_RLL1deg = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
gridfile_RLL10deg_CSne4 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"
gridfile_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
gridfile_fesom = current_path / "meshfiles" / "ugrid" / "fesom" / "fesom.mesh.diag.nc"
gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
gridfile_mpas = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'

grid_files = [gridfile_CSne8,
              gridfile_RLL1deg,
              gridfile_RLL10deg_CSne4,
              gridfile_CSne30,
              gridfile_fesom,
              gridfile_geoflow,
              gridfile_mpas]

def test_construction():
    """Tests the construction of the SpatialHash object"""
    for grid_file in grid_files:
        uxgrid = ux.open_grid(grid_file)
        face_ids, bcoords = uxgrid.get_spatialhash().query([0.9, 1.8])
        assert face_ids.shape[0] == bcoords.shape[0]


def test_is_inside():
    """Verifies element search across Antimeridian."""
    verts = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]
    uxgrid = ux.open_grid(verts, latlon=True)
    # Verify that a point outside the element returns a face id of -1
    face_ids, bcoords = uxgrid.get_spatialhash().query([90.0, 0.0])
    assert face_ids[0] == -1
    # Verify that a point inside the element returns a face id of 0
    face_ids, bcoords = uxgrid.get_spatialhash().query([-90.0, 0.0])
    print(face_ids[0], bcoords[0])

    assert face_ids[0] == 0
    assert np.allclose(bcoords[0], [0.25, 0.5, 0.25], atol=1e-06)
