import os
import numpy as np
import pytest
import xarray as xr
from pathlib import Path
import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE

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
        face_ids, bcoords = uxgrid.get_spatial_hash().query([0.9, 1.8])
        assert face_ids.shape[0] == bcoords.shape[0]


def test_is_inside():
    """Verifies simple test for points inside and outside an element."""
    verts = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]
    uxgrid = ux.open_grid(verts, latlon=True)
    # Verify that a point outside the element returns a face id of -1
    face_ids, bcoords = uxgrid.get_spatial_hash().query([90.0, 0.0])
    assert face_ids[0] == -1
    # Verify that a point inside the element returns a face id of 0
    face_ids, bcoords = uxgrid.get_spatial_hash().query([-90.0, 0.0])

    assert face_ids[0] == 0
    assert np.allclose(bcoords[0], [0.25, 0.5, 0.25], atol=1e-06)


def test_query_on_vertex():
    """Verifies correct values when a query is made exactly on a vertex"""
    verts = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]
    uxgrid = ux.open_grid(verts, latlon=True)
    # Verify that a point outside the element returns a face id of -1
    face_ids, bcoords = uxgrid.get_spatial_hash().query([0.0, 90.0])
    assert face_ids[0] == 0
    assert np.isclose(bcoords[0,0],1.0,atol=ERROR_TOLERANCE)
    assert np.isclose(bcoords[0,1],0.0,atol=ERROR_TOLERANCE)
    assert np.isclose(bcoords[0,2],0.0,atol=ERROR_TOLERANCE)


def test_query_on_edge():
    """Verifies correct values when a query is made exactly on an edge of a face"""
    verts = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]
    uxgrid = ux.open_grid(verts, latlon=True)
    # Verify that a point outside the element returns a face id of -1
    face_ids, bcoords = uxgrid.get_spatial_hash().query([0.0, 0.0])
    assert face_ids[0] == 0
    assert np.isclose(bcoords[0,0],0.5,atol=ERROR_TOLERANCE)
    assert np.isclose(bcoords[0,1],0.0,atol=ERROR_TOLERANCE)
    assert np.isclose(bcoords[0,2],0.5,atol=ERROR_TOLERANCE)


def test_list_of_coords_simple():
    """Verifies test using list of points inside and outside an element"""
    verts = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]
    uxgrid = ux.open_grid(verts, latlon=True)

    coords = [[90.0, 0.0], [-90.0, 0.0]]
    face_ids, bcoords = uxgrid.get_spatial_hash().query(coords)
    assert face_ids[0] == -1
    assert face_ids[1] == 0
    assert np.allclose(bcoords[1], [0.25, 0.5, 0.25], atol=1e-06)


def test_list_of_coords_fesom():
    """Verifies test using list of points on the fesom grid"""
    uxgrid = ux.open_grid(gridfile_fesom)

    num_particles = 20
    coords = np.zeros((num_particles,2))
    x_min = 1.0
    x_max = 3.0
    y_min = 2.0
    y_max = 10.0
    for k in range(num_particles):
        coords[k,0] = np.deg2rad(np.random.uniform(x_min, x_max))
        coords[k,1] = np.deg2rad(np.random.uniform(y_min, y_max))
    face_ids, bcoords = uxgrid.get_spatial_hash().query(coords)
    assert len(face_ids) == num_particles
    assert bcoords.shape[0] == num_particles
    assert bcoords.shape[1] == 3
    assert np.all(face_ids >= 0) # All particles should be inside an element


def test_list_of_coords_mpas_dual():
    """Verifies test using list of points on the dual MPAS grid"""
    uxgrid = ux.open_grid(gridfile_mpas, use_dual=True)

    num_particles = 20
    coords = np.zeros((num_particles,2))
    x_min = -40.0
    x_max = 40.0
    y_min = -20.0
    y_max = 20.0
    for k in range(num_particles):
        coords[k,0] = np.deg2rad(np.random.uniform(x_min, x_max))
        coords[k,1] = np.deg2rad(np.random.uniform(y_min, y_max))
    face_ids, bcoords = uxgrid.get_spatial_hash().query(coords)
    assert len(face_ids) == num_particles
    assert bcoords.shape[0] == num_particles
    assert bcoords.shape[1] == 3 # max sides of an element
    assert np.all(face_ids >= 0) # All particles should be inside an element


def test_list_of_coords_mpas_primal():
    """Verifies test using list of points on the primal MPAS grid"""
    uxgrid = ux.open_grid(gridfile_mpas, use_dual=False)

    num_particles = 20
    coords = np.zeros((num_particles,2))
    x_min = -40.0
    x_max = 40.0
    y_min = -20.0
    y_max = 20.0
    for k in range(num_particles):
        coords[k,0] = np.deg2rad(np.random.uniform(x_min, x_max))
        coords[k,1] = np.deg2rad(np.random.uniform(y_min, y_max))
    face_ids, bcoords = uxgrid.get_spatial_hash().query(coords)
    assert len(face_ids) == num_particles
    assert bcoords.shape[0] == num_particles
    assert bcoords.shape[1] == 6 # max sides of an element
    assert np.all(face_ids >= 0) # All particles should be inside an element
