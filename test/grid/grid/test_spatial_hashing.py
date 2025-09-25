import numpy as np
import pytest
import xarray as xr
import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE




def test_construction(gridpath):
    """Tests the construction of the SpatialHash object"""
    grid_files = [
        gridpath("scrip", "outCSne8", "outCSne8.nc"),
        gridpath("ugrid", "outRLL1deg", "outRLL1deg.ug"),
        gridpath("ugrid", "ov_RLL10deg_CSne4", "ov_RLL10deg_CSne4.ug"),
        gridpath("ugrid", "fesom", "fesom.mesh.diag.nc"),
        gridpath("ugrid", "geoflow-small", "grid.nc")
    ]
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

def test_list_of_coords_fesom(gridpath):
    """Verifies test using list of points on the fesom grid"""
    uxgrid = ux.open_grid(gridpath("ugrid", "fesom", "fesom.mesh.diag.nc"))

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

def test_list_of_coords_mpas_dual(gridpath):
    """Verifies test using list of points on the dual MPAS grid"""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=True)

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

def test_list_of_coords_mpas_primal(gridpath):
    """Verifies test using list of points on the primal MPAS grid"""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=False)

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
