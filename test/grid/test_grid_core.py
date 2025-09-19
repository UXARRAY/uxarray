import numpy as np
import numpy.testing as nt
import pytest
import xarray as xr

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE


def test_grid_validate(gridpath):
    """Test to check the validate function."""
    grid_mpas = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    assert grid_mpas.validate()


def test_grid_with_holes(gridpath):
    """Test _holes_in_mesh function."""
    grid_without_holes = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    grid_with_holes = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))

    assert grid_with_holes.partial_sphere_coverage
    assert grid_without_holes.global_sphere_coverage


def test_grid_init_verts():
    """Create a uxarray grid from multiple face vertices with duplicate nodes and saves a ugrid file."""
    cart_x = [
        0.577340924821405, 0.577340924821405, 0.577340924821405,
        0.577340924821405, -0.577345166204668, -0.577345166204668,
        -0.577345166204668, -0.577345166204668
    ]
    cart_y = [
        0.577343045516932, 0.577343045516932, -0.577343045516932,
        -0.577343045516932, 0.577338804118089, 0.577338804118089,
        -0.577338804118089, -0.577338804118089
    ]
    cart_z = [
        0.577366836872017, -0.577366836872017, 0.577366836872017,
        -0.577366836872017, 0.577366836872017, -0.577366836872017,
        0.577366836872017, -0.577366836872017
    ]

    face_vertices = [
        [0, 1, 2, 3],  # front face
        [1, 5, 6, 2],  # right face
        [5, 4, 7, 6],  # back face
        [4, 0, 3, 7],  # left face
        [3, 2, 6, 7],  # top face
        [4, 5, 1, 0]  # bottom face
    ]

    faces_coords = []
    for face in face_vertices:
        face_coords = []
        for vertex_index in face:
            face_coords.append([cart_x[vertex_index], cart_y[vertex_index], cart_z[vertex_index]])
        faces_coords.append(face_coords)

    grid_verts = ux.open_grid(faces_coords, latlon=False)

    # validate the grid
    assert grid_verts.validate()


def test_grid_properties(gridpath):
    """Test to check the grid properties."""
    grid_mpas = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    # Test n_face
    assert grid_mpas.n_face == 2562

    # Test n_node
    assert grid_mpas.n_node == 1281

    # Test n_edge
    assert grid_mpas.n_edge == 3840


def test_read_scrip(gridpath):
    """Test to check the read_scrip function."""
    grid_scrip = ux.open_grid(gridpath("scrip", "outCSne8", "outCSne8.nc"))
    assert grid_scrip.validate()


def test_operators_eq(gridpath):
    """Test to check the == operator."""
    grid_mpas_1 = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    grid_mpas_2 = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    assert grid_mpas_1 == grid_mpas_2


def test_operators_ne(gridpath):
    """Test to check the != operator."""
    grid_mpas = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    grid_scrip = ux.open_grid(gridpath("scrip", "outCSne8", "outCSne8.nc"))

    assert grid_mpas != grid_scrip


def test_grid_properties(gridpath):
    """Tests to see if accessing variables through set properties is equal to using the dict."""
    grid_CSne30 = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    xr.testing.assert_equal(grid_CSne30.node_lon, grid_CSne30._ds["node_lon"])
    xr.testing.assert_equal(grid_CSne30.node_lat, grid_CSne30._ds["node_lat"])
    xr.testing.assert_equal(grid_CSne30.face_node_connectivity, grid_CSne30._ds["face_node_connectivity"])

    n_nodes = grid_CSne30.node_lon.shape[0]
    n_faces, n_face_nodes = grid_CSne30.face_node_connectivity.shape

    assert n_nodes == grid_CSne30.n_node
    assert n_faces == grid_CSne30.n_face
    assert n_face_nodes == grid_CSne30.n_max_face_nodes

    grid_geoflow = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))


def test_class_methods_from_dataset(gridpath):
    # UGRID
    xrds = xr.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"))
    uxgrid = ux.Grid.from_dataset(xrds)

    # MPAS
    xrds = xr.open_dataset(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    uxgrid = ux.Grid.from_dataset(xrds, use_dual=False)
    uxgrid = ux.Grid.from_dataset(xrds, use_dual=True)

    # Exodus
    xrds = xr.open_dataset(gridpath("exodus", "outCSne8", "outCSne8.g"))
    uxgrid = ux.Grid.from_dataset(xrds)

    # SCRIP
    xrds = xr.open_dataset(gridpath("scrip", "outCSne8", "outCSne8.nc"))
    uxgrid = ux.Grid.from_dataset(xrds)