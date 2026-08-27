import numpy as np
import numpy.testing as nt
import pytest
import xarray as xr

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.validation import (
    _check_duplicate_nodes_indices,
    _find_duplicate_nodes,
)
from uxarray.errors import GridInvalidError


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


def test_dual_mesh_mpas(gridpath):
    """Test dual mesh creation for MPAS grids."""
    grid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=False)
    mpas_dual = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=True)

    dual = grid.get_dual()

    assert dual.n_face == mpas_dual.n_face
    assert dual.n_node == mpas_dual.n_node
    assert dual.n_max_face_nodes == mpas_dual.n_max_face_nodes

    nt.assert_equal(dual.face_node_connectivity.values, mpas_dual.face_node_connectivity.values)


def test_dual_duplicate(gridpath):
    """Test dual mesh creation on a grid whose source file has duplicate
    (coincident) node indices, merged at construction time."""
    grid_path = gridpath("ugrid", "geoflow-small", "grid.nc")
    grid = ux.open_grid(grid_path)

    # The source file really does contain duplicates: 6000 node coordinates for
    # 3850 distinct locations, so 2150 indices are coincident with an earlier one.
    duplicates = _find_duplicate_nodes(grid)
    assert grid.n_node == 6000
    assert len(duplicates) == 2150

    # Connectivity is canonicalized to a single index per coincident group, so no
    # face references any of those 2150 duplicate indices.
    assert not _check_duplicate_nodes_indices(grid)
    # duplicate coordinates are left in place by design, but connectivity is
    # fully canonicalized, so validation passes
    assert grid.validate()

    dual = grid.get_dual()

    assert dual.n_node == grid.n_face

    # One dual face per node that is a corner of at least three faces. After the
    # merge, 3850 distinct nodes remain, ten of which are touched by a single face
    # only and so produce no dual cell, leaving 3840.
    face_nodes = grid.face_node_connectivity.values
    faces_per_node = np.bincount(
        face_nodes[face_nodes != INT_FILL_VALUE], minlength=grid.n_node
    )
    assert grid.n_node - len(duplicates) == 3850
    assert (faces_per_node >= 3).sum() == 3840
    assert dual.n_face == 3840

    dataset = ux.open_dataset(grid_path, grid_path)
    dual_ds = dataset.get_dual()
    assert dual_ds.uxgrid.n_face == dual.n_face


def test_dual_duplicate_geos_cs(gridpath):
    """Test dual mesh creation on a cube-sphere grid with duplicate node
    indices (issue #865)."""
    grid_path = gridpath("geos-cs", "c12", "test-c12.native.nc4")
    grid = ux.open_grid(grid_path)

    assert len(_find_duplicate_nodes(grid)) > 0
    assert not _check_duplicate_nodes_indices(grid)

    dual = grid.get_dual()
    assert dual.n_node == grid.n_face
    assert dual.n_face > 0


def test_duplicate_nodes_minimal_example():
    """Two quads that share an edge, but whose shared corners are stored twice.

    Nodes 2 and 3 are repeated as nodes 6 and 7, so the file describes 8 nodes at
    6 distinct locations. Node 6 must canonicalize to node 2 and node 7 to node 3,
    leaving the second face pointing at the first face's corners.

        3---2---7      lat 1   nodes 2,3 are the shared edge
        |   |   |              nodes 7,6 are their duplicates
        0---1---6      lat 0
    """
    node_lon = np.array([0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 1.0])
    node_lat = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    #                    left quad        right quad, via the duplicates
    face_node_connectivity = np.array([[0, 1, 2, 3], [6, 4, 5, 7]])

    grid = ux.Grid.from_topology(node_lon, node_lat, face_node_connectivity)

    duplicates = _find_duplicate_nodes(grid)
    assert duplicates == {6: 1, 7: 2}

    # No face may still reference a duplicate index.
    assert not _check_duplicate_nodes_indices(grid)
    nt.assert_equal(
        grid.face_node_connectivity.values, np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    )


def test_get_dual_rejects_faces_referencing_duplicate_nodes():
    """``construct_dual`` reads ``node_face_connectivity`` with no duplicate
    handling, so a face still pointing at a dead duplicate index would yield a
    degenerate dual face instead of an error. Merging at construction makes this
    unreachable today; the guard keeps it that way."""
    node_lon = np.array([0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 1.0])
    node_lat = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    unmerged = np.array([[0, 1, 2, 3], [6, 4, 5, 7]])

    grid = ux.Grid.from_topology(node_lon, node_lat, unmerged)
    # Construction canonicalized the connectivity; put the duplicates back.
    grid.face_node_connectivity = xr.DataArray(
        unmerged, dims=grid.face_node_connectivity.dims
    )

    assert _check_duplicate_nodes_indices(grid)
    with pytest.raises(GridInvalidError):
        grid.get_dual()


def test_pole_exception_uses_a_chord_tolerance():
    """The pole carve-out must be a chord radius, not a raw ``|z|`` deviation.

    ``np.isclose(|z|, 1.0, atol=tolerance)`` also carries numpy's default
    ``rtol=1e-5``, so the carve-out spanned ``1 - |z| <= 1.001e-5`` -- a chord of
    4.5e-3, or ~28 km on Earth. Every node within that cap was exempted from
    merging. Only nodes at the pole itself may be exempt.
    """
    from uxarray.grid.validation import _coincident_node_canonical_indices

    # Colatitude chosen so 1 - z = 1e-6: well inside the old carve-out, and far
    # outside a chord of ERROR_TOLERANCE (whose cap is 1 - z <= 5e-17).
    z = 1.0 - 1e-6
    x = np.sqrt(1.0 - z * z)

    points_xyz = np.array(
        [
            [0.0, 0.0, 1.0],  # north pole, kept distinct from the next node
            [0.0, 0.0, 1.0],  # same location, its own face-specific longitude
            [x, 0.0, z],  # near the pole, genuinely coincident with the next
            [x, 0.0, z],
        ]
    )

    canonical = _coincident_node_canonical_indices(points_xyz)

    # Nodes at a pole are still never merged with one another.
    nt.assert_equal(canonical[:2], np.array([0, 1]))
    # Near-pole coincident nodes now merge; before the fix they were exempt.
    nt.assert_equal(canonical[2:], np.array([2, 2]))


def test_no_duplicate_nodes_ne30pg3(gridpath):
    """``esmf/ne30/ne30pg3.grid.nc`` no longer reproduces issue #865's
    duplicate-node bug; this only checks the general fix is a safe no-op."""
    grid_path = gridpath("esmf", "ne30", "ne30pg3.grid.nc")
    grid = ux.open_grid(grid_path)

    assert len(_find_duplicate_nodes(grid)) == 0
