import os
import numpy as np
import numpy.testing as nt
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray
import uxarray as ux
import pytest

from uxarray.grid.connectivity import _populate_face_edge_connectivity, _build_edge_face_connectivity, \
    _build_edge_node_connectivity, _build_face_face_connectivity, _populate_face_face_connectivity

from uxarray.grid.coordinates import _populate_node_latlon, _lonlat_rad_to_xyz, _xyz_to_lonlat_rad_scalar

from uxarray.constants import INT_FILL_VALUE, ERROR_TOLERANCE

from uxarray.grid.arcs import extreme_gca_latitude

from uxarray.grid.validation import _find_duplicate_nodes
from .test_gradient import quad_hex_grid_path

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
gridfile_RLL1deg = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
gridfile_RLL10deg_CSne4 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"
gridfile_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
gridfile_fesom = current_path / "meshfiles" / "ugrid" / "fesom" / "fesom.mesh.diag.nc"
gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
gridfile_geos = current_path / 'meshfiles' / "geos-cs" / "c12" / 'test-c12.native.nc4'
gridfile_mpas_holes = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'

dsfile_vortex_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"
dsfile_var2_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"

shp_filename = current_path / "meshfiles" / "shp" / "grid_fire.shp"

grid_CSne30 = ux.open_grid(gridfile_CSne30)
grid_RLL1deg = ux.open_grid(gridfile_RLL1deg)
grid_RLL10deg_CSne4 = ux.open_grid(gridfile_RLL10deg_CSne4)

mpas_filepath = current_path / "meshfiles" / "mpas" / "QU" / "mesh.QU.1920km.151026.nc"
exodus_filepath = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
ugrid_filepath_01 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
ugrid_filepath_02 = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
ugrid_filepath_03 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"

grid_mpas = ux.open_grid(mpas_filepath)
grid_exodus = ux.open_grid(exodus_filepath)
grid_ugrid = ux.open_grid(ugrid_filepath_01)

f0_deg = [[120, -20], [130, -10], [120, 0], [105, 0], [95, -10], [105, -20]]
f1_deg = [[120, 0], [120, 10], [115, 0],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]
f2_deg = [[115, 0], [120, 10], [100, 10], [105, 0],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]
f3_deg = [[95, -10], [105, 0], [95, 30], [80, 30], [70, 0], [75, -10]]
f4_deg = [[65, -20], [75, -10], [70, 0], [55, 0], [45, -10], [55, -20]]
f5_deg = [[70, 0], [80, 30], [70, 30], [60, 0],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]
f6_deg = [[60, 0], [70, 30], [40, 30], [45, 0],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]

gridfile_ugrid = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
gridfile_mpas = current_path / "meshfiles" / "mpas" / "QU" / "mesh.QU.1920km.151026.nc"
gridfile_mpas_two = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'
gridfile_exodus = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
gridfile_scrip = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"


def test_grid_validate():
    """Test to check the validate function."""
    grid_mpas = ux.open_grid(gridfile_mpas)
    assert grid_mpas.validate()


def test_grid_with_holes():
    """Test _holes_in_mesh function."""
    grid_without_holes = ux.open_grid(gridfile_mpas)
    grid_with_holes = ux.open_grid(gridfile_mpas_holes)

    assert grid_with_holes.partial_sphere_coverage
    assert grid_without_holes.global_sphere_coverage


def test_grid_encode_as():
    """Reads a ugrid file and encodes it as `xarray.Dataset` in various types."""
    grid_CSne30.encode_as("UGRID")
    grid_RLL1deg.encode_as("UGRID")
    grid_RLL10deg_CSne4.encode_as("UGRID")

    grid_CSne30.encode_as("Exodus")
    grid_RLL1deg.encode_as("Exodus")
    grid_RLL10deg_CSne4.encode_as("Exodus")


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
            x, y, z = cart_x[vertex_index], cart_y[vertex_index], cart_z[vertex_index]
            face_coords.append([x, y, z])
        faces_coords.append(face_coords)

    verts_cart = np.array(faces_coords)
    vgrid = ux.open_grid(verts_cart, latlon=False)

    assert vgrid.n_face == 6
    assert vgrid.n_node == 8
    vgrid.encode_as("UGRID")

    faces_verts_one = np.array([
        np.array([[150, 10], [160, 20], [150, 30], [135, 30], [125, 20], [135, 10]])
    ])
    vgrid = ux.open_grid(faces_verts_one, latlon=True)
    assert vgrid.n_face == 1
    assert vgrid.n_node == 6
    vgrid.encode_as("UGRID")

    faces_verts_single_face = np.array([[150, 10], [160, 20], [150, 30], [135, 30], [125, 20], [135, 10]])
    vgrid = ux.open_grid(faces_verts_single_face, latlon=True)
    assert vgrid.n_face == 1
    assert vgrid.n_node == 6
    vgrid.encode_as("UGRID")


def test_grid_init_verts_different_input_datatype():
    """Create a uxarray grid from multiple face vertices with different datatypes (ndarray, list, tuple) and saves a ugrid file."""
    faces_verts_ndarray = np.array([
        np.array([[150, 10], [160, 20], [150, 30], [135, 30], [125, 20], [135, 10]]),
        np.array([[125, 20], [135, 30], [125, 60], [110, 60], [100, 30], [105, 20]]),
        np.array([[95, 10], [105, 20], [100, 30], [85, 30], [75, 20], [85, 10]]),
    ])
    vgrid = ux.open_grid(faces_verts_ndarray, latlon=True)
    assert vgrid.n_face == 3
    assert vgrid.n_node == 14
    vgrid.encode_as("UGRID")

    faces_verts_list = [[[150, 10], [160, 20], [150, 30], [135, 30], [125, 20], [135, 10]],
                        [[125, 20], [135, 30], [125, 60], [110, 60], [100, 30], [105, 20]],
                        [[95, 10], [105, 20], [100, 30], [85, 30], [75, 20], [85, 10]]]
    vgrid = ux.open_grid(faces_verts_list, latlon=True)
    assert vgrid.n_face == 3
    assert vgrid.n_node == 14
    assert vgrid.validate()
    vgrid.encode_as("UGRID")

    faces_verts_tuples = [
        ((150, 10), (160, 20), (150, 30), (135, 30), (125, 20), (135, 10)),
        ((125, 20), (135, 30), (125, 60), (110, 60), (100, 30), (105, 20)),
        ((95, 10), (105, 20), (100, 30), (85, 30), (75, 20), (85, 10))
    ]
    vgrid = ux.open_grid(faces_verts_tuples, latlon=True)
    assert vgrid.n_face == 3
    assert vgrid.n_node == 14
    assert vgrid.validate()
    vgrid.encode_as("UGRID")


def test_grid_init_verts_fill_values():
    faces_verts_filled_values = [[[150, 10], [160, 20], [150, 30],
                                  [135, 30], [125, 20], [135, 10]],
                                 [[125, 20], [135, 30], [125, 60],
                                  [110, 60], [100, 30],
                                  [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]],
                                 [[95, 10], [105, 20], [100, 30], [85, 30],
                                  [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
                                  [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]]
    vgrid = ux.open_grid(faces_verts_filled_values, latlon=False)
    assert vgrid.n_face == 3
    assert vgrid.n_node == 12


def test_grid_properties():
    """Tests to see if accessing variables through set properties is equal to using the dict."""
    xr.testing.assert_equal(grid_CSne30.node_lon, grid_CSne30._ds["node_lon"])
    xr.testing.assert_equal(grid_CSne30.node_lat, grid_CSne30._ds["node_lat"])
    xr.testing.assert_equal(grid_CSne30.face_node_connectivity, grid_CSne30._ds["face_node_connectivity"])

    n_nodes = grid_CSne30.node_lon.shape[0]
    n_faces, n_face_nodes = grid_CSne30.face_node_connectivity.shape

    assert n_nodes == grid_CSne30.n_node
    assert n_faces == grid_CSne30.n_face
    assert n_face_nodes == grid_CSne30.n_max_face_nodes

    grid_geoflow = ux.open_grid(gridfile_geoflow)

    xr.testing.assert_equal(grid_geoflow.node_lon, grid_geoflow._ds["node_lon"])
    xr.testing.assert_equal(grid_geoflow.node_lat, grid_geoflow._ds["node_lat"])
    xr.testing.assert_equal(grid_geoflow.face_node_connectivity, grid_geoflow._ds["face_node_connectivity"])

    n_nodes = grid_geoflow.node_lon.shape[0]
    n_faces, n_face_nodes = grid_geoflow.face_node_connectivity.shape

    assert n_nodes == grid_geoflow.n_node
    assert n_faces == grid_geoflow.n_face
    assert n_face_nodes == grid_geoflow.n_max_face_nodes


def test_read_shpfile():
    """Reads a shape file and write ugrid file."""
    with pytest.raises(ValueError):
        grid_shp = ux.open_grid(shp_filename)


def test_read_scrip():
    """Reads a scrip file."""
    grid_CSne8 = ux.open_grid(gridfile_CSne8)  # tests from scrip


def test_operators_eq():
    """Test Equals ('==') operator."""
    grid_CSne30_01 = ux.open_grid(gridfile_CSne30)
    grid_CSne30_02 = ux.open_grid(gridfile_CSne30)
    assert grid_CSne30_01 == grid_CSne30_02


def test_operators_ne():
    """Test Not Equals ('!=') operator."""
    grid_CSne30_01 = ux.open_grid(gridfile_CSne30)
    grid_RLL1deg = ux.open_grid(gridfile_RLL1deg)
    assert grid_CSne30_01 != grid_RLL1deg


def test_face_areas_calculate_total_face_area_triangle():
    """Create a uxarray grid from vertices and saves an exodus file."""
    verts = [
    [[0.02974582, -0.74469018, 0.66674712],
    [0.1534193, -0.88744577, 0.43462917],
    [0.18363692, -0.72230586, 0.66674712]]
    ]

    grid_verts = ux.open_grid(verts, latlon=False)

    # validate the grid
    assert grid_verts.validate()

    # calculate area without correction
    area_triangular = grid_verts.calculate_total_face_area(
        quadrature_rule="triangular", order=4)
    nt.assert_almost_equal(area_triangular, constants.TRI_AREA, decimal=1)

    # calculate area
    area_gaussian = grid_verts.calculate_total_face_area(
        quadrature_rule="gaussian", order=5, latitude_adjusted_area=True)
    nt.assert_almost_equal(area_gaussian, constants.CORRECTED_TRI_AREA, decimal=3)


def test_face_areas_calculate_total_face_area_file():
    """Create a uxarray grid from vertices and saves an exodus file."""
    area = ux.open_grid(gridfile_CSne30).calculate_total_face_area()
    nt.assert_almost_equal(area, constants.MESH30_AREA, decimal=3)


def test_face_areas_calculate_total_face_area_sphere():
    """Computes the total face area of an MPAS mesh that lies on a unit sphere, with an expected total face area of 4pi."""
    mpas_grid_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'

    primal_grid = ux.open_grid(mpas_grid_path, use_dual=False)
    dual_grid = ux.open_grid(mpas_grid_path, use_dual=True)

    primal_face_area = primal_grid.calculate_total_face_area()
    dual_face_area = dual_grid.calculate_total_face_area()

    nt.assert_almost_equal(primal_face_area, constants.UNIT_SPHERE_AREA, decimal=3)
    nt.assert_almost_equal(dual_face_area, constants.UNIT_SPHERE_AREA, decimal=3)


def test_face_areas_compute_face_areas_geoflow_small():
    """Checks if the GeoFlow Small can generate a face areas output."""
    grid_geoflow = ux.open_grid(gridfile_geoflow)
    grid_geoflow.compute_face_areas()


def test_face_areas_verts_calc_area():
    faces_verts_ndarray = np.array([
        np.array([[150, 10, 0], [160, 20, 0], [150, 30, 0], [135, 30, 0],
                  [125, 20, 0], [135, 10, 0]]),
        np.array([[125, 20, 0], [135, 30, 0], [125, 60, 0], [110, 60, 0],
                  [100, 30, 0], [105, 20, 0]]),
        np.array([[95, 10, 0], [105, 20, 0], [100, 30, 0], [85, 30, 0],
                  [75, 20, 0], [85, 10, 0]]),
    ])
    verts_grid = ux.open_grid(faces_verts_ndarray, latlon=True)
    face_verts_areas = verts_grid.face_areas
    nt.assert_almost_equal(face_verts_areas.sum(), constants.FACE_VERTS_AREA, decimal=3)


def test_populate_coordinates_populate_cartesian_xyz_coord():
    # The following testcases are generated through the matlab cart2sph/sph2cart functions
    lon_deg = [
        45.0001052295749, 45.0001052295749, 360 - 45.0001052295749,
                                            360 - 45.0001052295749
    ]
    lat_deg = [
        35.2655522903022, -35.2655522903022, 35.2655522903022,
        -35.2655522903022
    ]
    cart_x = [
        0.577340924821405, 0.577340924821405, 0.577340924821405,
        0.577340924821405
    ]
    cart_y = [
        0.577343045516932, 0.577343045516932, -0.577343045516932,
        -0.577343045516932
    ]
    cart_z = [
        -0.577366836872017, 0.577366836872017, -0.577366836872017,
        0.577366836872017
    ]

    verts_degree = np.stack((lon_deg, lat_deg), axis=1)
    vgrid = ux.open_grid(verts_degree, latlon=True)

    for i in range(0, vgrid.n_node):
        nt.assert_almost_equal(vgrid.node_x.values[i], cart_x[i], decimal=12)
        nt.assert_almost_equal(vgrid.node_y.values[i], cart_y[i], decimal=12)
        nt.assert_almost_equal(vgrid.node_z.values[i], cart_z[i], decimal=12)


def test_populate_coordinates_populate_lonlat_coord():
    lon_deg = [
        45.0001052295749, 45.0001052295749, 360 - 45.0001052295749,
                                            360 - 45.0001052295749
    ]
    lat_deg = [
        35.2655522903022, -35.2655522903022, 35.2655522903022,
        -35.2655522903022
    ]
    cart_x = [
        0.577340924821405, 0.577340924821405, 0.577340924821405,
        0.577340924821405
    ]
    cart_y = [
        0.577343045516932, 0.577343045516932, -0.577343045516932,
        -0.577343045516932
    ]
    cart_z = [
        0.577366836872017, -0.577366836872017, 0.577366836872017,
        -0.577366836872017
    ]

    verts_cart = np.stack((cart_x, cart_y, cart_z), axis=1)
    vgrid = ux.open_grid(verts_cart, latlon=False)
    _populate_node_latlon(vgrid)
    lon_deg, lat_deg = zip(*reversed(list(zip(lon_deg, lat_deg))))
    for i in range(0, vgrid.n_node):
        nt.assert_almost_equal(vgrid._ds["node_lon"].values[i], lon_deg[i], decimal=12)
        nt.assert_almost_equal(vgrid._ds["node_lat"].values[i], lat_deg[i], decimal=12)


def _revert_edges_conn_to_face_nodes_conn(edge_nodes_connectivity: np.ndarray,
                                          face_edges_connectivity: np.ndarray,
                                          original_face_nodes_connectivity: np.ndarray):
    """Utilize the edge_nodes_connectivity and face_edges_connectivity to
    generate the res_face_nodes_connectivity in the counter-clockwise
    order. The counter-clockwise order will be enforced by the passed in
    original_face_edges_connectivity. We will only use the first two nodes
    in the original_face_edges_connectivity. The order of these two nodes
    will provide a correct counter-clockwise order to build our
    res_face_nodes_connectivity. A ValueError will be raised if the first
    two nodes in the res_face_nodes_connectivity and the
    original_face_nodes_connectivity are not the same elements (The order
    doesn't matter here).
    """
    # Create a dictionary to store the face indices for each edge
    face_nodes_dict = {}

    # Loop through each face and edge to build the dictionary
    for face_idx, face_edges in enumerate(face_edges_connectivity):
        for edge_idx in face_edges:
            if edge_idx != ux.INT_FILL_VALUE:
                edge = edge_nodes_connectivity[edge_idx]
                if face_idx not in face_nodes_dict:
                    face_nodes_dict[face_idx] = []
                face_nodes_dict[face_idx].append(edge[0])
                face_nodes_dict[face_idx].append(edge[1])

    # Make sure the face_nodes_dict is in the counter-clockwise order and remove duplicate nodes
    for face_idx, face_nodes in face_nodes_dict.items():
        first_edge_correct = np.array([
            original_face_nodes_connectivity[face_idx][0],
            original_face_nodes_connectivity[face_idx][1]
        ])
        first_edge = np.array([face_nodes[0], face_nodes[1]])

        first_edge_correct_copy = first_edge_correct.copy()
        first_edge_copy = first_edge.copy()
        assert np.array_equal(np.sort(first_edge_correct_copy), np.sort(first_edge_copy))
        face_nodes[0] = first_edge_correct[0]
        face_nodes[1] = first_edge_correct[1]

        i = 2
        while i < len(face_nodes):
            if face_nodes[i] != face_nodes[i - 1]:
                old = face_nodes[i]
                face_nodes[i] = face_nodes[i - 1]
                face_nodes[i + 1] = old
            i += 2

        after_swapped = face_nodes
        after_swapped_remove = [after_swapped[0]]

        for i in range(1, len(after_swapped) - 1):
            if after_swapped[i] != after_swapped[i - 1]:
                after_swapped_remove.append(after_swapped[i])

        face_nodes_dict[face_idx] = after_swapped_remove

    # Convert the dictionary to a list
    res_face_nodes_connectivity = []
    for face_idx in range(len(face_edges_connectivity)):
        res_face_nodes_connectivity.append(face_nodes_dict[face_idx])
        while len(res_face_nodes_connectivity[face_idx]) < original_face_nodes_connectivity.shape[1]:
            res_face_nodes_connectivity[face_idx].append(ux.INT_FILL_VALUE)

    return np.array(res_face_nodes_connectivity)


def test_connectivity_build_n_nodes_per_face():
    """Tests the construction of the ``n_nodes_per_face`` variable."""
    grids = [grid_mpas, grid_exodus, grid_ugrid]

    for grid in grids:
        max_dimension = grid.n_max_face_nodes
        min_dimension = 3

        assert grid.n_nodes_per_face.values.min() >= min_dimension
        assert grid.n_nodes_per_face.values.max() <= max_dimension

    verts = [f0_deg, f1_deg, f2_deg, f3_deg, f4_deg, f5_deg, f6_deg]
    grid_from_verts = ux.open_grid(verts)

    expected_nodes_per_face = np.array([6, 3, 4, 6, 6, 4, 4], dtype=int)
    nt.assert_equal(grid_from_verts.n_nodes_per_face.values, expected_nodes_per_face)


def test_connectivity_edge_nodes_euler():
    """Verifies that (``n_edge``) follows euler's formula."""
    grid_paths = [exodus_filepath, ugrid_filepath_01, ugrid_filepath_02, ugrid_filepath_03]

    for grid_path in grid_paths:
        grid_ux = ux.open_grid(grid_path)

        n_face = grid_ux.n_face
        n_node = grid_ux.n_node
        n_edge = grid_ux.n_edge

        assert (n_face == n_edge - n_node + 2)


def test_connectivity_build_face_edges_connectivity_mpas():
    """Tests the construction of (``Mesh2_edge_nodes``) on an MPAS grid with known edge nodes."""
    from uxarray.grid.connectivity import _build_edge_node_connectivity

    mpas_grid_ux = ux.open_grid(mpas_filepath)
    edge_nodes_expected = mpas_grid_ux._ds['edge_node_connectivity'].values

    edge_nodes_expected.sort(axis=1)
    edge_nodes_expected = np.unique(edge_nodes_expected, axis=0)

    edge_nodes_output, _, _ = _build_edge_node_connectivity(mpas_grid_ux.face_node_connectivity.values,
                                                            mpas_grid_ux.n_face,
                                                            mpas_grid_ux.n_max_face_nodes)

    assert np.array_equal(edge_nodes_expected, edge_nodes_output)

    n_face = mpas_grid_ux.n_node
    n_node = mpas_grid_ux.n_face
    n_edge = edge_nodes_output.shape[0]

    assert (n_face == n_edge - n_node + 2)


def test_connectivity_build_face_edges_connectivity():
    """Generates Grid.Mesh2_edge_nodes from Grid.face_node_connectivity."""
    ug_filename_list = [ugrid_filepath_01, ugrid_filepath_02, ugrid_filepath_03]
    for ug_file_name in ug_filename_list:
        tgrid = ux.open_grid(ug_file_name)

        face_node_connectivity = tgrid._ds["face_node_connectivity"]

        _populate_face_edge_connectivity(tgrid)
        face_edge_connectivity = tgrid._ds.face_edge_connectivity
        edge_node_connectivity = tgrid._ds.edge_node_connectivity

        assert face_edge_connectivity.sizes["n_face"] == face_node_connectivity.sizes["n_face"]
        assert face_edge_connectivity.sizes["n_max_face_edges"] == face_node_connectivity.sizes["n_max_face_nodes"]

        num_edges = face_edge_connectivity.sizes["n_face"] + tgrid._ds["node_lon"].sizes["n_node"] - 2
        size = edge_node_connectivity.sizes["n_edge"]
        assert edge_node_connectivity.sizes["n_edge"] == num_edges

        original_face_nodes_connectivity = tgrid._ds.face_node_connectivity.values

        reverted_mesh2_edge_nodes = _revert_edges_conn_to_face_nodes_conn(
            edge_nodes_connectivity=edge_node_connectivity.values,
            face_edges_connectivity=face_edge_connectivity.values,
            original_face_nodes_connectivity=original_face_nodes_connectivity
        )

        for i in range(len(reverted_mesh2_edge_nodes)):
            assert np.array_equal(reverted_mesh2_edge_nodes[i], original_face_nodes_connectivity[i])


def test_connectivity_build_face_edges_connectivity_fillvalues():
    verts = [f0_deg, f1_deg, f2_deg, f3_deg, f4_deg, f5_deg, f6_deg]
    uds = ux.open_grid(verts)
    _populate_face_edge_connectivity(uds)
    n_face = len(uds._ds["face_edge_connectivity"].values)
    n_node = uds.n_node
    n_edge = len(uds._ds["edge_node_connectivity"].values)

    assert n_face == 7
    assert n_node == 21
    assert n_edge == 28

    edge_nodes_connectivity = uds._ds["edge_node_connectivity"].values
    face_edges_connectivity = uds._ds["face_edge_connectivity"].values
    face_nodes_connectivity = uds._ds["face_node_connectivity"].values

    res_face_nodes_connectivity = _revert_edges_conn_to_face_nodes_conn(
        edge_nodes_connectivity, face_edges_connectivity, face_nodes_connectivity)

    assert np.array_equal(res_face_nodes_connectivity, uds._ds["face_node_connectivity"].values)


def test_connectivity_node_face_connectivity_from_verts():
    """Test generating Grid.Mesh2_node_faces from array input."""
    face_nodes_conn_lonlat_degree = [[162., 30], [216., 30], [70., 30],
                                     [162., -30], [216., -30], [70., -30]]

    face_nodes_conn_index = np.array([[3, 4, 5, ux.INT_FILL_VALUE],
                                      [3, 0, 2, 5], [3, 4, 1, 0],
                                      [0, 1, 2, ux.INT_FILL_VALUE]])
    face_nodes_conn_lonlat = np.full(
        (face_nodes_conn_index.shape[0], face_nodes_conn_index.shape[1], 2),
        ux.INT_FILL_VALUE)

    for i, face_nodes_conn_index_row in enumerate(face_nodes_conn_index):
        for j, node_index in enumerate(face_nodes_conn_index_row):
            if node_index != ux.INT_FILL_VALUE:
                face_nodes_conn_lonlat[i, j] = face_nodes_conn_lonlat_degree[node_index]

    vgrid = ux.Grid.from_face_vertices(face_nodes_conn_lonlat, latlon=True)

    expected = np.array([
        np.array([0, 1, ux.INT_FILL_VALUE]),
        np.array([1, 3, ux.INT_FILL_VALUE]),
        np.array([0, 1, 2]),
        np.array([1, 2, 3]),
        np.array([0, 2, ux.INT_FILL_VALUE]),
        np.array([2, 3, ux.INT_FILL_VALUE])
    ])

    assert np.array_equal(vgrid.node_face_connectivity.values, expected)


def test_connectivity_node_face_connectivity_from_files():
    """Test generating Grid.Mesh2_node_faces from file input."""
    grid_paths = [exodus_filepath, ugrid_filepath_01, ugrid_filepath_02, ugrid_filepath_03]

    for grid_path in grid_paths:
        grid_xr = xr.open_dataset(grid_path)
        grid_ux = ux.Grid.from_dataset(grid_xr)

        node_face_connectivity = {}
        n_nodes_per_face = grid_ux.n_nodes_per_face.values
        face_nodes = grid_ux._ds["face_node_connectivity"].values
        for face_idx, max_nodes in enumerate(n_nodes_per_face):
            cur_face_nodes = face_nodes[face_idx, 0:max_nodes]
            for j in cur_face_nodes:
                if j not in node_face_connectivity:
                    node_face_connectivity[j] = []
                node_face_connectivity[j].append(face_idx)

        for i in range(grid_ux.n_node):
            face_index_from_sparse_matrix = grid_ux.node_face_connectivity.values[i]
            valid_face_index_from_sparse_matrix = face_index_from_sparse_matrix[
                face_index_from_sparse_matrix != grid_ux.node_face_connectivity.attrs["_FillValue"]]
            valid_face_index_from_sparse_matrix.sort()
            face_index_from_dict = node_face_connectivity[i]
            face_index_from_dict.sort()
            assert np.array_equal(valid_face_index_from_sparse_matrix, face_index_from_dict)


def test_connectivity_edge_face_connectivity_mpas():
    """Tests the construction of ``Mesh2_face_edges`` to the expected results of an MPAS grid."""
    uxgrid = ux.open_grid(mpas_filepath)

    edge_faces_gold = uxgrid.edge_face_connectivity.values

    edge_faces_output = _build_edge_face_connectivity(
        uxgrid.face_edge_connectivity.values,
        uxgrid.n_nodes_per_face.values, uxgrid.n_edge)

    nt.assert_array_equal(edge_faces_output, edge_faces_gold)


def test_connectivity_edge_face_connectivity_sample():
    """Tests the construction of ``Mesh2_face_edges`` on an example with one shared edge, and the remaining edges only being part of one face."""
    verts = [[(0.0, -90.0), (180, 0.0), (0.0, 90)],
             [(-180, 0.0), (0, 90.0), (0.0, -90)]]

    uxgrid = ux.open_grid(verts)

    n_shared = 0
    n_solo = 0
    n_invalid = 0
    for edge_face in uxgrid.edge_face_connectivity.values:
        if edge_face[0] != INT_FILL_VALUE and edge_face[1] != INT_FILL_VALUE:
            n_shared += 1
        elif edge_face[0] != INT_FILL_VALUE and edge_face[1] == INT_FILL_VALUE:
            n_solo += 1
        else:
            n_invalid += 1

    assert n_shared == 1
    assert n_solo == uxgrid.n_edge - n_shared
    assert n_invalid == 0


def test_connectivity_face_face_connectivity_construction():
    """Tests the construction of face-face connectivity."""
    grid = ux.open_grid(mpas_filepath)
    face_face_conn_old = grid.face_face_connectivity.values

    face_face_conn_new = _build_face_face_connectivity(grid)

    face_face_conn_old_sorted = np.sort(face_face_conn_old, axis=None)
    face_face_conn_new_sorted = np.sort(face_face_conn_new, axis=None)

    nt.assert_array_equal(face_face_conn_new_sorted, face_face_conn_old_sorted)


def test_class_methods_from_dataset():
    # UGRID
    xrds = xr.open_dataset(gridfile_ugrid)
    uxgrid = ux.Grid.from_dataset(xrds)

    # MPAS
    xrds = xr.open_dataset(gridfile_mpas)
    uxgrid = ux.Grid.from_dataset(xrds, use_dual=False)
    uxgrid = ux.Grid.from_dataset(xrds, use_dual=True)

    # Exodus
    xrds = xr.open_dataset(gridfile_exodus)
    uxgrid = ux.Grid.from_dataset(xrds)

    # SCRIP
    xrds = xr.open_dataset(gridfile_scrip)
    uxgrid = ux.Grid.from_dataset(xrds)


def test_class_methods_from_face_vertices():
    single_face_latlon = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]
    uxgrid = ux.Grid.from_face_vertices(single_face_latlon, latlon=True)

    multi_face_latlon = [[(0.0, 90.0), (-180, 0.0), (0.0, -90)],
                         [(0.0, 90.0), (180, 0.0), (0.0, -90)]]
    uxgrid = ux.Grid.from_face_vertices(multi_face_latlon, latlon=True)

    single_face_cart = [(0.0,)]


def test_latlon_bounds_populate_bounds_GCA_mix():
    gridfile_mpas = current_path / "meshfiles" / "mpas" / "QU" / "oQU480.231010.nc"
    face_1 = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    face_2 = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
    face_3 = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
    face_4 = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]

    faces = [face_1, face_2, face_3, face_4]

    expected_bounds = [[[0.17453293, 1.07370494], [0.17453293, 0.87266463]],
                       [[0.17453293, 1.10714872], [6.10865238, 0.87266463]],
                       [[1.04719755, 1.57079633], [3.66519143, 0.52359878]],
                       [[1.04719755, 1.57079633], [0., 6.28318531]]]

    grid = ux.Grid.from_face_vertices(faces, latlon=True)
    bounds_xarray = grid.bounds
    nt.assert_allclose(bounds_xarray.values, expected_bounds, atol=ERROR_TOLERANCE)


def test_latlon_bounds_populate_bounds_MPAS():
    gridfile_mpas = current_path / "meshfiles" / "mpas" / "QU" / "oQU480.231010.nc"
    uxgrid = ux.open_grid(gridfile_mpas)
    bounds_xarray = uxgrid.bounds


def test_dual_mesh_mpas():
    grid = ux.open_grid(gridfile_mpas, use_dual=False)
    mpas_dual = ux.open_grid(gridfile_mpas, use_dual=True)

    dual = grid.get_dual()

    assert dual.n_face == mpas_dual.n_face
    assert dual.n_node == mpas_dual.n_node
    assert dual.n_max_face_nodes == mpas_dual.n_max_face_nodes

    nt.assert_equal(dual.face_node_connectivity.values, mpas_dual.face_node_connectivity.values)


def test_dual_duplicate():
    dataset = ux.open_dataset(gridfile_geoflow, gridfile_geoflow)
    with pytest.raises(RuntimeError):
        dataset.get_dual()


def test_normalize_existing_coordinates_non_norm_initial():
    gridfile_mpas = current_path / "meshfiles" / "mpas" / "QU" / "mesh.QU.1920km.151026.nc"
    from uxarray.grid.validation import _check_normalization
    uxgrid = ux.open_grid(gridfile_mpas)

    uxgrid.node_x.data = 5 * uxgrid.node_x.data
    uxgrid.node_y.data = 5 * uxgrid.node_y.data
    uxgrid.node_z.data = 5 * uxgrid.node_z.data
    assert not _check_normalization(uxgrid)

    uxgrid.normalize_cartesian_coordinates()
    assert _check_normalization(uxgrid)


def test_normalize_existing_coordinates_norm_initial():
    gridfile_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    from uxarray.grid.validation import _check_normalization
    uxgrid = ux.open_grid(gridfile_CSne30)

    assert _check_normalization(uxgrid)


def test_number_of_faces_found():
    """Test function for `self.get_face_containing_point`,
    to ensure the correct number of faces is found, depending on where the point is."""
    grid = ux.open_grid(gridfile_mpas)
    partial_grid = ux.open_grid(quad_hex_grid_path)

    # For a face center only one face should be found
    point_xyz = np.array([grid.face_x[100].values, grid.face_y[100].values, grid.face_z[100].values], dtype=np.float64)

    assert len(grid.get_faces_containing_point(point_xyz=point_xyz)) == 1

    # For an edge two faces should be found
    point_xyz = np.array([grid.edge_x[100].values, grid.edge_y[100].values, grid.edge_z[100].values], dtype=np.float64)

    assert len(grid.get_faces_containing_point(point_xyz=point_xyz)) == 2

    # For a node three faces should be found
    point_xyz = np.array([grid.node_x[100].values, grid.node_y[100].values, grid.node_z[100].values], dtype=np.float64)

    assert len(grid.get_faces_containing_point(point_xyz=point_xyz)) == 3

    partial_grid.normalize_cartesian_coordinates()

    # Test for a node on the edge where only 2 faces should be found
    point_xyz = np.array([partial_grid.node_x[1].values, partial_grid.node_y[1].values, partial_grid.node_z[1].values], dtype=np.float64)

    assert len(partial_grid.get_faces_containing_point(point_xyz=point_xyz)) == 2


def test_whole_grid():
    """Tests `self.get_faces_containing_point`on an entire grid,
    checking that for each face center, one face is found to contain it"""

    grid = ux.open_grid(gridfile_mpas_two)
    grid.normalize_cartesian_coordinates()

    # Ensure a face is found on the grid for every face center
    for i in range(len(grid.face_x.values)):
        point_xyz = np.array([grid.face_x[i].values, grid.face_y[i].values, grid.face_z[i].values], dtype=np.float64)

        assert len(grid.get_faces_containing_point(point_xyz=point_xyz)) == 1

def test_point_types():
    """Tests that `self.get_faces_containing_point` works with cartesian and lonlat"""

    # Open the grid
    grid = ux.open_grid(gridfile_mpas)

    # Assign a cartesian point and a lon/lat point
    point_xyz = np.array([grid.node_x[100].values, grid.node_y[100].values, grid.node_z[100].values], dtype=np.float64)
    point_lonlat = np.array([grid.node_lon[100].values, grid.node_lat[100].values])

    # Test both points find faces
    assert len(grid.get_faces_containing_point(point_xyz=point_xyz)) != 0
    assert len(grid.get_faces_containing_point(point_lonlat=point_lonlat)) !=0

def test_point_along_arc():
    node_lon = np.array([-40, -40, 40, 40])
    node_lat = np.array([-20, 20, 20, -20])
    face_node_connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)

    uxgrid = ux.Grid.from_topology(node_lon, node_lat, face_node_connectivity)

    # point at exactly 20 degrees latitude
    out1 = uxgrid.get_faces_containing_point(point_lonlat=np.array([0, 20], dtype=np.float64))

    # point at 25.41 degrees latitude (max along the great circle arc)
    out2 = uxgrid.get_faces_containing_point(point_lonlat=np.array([0, 25.41], dtype=np.float64))

    nt.assert_array_equal(out1, out2)

def test_from_topology():
    node_lon = np.array([-20.0, 0.0, 20.0, -20, -40])
    node_lat = np.array([-10.0, 10.0, -10.0, 10, -10])
    face_node_connectivity = np.array([[0, 1, 2, -1], [0, 1, 3, 4]])

    uxgrid = ux.Grid.from_topology(
        node_lon=node_lon,
        node_lat=node_lat,
        face_node_connectivity=face_node_connectivity,
        fill_value=-1,
    )
