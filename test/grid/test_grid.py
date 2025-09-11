import os
import numpy as np
import numpy.testing as nt
import xarray as xr

from unittest import TestCase

import uxarray
import uxarray as ux
import pytest

from uxarray.grid.connectivity import _populate_face_edge_connectivity, _build_edge_face_connectivity, \
    _build_edge_node_connectivity, _build_face_face_connectivity, _populate_face_face_connectivity

from uxarray.grid.coordinates import _populate_node_latlon, _lonlat_rad_to_xyz, _xyz_to_lonlat_rad_scalar

from uxarray.constants import INT_FILL_VALUE, ERROR_TOLERANCE

from uxarray.grid.arcs import extreme_gca_latitude

from uxarray.grid.validation import _find_duplicate_nodes

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

def test_grid_ugrid_exodus_roundtrip(gridpath):
    """Test round-trip serialization of grid objects through UGRID and Exodus xarray formats.

    Validates that grid objects can be successfully converted to xarray.Dataset
    objects in both UGRID and Exodus formats, serialized to disk, and reloaded
    while maintaining numerical accuracy and topological integrity.

    The test verifies:
    - Successful conversion to UGRID and Exodus xarray formats
    - File I/O round-trip consistency
    - Preservation of face-node connectivity (exact)
    - Preservation of node coordinates (within numerical tolerance)

    Raises:
        AssertionError: If any round-trip validation fails
    """
    # Load grids
    grid_CSne30 = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    grid_RLL1deg = ux.open_grid(gridpath("ugrid", "outRLL1deg", "outRLL1deg.ug"))
    grid_RLL10deg_CSne4 = ux.open_grid(gridpath("ugrid", "ov_RLL10deg_CSne4", "ov_RLL10deg_CSne4.ug"))

    # Convert grids to xarray.Dataset objects in different formats
    ugrid_datasets = {
        'CSne30': grid_CSne30.to_xarray("UGRID"),
        'RLL1deg': grid_RLL1deg.to_xarray("UGRID"),
        'RLL10deg_CSne4': grid_RLL10deg_CSne4.to_xarray("UGRID")
    }

    exodus_datasets = {
        'CSne30': grid_CSne30.to_xarray("Exodus"),
        'RLL1deg': grid_RLL1deg.to_xarray("Exodus"),
        'RLL10deg_CSne4': grid_RLL10deg_CSne4.to_xarray("Exodus")
    }

    # Define test cases with corresponding grid objects
    test_grids = {
        'CSne30': grid_CSne30,
        'RLL1deg': grid_RLL1deg,
        'RLL10deg_CSne4': grid_RLL10deg_CSne4
    }

    # Perform round-trip validation for each grid type
    test_files = []

    for grid_name in test_grids.keys():
        ugrid_dataset = ugrid_datasets[grid_name]
        exodus_dataset = exodus_datasets[grid_name]
        original_grid = test_grids[grid_name]

        # Define output file paths
        ugrid_filepath = f"test_ugrid_{grid_name}.nc"
        exodus_filepath = f"test_exodus_{grid_name}.exo"
        test_files.append(ugrid_filepath)
        test_files.append(exodus_filepath)

        # Serialize datasets to disk
        ugrid_dataset.to_netcdf(ugrid_filepath)
        exodus_dataset.to_netcdf(exodus_filepath)

        # Reload grids from serialized files
        reloaded_ugrid = ux.open_grid(ugrid_filepath)
        reloaded_exodus = ux.open_grid(exodus_filepath)

        # Validate topological consistency (face-node connectivity)
        # Integer connectivity arrays must be exactly preserved
        np.testing.assert_array_equal(
            original_grid.face_node_connectivity.values,
            reloaded_ugrid.face_node_connectivity.values,
            err_msg=f"UGRID face connectivity mismatch for {grid_name}"
        )
        np.testing.assert_array_equal(
            original_grid.face_node_connectivity.values,
            reloaded_exodus.face_node_connectivity.values,
            err_msg=f"Exodus face connectivity mismatch for {grid_name}"
        )

        # Validate coordinate consistency with numerical tolerance
        # Coordinate transformations and I/O precision may introduce minor differences
        np.testing.assert_allclose(
            original_grid.node_lon.values,
            reloaded_ugrid.node_lon.values,
            err_msg=f"UGRID longitude mismatch for {grid_name}",
            rtol=ERROR_TOLERANCE
        )
        np.testing.assert_allclose(
            original_grid.node_lon.values,
            reloaded_exodus.node_lon.values,
            err_msg=f"Exodus longitude mismatch for {grid_name}",
            rtol=ERROR_TOLERANCE
        )
        np.testing.assert_allclose(
            original_grid.node_lat.values,
            reloaded_ugrid.node_lat.values,
            err_msg=f"UGRID latitude mismatch for {grid_name}",
            rtol=ERROR_TOLERANCE
        )
        np.testing.assert_allclose(
            original_grid.node_lat.values,
            reloaded_exodus.node_lat.values,
            err_msg=f"Exodus latitude mismatch for {grid_name}",
            rtol=ERROR_TOLERANCE
        )

    # This might be need for windows "ermissionError: [WinError 32] -- file accessed by another process"
    reloaded_exodus._ds.close()
    reloaded_ugrid._ds.close()
    del reloaded_exodus
    del reloaded_ugrid

    # Clean up temporary test files
    for filepath in test_files:
        if os.path.exists(filepath):
            os.remove(filepath)

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
    vgrid.to_xarray("UGRID")

    faces_verts_one = np.array([
        np.array([[150, 10], [160, 20], [150, 30], [135, 30], [125, 20], [135, 10]])
    ])
    vgrid = ux.open_grid(faces_verts_one, latlon=True)
    assert vgrid.n_face == 1
    assert vgrid.n_node == 6
    vgrid.to_xarray("UGRID")

    faces_verts_single_face = np.array([[150, 10], [160, 20], [150, 30], [135, 30], [125, 20], [135, 10]])
    vgrid = ux.open_grid(faces_verts_single_face, latlon=True)
    assert vgrid.n_face == 1
    assert vgrid.n_node == 6
    vgrid.to_xarray("UGRID")

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
    vgrid.to_xarray("UGRID")

    faces_verts_list = [[[150, 10], [160, 20], [150, 30], [135, 30], [125, 20], [135, 10]],
                        [[125, 20], [135, 30], [125, 60], [110, 60], [100, 30], [105, 20]],
                        [[95, 10], [105, 20], [100, 30], [85, 30], [75, 20], [85, 10]]]
    vgrid = ux.open_grid(faces_verts_list, latlon=True)
    assert vgrid.n_face == 3
    assert vgrid.n_node == 14
    assert vgrid.validate()
    vgrid.to_xarray("UGRID")

    faces_verts_tuples = [
        ((150, 10), (160, 20), (150, 30), (135, 30), (125, 20), (135, 10)),
        ((125, 20), (135, 30), (125, 60), (110, 60), (100, 30), (105, 20)),
        ((95, 10), (105, 20), (100, 30), (85, 30), (75, 20), (85, 10))
    ]
    vgrid = ux.open_grid(faces_verts_tuples, latlon=True)
    assert vgrid.n_face == 3
    assert vgrid.n_node == 14
    assert vgrid.validate()
    vgrid.to_xarray("UGRID")

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

    xr.testing.assert_equal(grid_geoflow.node_lon, grid_geoflow._ds["node_lon"])
    xr.testing.assert_equal(grid_geoflow.node_lat, grid_geoflow._ds["node_lat"])
    xr.testing.assert_equal(grid_geoflow.face_node_connectivity, grid_geoflow._ds["face_node_connectivity"])

    n_nodes = grid_geoflow.node_lon.shape[0]
    n_faces, n_face_nodes = grid_geoflow.face_node_connectivity.shape

    assert n_nodes == grid_geoflow.n_node
    assert n_faces == grid_geoflow.n_face
    assert n_face_nodes == grid_geoflow.n_max_face_nodes

def test_read_shpfile(test_data_dir):
    """Reads a shape file and write ugrid file."""
    shp_filename = test_data_dir / "shp" / "grid_fire.shp"
    with pytest.raises(ValueError):
        grid_shp = ux.open_grid(shp_filename)

def test_read_scrip(gridpath):
    """Reads a scrip file."""
    grid_CSne8 = ux.open_grid(gridpath("scrip", "outCSne8", "outCSne8.nc"))  # tests from scrip

def test_operators_eq(gridpath):
    """Test Equals ('==') operator."""
    grid_CSne30_01 = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    grid_CSne30_02 = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    assert grid_CSne30_01 == grid_CSne30_02

def test_operators_ne(gridpath):
    """Test Not Equals ('!=') operator."""
    grid_CSne30_01 = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    grid_RLL1deg = ux.open_grid(gridpath("ugrid", "outRLL1deg", "outRLL1deg.ug"))
    assert grid_CSne30_01 != grid_RLL1deg

def test_face_areas_calculate_total_face_area_triangle(mesh_constants):
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
    nt.assert_almost_equal(area_triangular, mesh_constants['TRI_AREA'], decimal=1)

    # calculate area
    area_gaussian = grid_verts.calculate_total_face_area(
        quadrature_rule="gaussian", order=5, latitude_adjusted_area=True)
    nt.assert_almost_equal(area_gaussian, mesh_constants['CORRECTED_TRI_AREA'], decimal=3)

    def test_face_areas_calculate_total_face_area_file(self, gridpath, mesh_constants):
        """Create a uxarray grid from vertices and saves an exodus file."""
        area = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug")).calculate_total_face_area()
        nt.assert_almost_equal(area, mesh_constants['MESH30_AREA'], decimal=3)

    def test_face_areas_calculate_total_face_area_sphere(self, gridpath, mesh_constants):
        """Computes the total face area of an MPAS mesh that lies on a unit sphere, with an expected total face area of 4pi."""
        mpas_grid_path = gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")

        primal_grid = ux.open_grid(mpas_grid_path, use_dual=False)
        dual_grid = ux.open_grid(mpas_grid_path, use_dual=True)

        primal_face_area = primal_grid.calculate_total_face_area()
        dual_face_area = dual_grid.calculate_total_face_area()

        nt.assert_almost_equal(primal_face_area, mesh_constants['UNIT_SPHERE_AREA'], decimal=3)
        nt.assert_almost_equal(dual_face_area, mesh_constants['UNIT_SPHERE_AREA'], decimal=3)

def test_face_areas_compute_face_areas_geoflow_small(gridpath):
    """Checks if the GeoFlow Small can generate a face areas output."""
    grid_geoflow = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))
    grid_geoflow.compute_face_areas()

class TestFaceAreas:
    def test_face_areas_verts_calc_area(self, gridpath, mesh_constants):
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
        nt.assert_almost_equal(face_verts_areas.sum(), mesh_constants['FACE_VERTS_AREA'], decimal=3)

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

def test_connectivity_build_n_nodes_per_face(gridpath):
    """Tests the construction of the ``n_nodes_per_face`` variable."""
    grid_mpas = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    grid_exodus = ux.open_grid(gridpath("exodus", "outCSne8", "outCSne8.g"))
    grid_ugrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
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

def test_connectivity_edge_nodes_euler(gridpath):
    """Verifies that (``n_edge``) follows euler's formula."""
    grid_paths = [gridpath("exodus", "outCSne8", "outCSne8.g"), gridpath("ugrid", "outCSne30", "outCSne30.ug"), gridpath("ugrid", "outRLL1deg", "outRLL1deg.ug"), gridpath("ugrid", "ov_RLL10deg_CSne4", "ov_RLL10deg_CSne4.ug")]

    for grid_path in grid_paths:
        grid_ux = ux.open_grid(grid_path)

        n_face = grid_ux.n_face
        n_node = grid_ux.n_node
        n_edge = grid_ux.n_edge

        assert (n_face == n_edge - n_node + 2)

def test_connectivity_build_face_edges_connectivity_mpas(gridpath):
    """Tests the construction of (``Mesh2_edge_nodes``) on an MPAS grid with known edge nodes."""
    from uxarray.grid.connectivity import _build_edge_node_connectivity

    mpas_grid_ux = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
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

def test_connectivity_build_face_edges_connectivity(gridpath):
    """Generates Grid.Mesh2_edge_nodes from Grid.face_node_connectivity."""
    ug_filename_list = [gridpath("ugrid", "outCSne30", "outCSne30.ug"), gridpath("ugrid", "outRLL1deg", "outRLL1deg.ug"), gridpath("ugrid", "ov_RLL10deg_CSne4", "ov_RLL10deg_CSne4.ug")]
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

def test_connectivity_node_face_connectivity_from_files(gridpath):
    """Test generating Grid.Mesh2_node_faces from file input."""
    grid_paths = [gridpath("exodus", "outCSne8", "outCSne8.g"), gridpath("ugrid", "outCSne30", "outCSne30.ug"), gridpath("ugrid", "outRLL1deg", "outRLL1deg.ug"), gridpath("ugrid", "ov_RLL10deg_CSne4", "ov_RLL10deg_CSne4.ug")]

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

def test_connectivity_edge_face_connectivity_mpas(gridpath):
    """Tests the construction of ``Mesh2_face_edges`` to the expected results of an MPAS grid."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

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

def test_connectivity_face_face_connectivity_construction(gridpath):
    """Tests the construction of face-face connectivity."""
    grid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    face_face_conn_old = grid.face_face_connectivity.values

    face_face_conn_new = _build_face_face_connectivity(grid)

    face_face_conn_old_sorted = np.sort(face_face_conn_old, axis=None)
    face_face_conn_new_sorted = np.sort(face_face_conn_new, axis=None)

    nt.assert_array_equal(face_face_conn_new_sorted, face_face_conn_old_sorted)

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

def test_class_methods_from_face_vertices():
    single_face_latlon = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]
    uxgrid = ux.Grid.from_face_vertices(single_face_latlon, latlon=True)

    multi_face_latlon = [[(0.0, 90.0), (-180, 0.0), (0.0, -90)],
                         [(0.0, 90.0), (180, 0.0), (0.0, -90)]]
    uxgrid = ux.Grid.from_face_vertices(multi_face_latlon, latlon=True)

    single_face_cart = [(0.0,)]

def test_latlon_bounds_populate_bounds_GCA_mix():
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

def test_latlon_bounds_populate_bounds_MPAS(gridpath):
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))
    bounds_xarray = uxgrid.bounds

def test_dual_mesh_mpas(gridpath):
    grid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=False)
    mpas_dual = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"), use_dual=True)

    dual = grid.get_dual()

    assert dual.n_face == mpas_dual.n_face
    assert dual.n_node == mpas_dual.n_node
    assert dual.n_max_face_nodes == mpas_dual.n_max_face_nodes

    nt.assert_equal(dual.face_node_connectivity.values, mpas_dual.face_node_connectivity.values)

def test_dual_duplicate(gridpath):
    dataset = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), gridpath("ugrid", "geoflow-small", "grid.nc"))
    with pytest.raises(RuntimeError):
        dataset.get_dual()

def test_normalize_existing_coordinates_non_norm_initial(gridpath):
    from uxarray.grid.validation import _check_normalization
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    uxgrid.node_x.data = 5 * uxgrid.node_x.data
    uxgrid.node_y.data = 5 * uxgrid.node_y.data
    uxgrid.node_z.data = 5 * uxgrid.node_z.data
    assert not _check_normalization(uxgrid)

    uxgrid.normalize_cartesian_coordinates()
    assert _check_normalization(uxgrid)

def test_normalize_existing_coordinates_norm_initial(gridpath):
    from uxarray.grid.validation import _check_normalization
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    assert _check_normalization(uxgrid)

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

def test_sphere_radius_mpas_ocean(gridpath):
    """Test sphere radius functionality with MPAS ocean mesh."""
    # Test with MPAS ocean mesh file
    mpas_ocean_file = gridpath("mpas", "QU", "oQU480.231010.nc")
    grid = ux.open_grid(mpas_ocean_file)

    # Check that MPAS sphere radius is preserved (Earth's radius)
    assert np.isclose(grid.sphere_radius, 6371229.0, rtol=1e-10)

    # Test setting a new radius
    new_radius = 1000.0
    grid.sphere_radius = new_radius
    assert np.isclose(grid.sphere_radius, new_radius, rtol=1e-10)

    # Test invalid radius
    with pytest.raises(ValueError, match="Sphere radius must be positive"):
        grid.sphere_radius = -1.0
