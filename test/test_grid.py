import os
import numpy as np
import numpy.testing as nt
import xarray as xr
import pytest
from pathlib import Path
import uxarray as ux
from uxarray.grid.connectivity import _populate_face_edge_connectivity, _build_edge_face_connectivity, \
    _build_edge_node_connectivity, _build_face_face_connectivity, _populate_face_face_connectivity
from uxarray.grid.coordinates import _populate_node_latlon
from uxarray.constants import INT_FILL_VALUE, ERROR_TOLERANCE

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
gridfile_mpas = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'
gridfile_mpas_holes = current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc'
shp_filename = current_path / "meshfiles" / "shp" / "grid_fire.shp"

@pytest.fixture
def grid_CSne30():
    return ux.open_grid(gridfile_CSne30)

@pytest.fixture
def grid_RLL1deg():
    return ux.open_grid(gridfile_RLL1deg)

@pytest.fixture
def grid_RLL10deg_CSne4():
    return ux.open_grid(gridfile_RLL10deg_CSne4)

@pytest.fixture
def grid_mpas():
    return ux.open_grid(gridfile_mpas)

@pytest.fixture
def grid_mpas_holes():
    return ux.open_grid(gridfile_mpas_holes)

@pytest.fixture
def grid_geoflow():
    return ux.open_grid(gridfile_geoflow)

def test_validate(grid_mpas):
    """Test to check the validate function."""
    assert grid_mpas.validate()

def test_grid_with_holes(grid_mpas, grid_mpas_holes):
    """Test _holes_in_mesh function."""
    grid_without_holes = grid_mpas
    grid_with_holes = grid_mpas_holes

    assert grid_with_holes.partial_sphere_coverage
    assert grid_without_holes.global_sphere_coverage

def test_encode_as(grid_CSne30, grid_RLL1deg, grid_RLL10deg_CSne4):
    """Reads a ugrid file and encodes it as `xarray.Dataset` in various types."""
    grid_CSne30.encode_as("UGRID")
    grid_RLL1deg.encode_as("UGRID")
    grid_RLL10deg_CSne4.encode_as("UGRID")

    grid_CSne30.encode_as("Exodus")
    grid_RLL1deg.encode_as("Exodus")
    grid_RLL10deg_CSne4.encode_as("Exodus")

def test_init_verts():
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
        [4, 5, 1, 0]   # bottom face
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

def test_init_verts_different_input_datatype():
    """Create a uxarray grid from multiple face vertices with different datatypes(ndarray, list, tuple) and saves a ugrid file."""
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

def test_init_verts_fill_values():
    faces_verts_filled_values = [[[150, 10], [160, 20], [150, 30], [135, 30], [125, 20], [135, 10]],
                                  [[125, 20], [135, 30], [125, 60], [110, 60], [100, 30],
                                   [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]],
                                  [[95, 10], [105, 20], [100, 30], [85, 30],
                                   [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
                                   [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]]
    vgrid = ux.open_grid(faces_verts_filled_values, latlon=False)
    assert vgrid.n_face == 3
    assert vgrid.n_node == 12

def test_grid_properties(grid_CSne30):
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

def test_eq(grid_CSne30):
    """Test Equals ('==') operator."""
    grid_CSne30_02 = ux.open_grid(gridfile_CSne30)
    assert grid_CSne30 == grid_CSne30_02

def test_ne(grid_CSne30):
    """Test Not Equals ('!=') operator."""
    grid_RLL1deg = ux.open_grid(gridfile_RLL1deg)
    assert grid_CSne30 != grid_RLL1deg

def test_calculate_total_face_area_triangle():
    """Create a uxarray grid from vertices and saves an exodus file."""
    verts = [[[0.57735027, -5.77350269e-01, -0.57735027],
              [0.57735027, 5.77350269e-01, -0.57735027],
              [-0.57735027, 5.77350269e-01, -0.57735027]]]

    grid_verts = ux.open_grid(verts, latlon=False)
    assert grid_verts.validate()

    area_gaussian = grid_verts.calculate_total_face_area(quadrature_rule="gaussian", order=5)
    nt.assert_almost_equal(area_gaussian, constants.TRI_AREA, decimal=3)

    area_triangular = grid_verts.calculate_total_face_area(quadrature_rule="triangular", order=4)
    nt.assert_almost_equal(area_triangular, constants.TRI_AREA, decimal=1)

def test_calculate_total_face_area_file():
    """Create a uxarray grid from vertices and saves an exodus file."""
    area = ux.open_grid(gridfile_CSne30).calculate_total_face_area()
    nt.assert_almost_equal(area, constants.MESH30_AREA, decimal=3)

def test_calculate_total_face_area_sphere():
    """Computes the total face area of an MPAS mesh that lies on a unit sphere, with an expected total face area of 4pi."""
    mpas_grid_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'
    primal_grid = ux.open_grid(mpas_grid_path, use_dual=False)
    dual_grid = ux.open_grid(mpas_grid_path, use_dual=True)

    primal_face_area = primal_grid.calculate_total_face_area()
    dual_face_area = dual_grid.calculate_total_face_area()

    nt.assert_almost_equal(primal_face_area, constants.UNIT_SPHERE_AREA, decimal=3)
    nt.assert_almost_equal(dual_face_area, constants.UNIT_SPHERE_AREA, decimal=3)

def test_compute_face_areas_geoflow_small():
    """Checks if the GeoFlow Small can generate a face areas output."""
    grid_geoflow = ux.open_grid(gridfile_geoflow)
    grid_geoflow.compute_face_areas()

def test_verts_calc_area():
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

def test_populate_cartesian_xyz_coord():
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

def test_populate_lonlat_coord():
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

def test_edge_face_connectivity_mpas():
    """Tests the construction of ``Mesh2_face_edges`` to the expected results of an MPAS grid."""
    uxgrid = ux.open_grid(gridfile_mpas)
    edge_faces_gold = uxgrid.edge_face_connectivity.values

    edge_faces_output = _build_edge_face_connectivity(
        uxgrid.face_edge_connectivity.values,
        uxgrid.n_nodes_per_face.values, uxgrid.n_edge)

    nt.assert_array_equal(edge_faces_output, edge_faces_gold)

def test_edge_face_connectivity_sample():
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

def test_face_face_connectivity_construction():
    """Tests the construction of face-face connectivity."""
    grid = ux.open_grid(gridfile_mpas)
    face_face_conn_old = grid.face_face_connectivity.values
    face_face_conn_new = _build_face_face_connectivity(grid)

    face_face_conn_old_sorted = np.sort(face_face_conn_old, axis=None)
    face_face_conn_new_sorted = np.sort(face_face_conn_new, axis=None)

    nt.assert_array_equal(face_face_conn_new_sorted, face_face_conn_old_sorted)

def test_dual_mesh_mpas():
    """Test Dual Mesh Construction."""
    grid = ux.open_grid(gridfile_mpas, use_dual=False)
    mpas_dual = ux.open_grid(gridfile_mpas, use_dual=True)

    dual = grid.get_dual()

    assert dual.n_face == mpas_dual.n_face
    assert dual.n_node == mpas_dual.n_node
    assert dual.n_max_face_nodes == mpas_dual.n_max_face_nodes

    nt.assert_equal(dual.face_node_connectivity.values, mpas_dual.face_node_connectivity.values)

def test_dual_duplicate():
    """Test that dual mesh throws an exception if duplicate nodes exist."""
    dataset = ux.open_dataset(gridfile_geoflow, gridfile_geoflow)
    with pytest.raises(RuntimeError):
        dataset.get_dual()

def test_non_norm_initial():
    """Check the normalization of coordinates that were initially parsed as non-normalized."""
    from uxarray.grid.validation import _check_normalization
    uxgrid = ux.open_grid(gridfile_mpas)

    uxgrid.node_x.data = 5 * uxgrid.node_x.data
    uxgrid.node_y.data = 5 * uxgrid.node_y.data
    uxgrid.node_z.data = 5 * uxgrid.node_z.data
    assert not _check_normalization(uxgrid)

    uxgrid.normalize_cartesian_coordinates()
    assert _check_normalization(uxgrid)

def test_norm_initial():
    """Coordinates should be normalized for grids that we construct them."""
    from uxarray.grid.validation import _check_normalization
    uxgrid = ux.open_grid(gridfile_CSne30)
    assert _check_normalization(uxgrid)
