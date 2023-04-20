import os
import numpy as np
import xarray as xr

from unittest import TestCase
from pathlib import Path

import xarray as xr
import uxarray as ux
import numpy.testing as nt

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestGrid(TestCase):
    ug_filename1 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    ug_filename2 = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
    ug_filename3 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"

    xr_ds1 = xr.open_dataset(ug_filename1)
    xr_ds2 = xr.open_dataset(ug_filename2)
    xr_ds3 = xr.open_dataset(ug_filename3)
    tgrid1 = ux.Grid(xr_ds1)
    tgrid2 = ux.Grid(xr_ds2)
    tgrid3 = ux.Grid(xr_ds3)

    def test_encode_as(self):
        """Reads a ugrid file and encodes it as `xarray.Dataset` in various
        types."""

        self.tgrid1.encode_as("ugrid")
        self.tgrid2.encode_as("ugrid")
        self.tgrid3.encode_as("ugrid")

        self.tgrid1.encode_as("exodus")
        self.tgrid2.encode_as("exodus")
        self.tgrid3.encode_as("exodus")

    def test_open_non_mesh2_write_exodus(self):
        """Loads grid files of different formats using uxarray's open_dataset
        call."""

        path = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
        xr_grid = xr.open_dataset(path)
        grid = ux.Grid(xr_grid)

        grid.encode_as("exodus")

    def test_init_verts(self):
        """Create a uxarray grid from multiple face vertices with duplicate
        nodes and saves a ugrid file.

        Also, test kwargs for grid initialization

        The input cartesian coordinates represents 8 vertices on a cube
             7---------6
            /|        /|
           / |       / |
          3---------2  |
          |  |      |  |
          |  4------|--5
          | /       | /
          |/        |/
          0---------1
        """
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

        # The order of the vertexes is irrelevant, the following indexing is just for forming a face matrix
        face_vertices = [
            [0, 1, 2, 3],  # front face
            [1, 5, 6, 2],  # right face
            [5, 4, 7, 6],  # back face
            [4, 0, 3, 7],  # left face
            [3, 2, 6, 7],  # top face
            [4, 5, 1, 0]  # bottom face
        ]

        # Pack the cart_x/y/z into the face matrix using the index from face_vertices
        faces_coords = []
        for face in face_vertices:
            face_coords = []
            for vertex_index in face:
                x, y, z = cart_x[vertex_index], cart_y[vertex_index], cart_z[
                    vertex_index]
                face_coords.append([x, y, z])
            faces_coords.append(face_coords)

        # Now consturct the grid using the faces_coords
        verts_cart = np.array(faces_coords)
        vgrid = ux.Grid(verts_cart,
                        vertices=True,
                        islatlon=False,
                        concave=False)

        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 6)
        assert (vgrid.nMesh2_node == 8)
        vgrid.encode_as("ugrid")

        # Test the case when user created a nested one-face grid
        faces_verts_one = np.array([
            np.array([[150, 10], [160, 20], [150, 30], [135, 30], [125, 20],
                      [135, 10]])
        ])
        vgrid = ux.Grid(faces_verts_one,
                        vertices=True,
                        islatlon=True,
                        concave=False)
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 1)
        assert (vgrid.nMesh2_node == 6)
        vgrid.encode_as("ugrid")

        # Test the case when user created a one-face grid
        faces_verts_single_face = np.array([[150, 10], [160, 20], [150, 30],
                                            [135, 30], [125, 20], [135, 10]])

        vgrid = ux.Grid(faces_verts_single_face,
                        vertices=True,
                        islatlon=True,
                        concave=False)
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 1)
        assert (vgrid.nMesh2_node == 6)
        vgrid.encode_as("ugrid")

    def test_init_verts_different_input_datatype(self):
        """Create a uxarray grid from multiple face vertices with different
        datatypes(ndarray, list, tuple) and saves a ugrid file.

        Also, test kwargs for grid initialization
        """

        # Test initializing Grid from ndarray
        faces_verts_ndarray = np.array([
            np.array([[150, 10], [160, 20], [150, 30], [135, 30], [125, 20],
                      [135, 10]]),
            np.array([[125, 20], [135, 30], [125, 60], [110, 60], [100, 30],
                      [105, 20]]),
            np.array([[95, 10], [105, 20], [100, 30], [85, 30], [75, 20],
                      [85, 10]]),
        ])
        vgrid = ux.Grid(faces_verts_ndarray,
                        vertices=True,
                        islatlon=True,
                        concave=False)
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 14)
        vgrid.encode_as("ugrid")

        # Test initializing Grid from list
        faces_verts_list = [[[150, 10], [160, 20], [150, 30], [135, 30],
                             [125, 20], [135, 10]],
                            [[125, 20], [135, 30], [125, 60], [110, 60],
                             [100, 30], [105, 20]],
                            [[95, 10], [105, 20], [100, 30], [85, 30], [75, 20],
                             [85, 10]]]
        vgrid = ux.Grid(faces_verts_list,
                        vertices=True,
                        islatlon=False,
                        concave=False)
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 14)
        vgrid.encode_as("ugrid")

        # Test initializing Grid from tuples
        faces_verts_tuples = [
            ((150, 10), (160, 20), (150, 30), (135, 30), (125, 20), (135, 10)),
            ((125, 20), (135, 30), (125, 60), (110, 60), (100, 30), (105, 20)),
            ((95, 10), (105, 20), (100, 30), (85, 30), (75, 20), (85, 10))
        ]
        vgrid = ux.Grid(faces_verts_tuples,
                        vertices=True,
                        islatlon=False,
                        concave=False)
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 14)
        vgrid.encode_as("ugrid")

    def test_init_verts_fill_values(self):
        faces_verts_filled_values = [[[150, 10], [160, 20], [150, 30],
                                      [135, 30], [125, 20], [135, 10]],
                                     [[125, 20], [135, 30], [125, 60],
                                      [110, 60], [100, 30],
                                      [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]],
                                     [[95, 10], [105, 20], [100, 30], [85, 30],
                                      [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
                                      [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]]
        vgrid = ux.Grid(faces_verts_filled_values,
                        vertices=True,
                        islatlon=False,
                        concave=False)
        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.nMesh2_face == 3)
        assert (vgrid.nMesh2_node == 12)

    def test_init_grid_var_attrs(self):
        """Tests to see if accessing variables through set attributes is equal
        to using the dict."""

        # Dataset with standard UGRID variable names
        # Coordinates
        xr.testing.assert_equal(
            self.tgrid1.Mesh2_node_x,
            self.tgrid1.ds[self.tgrid1.ds_var_names["Mesh2_node_x"]])
        xr.testing.assert_equal(
            self.tgrid1.Mesh2_node_y,
            self.tgrid1.ds[self.tgrid1.ds_var_names["Mesh2_node_y"]])
        # Variables
        xr.testing.assert_equal(
            self.tgrid1.Mesh2_face_nodes,
            self.tgrid1.ds[self.tgrid1.ds_var_names["Mesh2_face_nodes"]])

        # Dimensions
        n_nodes = self.tgrid1.Mesh2_node_x.shape[0]
        n_faces, n_face_nodes = self.tgrid1.Mesh2_face_nodes.shape

        self.assertEqual(n_nodes, self.tgrid1.nMesh2_node)
        self.assertEqual(n_faces, self.tgrid1.nMesh2_face)
        self.assertEqual(n_face_nodes, self.tgrid1.nMaxMesh2_face_nodes)

        # xr.testing.assert_equal(
        #     self.tgrid1.nMesh2_node,
        #     self.tgrid1.ds[self.tgrid1.ds_var_names["nMesh2_node"]])
        # xr.testing.assert_equal(
        #     self.tgrid1.nMesh2_face,
        #     self.tgrid1.ds[self.tgrid1.ds_var_names["nMesh2_face"]])

        # Dataset with non-standard UGRID variable names
        path = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
        xr_grid = xr.open_dataset(path)
        grid = ux.Grid(xr_grid)
        xr.testing.assert_equal(grid.Mesh2_node_x,
                                grid.ds[grid.ds_var_names["Mesh2_node_x"]])
        xr.testing.assert_equal(grid.Mesh2_node_y,
                                grid.ds[grid.ds_var_names["Mesh2_node_y"]])
        # Variables
        xr.testing.assert_equal(grid.Mesh2_face_nodes,
                                grid.ds[grid.ds_var_names["Mesh2_face_nodes"]])
        # Dimensions
        n_nodes = grid.Mesh2_node_x.shape[0]
        n_faces, n_face_nodes = grid.Mesh2_face_nodes.shape

        self.assertEqual(n_nodes, grid.nMesh2_node)
        self.assertEqual(n_faces, grid.nMesh2_face)
        self.assertEqual(n_face_nodes, grid.nMaxMesh2_face_nodes)

    def test_read_shpfile(self):
        """Reads a shape file and write ugrid file."""
        with self.assertRaises(RuntimeError):
            shp_filename = current_path / "meshfiles" / "shp" / "grid_fire.shp"
            tgrid = ux.Grid(str(shp_filename))

    def test_read_scrip(self):
        """Reads a scrip file."""

        scrip_8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
        ug_30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"

        # Test read from scrip and from ugrid for grid class
        xr_grid_s8 = xr.open_dataset(scrip_8)
        ux_grid_s8 = ux.Grid(xr_grid_s8)  # tests from scrip

        xr_grid_u30 = xr.open_dataset(ug_30)
        ux_grid_u30 = ux.Grid(xr_grid_u30)  # tests from ugrid

    def test_build_face_dimension(self):
        """Tests the construction of the ``Mesh2_face_dimension`` variable."""
        grids = [self.tgrid1, self.tgrid2, self.tgrid3]

        for grid in grids:
            # highest possible dimension dimension for a face
            max_dimension = grid.nMaxMesh2_face_nodes

            # face must be at least a triangle
            min_dimension = 3

            assert grid.Mesh2_face_dimension.min() >= min_dimension
            assert grid.Mesh2_face_dimension.max() <= max_dimension


class TestIntegrate(TestCase):
    mesh_file30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    data_file30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"
    data_file30_v2 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"

    def test_calculate_total_face_area_triangle(self):
        """Create a uxarray grid from vertices and saves an exodus file."""
        verts = [[[0.57735027, -5.77350269e-01, -0.57735027],
                  [0.57735027, 5.77350269e-01, -0.57735027],
                  [-0.57735027, 5.77350269e-01, -0.57735027]]]

        # load grid
        vgrid = ux.Grid(verts, vertices=True, islatlon=False, concave=False)

        #calculate area
        area_gaussian = vgrid.calculate_total_face_area(
            quadrature_rule="gaussian", order=5)
        nt.assert_almost_equal(area_gaussian, constants.TRI_AREA, decimal=3)

        area_triangular = vgrid.calculate_total_face_area(
            quadrature_rule="triangular", order=4)
        nt.assert_almost_equal(area_triangular, constants.TRI_AREA, decimal=1)

    def test_calculate_total_face_area_file(self):
        """Create a uxarray grid from vertices and saves an exodus file."""

        xr_grid = xr.open_dataset(str(self.mesh_file30))
        grid = ux.Grid(xr_grid)

        area = grid.calculate_total_face_area()

        nt.assert_almost_equal(area, constants.MESH30_AREA, decimal=3)

    def test_calculate_total_face_area_sphere(self):
        """Computes the total face area of an MPAS mesh that lies on a unit
        sphere, with an expected total face area of 4pi."""
        mpas_grid_path = current_path / 'meshfiles' / "mpas" / "QU" / 'mesh.QU.1920km.151026.nc'

        ds = xr.open_dataset(mpas_grid_path)
        primal_grid = ux.Grid(ds, use_dual=False)
        dual_grid = ux.Grid(ds, use_dual=True)

        primal_face_area = primal_grid.calculate_total_face_area()
        dual_face_area = dual_grid.calculate_total_face_area()

        nt.assert_almost_equal(primal_face_area,
                               constants.UNIT_SPHERE_AREA,
                               decimal=3)

        nt.assert_almost_equal(dual_face_area,
                               constants.UNIT_SPHERE_AREA,
                               decimal=3)

    def test_integrate(self):
        xr_grid = xr.open_dataset(self.mesh_file30)
        xr_psi = xr.open_dataset(self.data_file30)
        xr_v2 = xr.open_dataset(self.data_file30_v2)

        u_grid = ux.Grid(xr_grid)

        integral_psi = u_grid.integrate(xr_psi)
        integral_var2 = u_grid.integrate(xr_v2)

        nt.assert_almost_equal(integral_psi, constants.PSI_INTG, decimal=3)
        nt.assert_almost_equal(integral_var2, constants.VAR2_INTG, decimal=3)


class TestFaceAreas(TestCase):

    def test_compute_face_areas_geoflow_small(self):
        """Checks if the GeoFlow Small can generate a face areas output."""
        geoflow_small_grid = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
        grid_1_ds = xr.open_dataset(geoflow_small_grid)
        grid_1 = ux.Grid(grid_1_ds)
        grid_1.compute_face_areas()

    # removed test until fix to tranposed face nodes
    # def test_compute_face_areas_fesom(self):
    #     """Checks if the FESOM PI-Grid Output can generate a face areas
    #     output."""
    #
    #     fesom_grid_small = current_path / "meshfiles" / "ugrid" / "fesom" / "fesom.mesh.diag.nc"
    #     grid_2_ds = xr.open_dataset(fesom_grid_small)
    #     grid_2 = ux.Grid(grid_2_ds)
    #     grid_2.compute_face_areas()

    def test_verts_calc_area(self):
        faces_verts_ndarray = np.array([
            np.array([[150, 10, 0], [160, 20, 0], [150, 30, 0], [135, 30, 0],
                      [125, 20, 0], [135, 10, 0]]),
            np.array([[125, 20, 0], [135, 30, 0], [125, 60, 0], [110, 60, 0],
                      [100, 30, 0], [105, 20, 0]]),
            np.array([[95, 10, 0], [105, 20, 0], [100, 30, 0], [85, 30, 0],
                      [75, 20, 0], [85, 10, 0]]),
        ])
        # load our vertices into a UXarray Grid object
        verts_grid = ux.Grid(faces_verts_ndarray,
                             vertices=True,
                             islatlon=True,
                             concave=False)

        face_verts_areas = verts_grid.face_areas

        nt.assert_almost_equal(face_verts_areas.sum(),
                               constants.FACE_VERTS_AREA,
                               decimal=3)


class TestPopulateCoordinates(TestCase):

    def test_populate_cartesian_xyz_coord(self):
        # The following testcases are generated through the matlab cart2sph/sph2cart functions
        # These points correspond to the eight vertices of a cube.
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
        vgrid = ux.Grid([verts_degree], islatlon=False)
        vgrid._populate_cartesian_xyz_coord()
        for i in range(0, vgrid.nMesh2_node):
            nt.assert_almost_equal(vgrid.ds["Mesh2_node_cart_x"].values[i],
                                   cart_x[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid.ds["Mesh2_node_cart_y"].values[i],
                                   cart_y[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid.ds["Mesh2_node_cart_z"].values[i],
                                   cart_z[i],
                                   decimal=12)

    def test_populate_lonlat_coord(self):
        # The following testcases are generated through the matlab cart2sph/sph2cart functions
        # These points correspond to the 4 vertexes on a cube.

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
        vgrid = ux.Grid([verts_cart], islatlon=False)
        vgrid._populate_lonlat_coord()
        # The connectivity in `__from_vert__()` will be formed in a reverse order
        lon_deg, lat_deg = zip(*reversed(list(zip(lon_deg, lat_deg))))
        for i in range(0, vgrid.nMesh2_node):
            nt.assert_almost_equal(vgrid.ds["Mesh2_node_x"].values[i],
                                   lon_deg[i],
                                   decimal=12)
            nt.assert_almost_equal(vgrid.ds["Mesh2_node_y"].values[i],
                                   lat_deg[i],
                                   decimal=12)


class TestConnectivity(TestCase):
    mpas_filepath = current_path / "meshfiles" / "mpas" / "QU" / "mesh.QU.1920km.151026.nc"
    exodus_filepath = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
    ugrid_filepath_01 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    ugrid_filepath_02 = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
    ugrid_filepath_03 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"

    def test_build_edge_nodes(self):
        """Tests the construction of (``Mesh2_edge_nodes``) on an MPAS grid
        with known edge nodes."""

        # grid with known edge node connectivity
        mpas_grid_xr = xr.open_dataset(self.mpas_filepath)
        mpas_grid_ux = ux.Grid(mpas_grid_xr)
        edge_nodes_expected = mpas_grid_ux.ds['Mesh2_edge_nodes'].values

        # arrange edge nodes in the same manner as Grid._build_edge_node_connectivity
        edge_nodes_expected.sort(axis=1)
        edge_nodes_expected = np.unique(edge_nodes_expected, axis=0)

        # construct edge nodes
        mpas_grid_ux._build_edge_node_connectivity()
        edge_nodes_output = mpas_grid_ux.ds['Mesh2_edge_nodes'].values

        assert np.array_equal(edge_nodes_expected, edge_nodes_output)

        # euler's formula (n_face = n_edges - n_nodes + 2)
        n_face = mpas_grid_ux.nMesh2_node
        n_node = mpas_grid_ux.nMesh2_face
        n_edge = edge_nodes_output.shape[0]

        assert (n_face == n_edge - n_node + 2)

    def test_edge_nodes_euler(self):
        """Verifies that (``nMesh2_edge``) follows euler's formula."""
        grid_paths = [
            self.exodus_filepath, self.ugrid_filepath_01,
            self.ugrid_filepath_02, self.ugrid_filepath_03
        ]

        for grid_path in grid_paths:
            grid_xr = xr.open_dataset(grid_path)
            grid_ux = ux.Grid(grid_xr)

            n_face = grid_ux.nMesh2_node
            n_node = grid_ux.nMesh2_face
            n_edge = grid_ux.nMesh2_edge

            # euler's formula (n_face = n_edges - n_nodes + 2)
            assert (n_face == n_edge - n_node + 2)
