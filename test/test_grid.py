import os
from pathlib import Path
from unittest import TestCase

import numpy as np
import xarray as xr

import uxarray as ux
from uxarray import helpers, _latlonbound_utilities

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestGrid(TestCase):

    def test_read_ugrid_write_exodus(self):
        """Reads a ugrid file and writes and exodus file."""

        ug_filename1 = current_path / "meshfiles" / "outCSne30.ug"
        ug_filename2 = current_path / "meshfiles" / "outRLL1deg.ug"
        ug_filename3 = current_path / "meshfiles" / "ov_RLL10deg_CSne4.ug"

        ug_outfile1 = current_path / "meshfiles" / "outCSne30.exo"
        ug_outfile2 = current_path / "meshfiles" / "outRLL1deg.g"
        ug_outfile3 = current_path / "meshfiles" / "ov_RLL10deg_CSne4.g"

        tgrid1 = ux.open_dataset(str(ug_filename1))
        tgrid2 = ux.open_dataset(str(ug_filename2))
        tgrid3 = ux.open_dataset(str(ug_filename3))

        tgrid1.write(str(ug_outfile1))
        tgrid2.write(str(ug_outfile2))
        tgrid3.write(str(ug_outfile3))

    def test_init_verts(self):
        """Create a uxarray grid from vertices and saves a ugrid file.

        Also, test kwargs for grid initialization
        """

        verts = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        vgrid = ux.Grid(verts, vertices=True, islatlon=True, concave=False)

        assert (vgrid.source_grid == "From vertices")
        assert (vgrid.source_datasets is None)

        face_filename = current_path / "meshfiles" / "1face.ug"
        vgrid.write(face_filename)

    def test_init_grid_var_attrs(self):
        """Tests to see if accessing variables through set attributes is equal
        to using the dict."""
        # Dataset with Variables in UGRID convention
        path = current_path / "meshfiles" / "outCSne30.ug"
        grid = ux.open_dataset(path)
        xr.testing.assert_equal(grid.Mesh2_node_x,
                                grid.ds[grid.ds_var_names["Mesh2_node_x"]])
        xr.testing.assert_equal(grid.Mesh2_node_y,
                                grid.ds[grid.ds_var_names["Mesh2_node_y"]])
        xr.testing.assert_equal(grid.Mesh2_face_nodes,
                                grid.ds[grid.ds_var_names["Mesh2_face_nodes"]])

        # Dataset with Variables NOT in UGRID convention
        path = current_path / "meshfiles" / "mesh.nc"
        grid = ux.open_dataset(path)
        xr.testing.assert_equal(grid.Mesh2_node_x,
                                grid.ds[grid.ds_var_names["Mesh2_node_x"]])
        xr.testing.assert_equal(grid.Mesh2_node_y,
                                grid.ds[grid.ds_var_names["Mesh2_node_y"]])
        xr.testing.assert_equal(grid.Mesh2_face_nodes,
                                grid.ds[grid.ds_var_names["Mesh2_face_nodes"]])

    def test_generate_edge_nodes(self):
        """Generates Grid.Mesh2_edge_nodes from Grid.Mesh2_face_nodes."""
        ug_filename_list = ["outRLL1deg.ug", "outCSne30.ug", "ov_RLL10deg_CSne4.ug"]
        for ug_file_name in ug_filename_list:
            ug_filename1 = current_path / "meshfiles" / ug_file_name
            tgrid1 = ux.open_dataset(str(ug_filename1))
            mesh2_face_nodes = tgrid1.ds["Mesh2_face_nodes"]

            tgrid1.build_edge_face_connectivity()
            mesh2_face_edges = tgrid1.ds.Mesh2_face_edges
            mesh2_edge_nodes = tgrid1.ds.Mesh2_edge_nodes

            # Assert if the mesh2_face_edges sizes are correct.
            self.assertEqual(mesh2_face_edges.sizes["nMesh2_face"],
                             mesh2_face_nodes.sizes["nMesh2_face"])
            self.assertEqual(mesh2_face_edges.sizes["nMaxMesh2_face_edges"],
                             mesh2_face_nodes.sizes["nMaxMesh2_face_nodes"])
            self.assertEqual(mesh2_face_edges.sizes["Two"], 2)

            # Assert if the mesh2_edge_nodes sizes are correct.
            # Euler formular for determining the edge numbers: n_face = n_edges - n_nodes + 2
            num_edges = mesh2_face_edges.sizes["nMesh2_face"] + tgrid1.ds[
                "Mesh2_node_x"].sizes["nMesh2_node"] - 2
            size = mesh2_edge_nodes.sizes["nMesh2_edge"]
            self.assertEqual(mesh2_edge_nodes.sizes["nMesh2_edge"], num_edges)

    def test_generate_Latlon_bounds_latitude_max(self):
        """Generates a latlon_bounds Xarray from grid file."""
        ug_filename_list = [ "outCSne30.ug", "ov_RLL10deg_CSne4.ug"]
        for ug_file_name in ug_filename_list:
            ug_filename1 = current_path / "meshfiles" / ug_file_name
            tgrid1 = ux.open_dataset(str(ug_filename1))
            tgrid1.buildlatlon_bounds()
            max_lat_list = [-np.pi] * len(tgrid1.ds["Mesh2_face_edges"])
            for i in range(0, len(tgrid1.ds["Mesh2_face_edges"])):
                face = tgrid1.ds["Mesh2_face_edges"].values[i]
                max_lat_face = -np.pi
                for j in range(0, len(face)):
                    edge = face[j]
                    # Skip the dumb edge
                    if edge[0] == -1 or edge[1] == -1:
                        continue

                    n1 = [tgrid1.ds["Mesh2_node_x"].values[edge[0]],
                          tgrid1.ds["Mesh2_node_y"].values[edge[0]]]
                    n2 = [tgrid1.ds["Mesh2_node_x"].values[edge[1]],
                          tgrid1.ds["Mesh2_node_y"].values[edge[1]]]
                    max_lat_edge = _latlonbound_utilities.max_latitude_rad(n1, n2)
                    max_lat_face = max(max_lat_edge, max_lat_face)
                max_lat_list[i] = max_lat_face

            for i in range(0, len(tgrid1.ds["Mesh2_face_edges"])):
                lat_max_algo = tgrid1.ds["Mesh2_latlon_bounds"].values[i][0][1]
                lat_max_quant = max_lat_list[i]
                self.assertLessEqual(np.absolute(lat_max_algo - lat_max_quant), 1.0e-12)

    def test_generate_Latlon_bounds_latitude_min(self):
        """Generates a latlon_bounds Xarray from grid file."""
        ug_filename_list = ["outCSne30.ug", "ov_RLL10deg_CSne4.ug"]
        for ug_file_name in ug_filename_list:
            ug_filename1 = current_path / "meshfiles" / ug_file_name
            tgrid1 = ux.open_dataset(str(ug_filename1))
            tgrid1.buildlatlon_bounds()
            min_lat_list = [np.pi] * len(tgrid1.ds["Mesh2_face_edges"])
            for i in range(0, len(tgrid1.ds["Mesh2_face_edges"])):
                face = tgrid1.ds["Mesh2_face_edges"].values[i]
                min_lat_face = np.pi
                for j in range(0, len(face)):
                    edge = face[j]
                    # Skip the dumb edge
                    if edge[0] == -1 or edge[1] == -1:
                        continue
                    n1 = [tgrid1.ds["Mesh2_node_x"].values[edge[0]],
                          tgrid1.ds["Mesh2_node_y"].values[edge[0]]]
                    n2 = [tgrid1.ds["Mesh2_node_x"].values[edge[1]],
                          tgrid1.ds["Mesh2_node_y"].values[edge[1]]]
                    min_lat_edge = _latlonbound_utilities.min_latitude_rad(n1, n2)
                    min_lat_face = min(min_lat_edge, min_lat_face)
                min_lat_list[i] = min_lat_face

            for i in range(0, len(tgrid1.ds["Mesh2_face_edges"])):
                lat_min_algo = tgrid1.ds["Mesh2_latlon_bounds"].values[i][0][0]
                lat_min_quant = min_lat_list[i]
                self.assertLessEqual(np.absolute(lat_min_algo - lat_min_quant), 1.0e-12)

    def test_generate_Latlon_bounds_longitude_minmax(self):
        """Generates a the longitude boundary Xarray from grid file."""
        ug_filename_list = ["outCSne30.ug", "ov_RLL10deg_CSne4.ug"]
        for ug_file_name in ug_filename_list:
            ug_filename1 = current_path / "meshfiles" / ug_file_name
            tgrid1 = ux.open_dataset(str(ug_filename1))
            tgrid1.buildlatlon_bounds()
            minmax_lon_rad_list = [[404.0, 404.0]] * len(tgrid1.ds["Mesh2_face_edges"])
            for i in range(0, len(tgrid1.ds["Mesh2_face_edges"])):
                face = tgrid1.ds["Mesh2_face_edges"].values[i]
                minmax_lon_rad_face = [404.0, 404.0]
                for j in range(0, len(face)):
                    edge = face[j]
                    # Skip the dumb edge
                    if edge[0] == -1 or edge[1] == -1:
                        continue
                    n1 = [tgrid1.ds["Mesh2_node_x"].values[edge[0]],
                          tgrid1.ds["Mesh2_node_y"].values[edge[0]]]
                    n2 = [tgrid1.ds["Mesh2_node_x"].values[edge[1]],
                          tgrid1.ds["Mesh2_node_y"].values[edge[1]]]

                    # if one of the point is the pole point, then insert the another point only:
                    # North Pole:
                    if (np.absolute(n1[0] - 0) < 1.0e-12 and np.absolute(n1[1] - 90) < 1.0e-12) or (
                            np.absolute(n1[0] - 180) < 1.0e-12 and np.absolute(
                        n1[1] - 90) < 1.0e-12):
                        n1 = n2

                    # South Pole:
                    if (np.absolute(n1[0] - 0) < 1.0e-12 and np.absolute(
                            n1[1] - (-90)) < 1.0e-12) or (
                            np.absolute(n1[0] - 180) < 1.0e-12 and np.absolute(
                        n1[1] - (-90)) < 1.0e-12):
                        n1 = n2

                    # North Pole:
                    if (np.absolute(n2[0] - 0) < 1.0e-12 and np.absolute(n2[1] - 90) < 1.0e-12) or (
                            np.absolute(n2[0] - 180) < 1.0e-12 and np.absolute(
                        n2[1] - 90) < 1.0e-12):
                        n2 = n1

                    # South Pole:
                    if (np.absolute(n2[0] - 0) < 1.0e-12 and np.absolute(
                            n2[1] - (-90)) < 1.0e-12) or (
                            np.absolute(n2[0] - 180) < 1.0e-12 and np.absolute(
                        n2[1] - (-90)) < 1.0e-12):
                        n2 = n1

                    [min_lon_rad_edge, max_lon_rad_edge] = _latlonbound_utilities.minmax_Longitude_rad(n1, n2)
                    if minmax_lon_rad_face[0] == minmax_lon_rad_face[1] == 404.0:
                        minmax_lon_rad_face = [min_lon_rad_edge, max_lon_rad_edge]
                        continue
                    minmax_lon_rad_face = _latlonbound_utilities.expand_longitude_rad(min_lon_rad_edge, max_lon_rad_edge,
                                                                       minmax_lon_rad_face)

                minmax_lon_rad_list[i] = minmax_lon_rad_face

            for i in range(0, len(tgrid1.ds["Mesh2_face_edges"])):
                lon_min_algo = tgrid1.ds["Mesh2_latlon_bounds"].values[i][1][0]
                lon_min_quant = minmax_lon_rad_list[i][0]
                self.assertLessEqual(np.absolute(lon_min_algo - lon_min_quant), 1.0e-12)

                lon_max_algo = tgrid1.ds["Mesh2_latlon_bounds"].values[i][1][1]
                lon_max_quant = minmax_lon_rad_list[i][1]
                self.assertLessEqual(np.absolute(lon_max_algo - lon_max_quant), 1.0e-12)

    def test_nc_zonal_average(self):
        ug_filename_list = ["outCSne30.ug"]
        for ug_file_name in ug_filename_list:
            ug_filename1 = current_path / "meshfiles" / ug_file_name
            tgrid1 = ux.open_dataset(str(ug_filename1))
            tgrid1.get_nc_zonal_avg("temperatrue", 1)

    def test_get_intersection_pt(self):
        ug_filename_list = ["outCSne30.ug"]
        for ug_file_name in ug_filename_list:
            ug_filename1 = current_path / "meshfiles" / ug_file_name


    # TODO: Move to test_shpfile/scrip when implemented
    # use external package to read?
    # https://gis.stackexchange.com/questions/113799/how-to-read-a-shapefile-in-python

    def test_read_shpfile(self):
        """Reads a shape file and write ugrid file."""
        with self.assertRaises(RuntimeError):
            shp_filename = current_path / "meshfiles" / "grid_fire.shp"
            tgrid = ux.open_dataset(str(shp_filename))

    def test_read_scrip(self):
        """Reads a scrip file and write ugrid file."""

        scrip_8 = current_path / "meshfiles" / "outCSne8.nc"
        ug_30 = current_path / "meshfiles" / "outCSne30.ug"

        # Test read from scrip and from ugrid for grid class
        ux.open_dataset(str(scrip_8))  # tests from scrip

        ux.open_dataset(str(ug_30))  # tests from ugrid
