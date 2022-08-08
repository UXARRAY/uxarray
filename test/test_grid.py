import os
import numpy as np
import xarray as xr
import random
from uxarray import helpers
from unittest import TestCase
from pathlib import Path

import uxarray as ux

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
        ug_filename1 = current_path / "meshfiles" / "outCSne30.ug"
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
        self.assertEqual(mesh2_edge_nodes.sizes["nMesh2_edge"], num_edges)

    def test_generate_Latlon_bounds(self):
        """Generates a latlon_bounds Xarray from grid file."""
        ug_filename1 = current_path / "meshfiles" / "outCSne30.ug"
        tgrid1 = ux.open_dataset(str(ug_filename1))
        tgrid1.buildlatlon_bounds()
        max_lat_list = [-np.pi] * len(tgrid1.ds["Mesh2_face_edges"])
        for i in range(0, len(tgrid1.ds["Mesh2_face_edges"])):
            face = tgrid1.ds["Mesh2_face_edges"].values[i]
            max_lat_face = -np.pi
            for j in range(0, len(face)):
                edge = face[j]
                # Convert the 2D [lon, lat] to 3D [x, y, z]
                n1 = helpers.convert_node_lonlat_rad_to_xyz([
                    np.deg2rad(tgrid1.ds["Mesh2_node_x"].values[edge[0]]),
                    np.deg2rad(tgrid1.ds["Mesh2_node_y"].values[edge[0]])
                ])
                n2 = helpers.convert_node_lonlat_rad_to_xyz([
                    np.deg2rad(tgrid1.ds["Mesh2_node_x"].values[edge[1]]),
                    np.deg2rad(tgrid1.ds["Mesh2_node_y"].values[edge[1]])
                ])
                max_lat_edge = helpers.max_latitude(n1, n2)
                max_lat_face = max(max_lat_edge, max_lat_face)
            max_lat_list[i] = max_lat_face

        for i in range(0, len(tgrid1.ds["Mesh2_face_edges"])):
            lat_max_algo = tgrid1.ds["Mesh2_latlon_bounds"].values[i][0][1]
            lat_max_quant = max_lat_list[i]
            self.assertLessEqual( np.absolute(lat_max_algo - lat_max_quant), 1.0e-12)
            # if np.absolute(lat_max_algo - lat_max_quant) <= 1.0e-12:
            #     print("i: %2d, pass" % (i))
            # else:
            #     # See how this face look like:
            #     face = tgrid1.ds["Mesh2_face_edges"].values[i]
            #     face_latlon_list = []
            #     face_latlon_deg_list = []
            #     for j in range(0, len(face)):
            #         edge = face[j]
            #         # Convert the 2D [lon, lat] to 3D [x, y, z]
            #         n1 = [
            #             np.deg2rad(tgrid1.ds["Mesh2_node_x"].values[edge[0]]),
            #             np.deg2rad(tgrid1.ds["Mesh2_node_y"].values[edge[0]])
            #         ]
            #         n2 = [
            #             np.deg2rad(tgrid1.ds["Mesh2_node_x"].values[edge[1]]),
            #             np.deg2rad(tgrid1.ds["Mesh2_node_y"].values[edge[1]])
            #         ]
            #         face_latlon_list.append([n1, n2])
            #         n1_deg = [
            #             tgrid1.ds["Mesh2_node_x"].values[edge[0]],
            #             tgrid1.ds["Mesh2_node_y"].values[edge[0]]
            #         ]
            #         n2_deg = [
            #             tgrid1.ds["Mesh2_node_x"].values[edge[1]],
            #             tgrid1.ds["Mesh2_node_y"].values[edge[1]]
            #         ]
            #         face_latlon_deg_list.append([n1_deg, n2_deg])
            #     pass

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
