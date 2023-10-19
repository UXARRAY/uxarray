import os
import numpy as np
import numpy.testing as nt
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

from spatialpandas.geometry import MultiPolygon

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
datafile_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
datafile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"

grid_files = [gridfile_CSne8, gridfile_geoflow]
data_files = [datafile_CSne30, datafile_geoflow]


class TestAntimeridian(TestCase):

    def test_crossing(self):
        verts = [[[-170, 40], [180, 30], [165, 25], [-170, 20]]]

        uxgrid = ux.open_grid(verts, latlon=True)

        gdf = uxgrid.to_geodataframe()

        assert len(uxgrid.antimeridian_face_indices) == 1

        assert len(gdf['geometry']) == 1

    def test_point_on(self):
        verts = [[[-170, 40], [180, 30], [-170, 20]]]

        uxgrid = ux.open_grid(verts, latlon=True)

        assert len(uxgrid.antimeridian_face_indices) == 1


class TestLineCollection(TestCase):

    def test_linecollection_execution(self):
        uxgrid = ux.open_grid(gridfile_CSne8)
        lines = uxgrid.to_linecollection()


class TestPredicate(TestCase):

    def test_pole_point_inside_polygon_from_vertice_north(self):
        # Define a face as a list of vertices on the unit sphere
        # Here, we're defining a square like structure around the North pole
        vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5]]

        # Normalize the vertices to ensure they lie on the unit sphere
        for i, vertex in enumerate(vertices):
            float_vertex = [float(coord) for coord in vertex]
            vertices[i] = ux.grid.coordinates.normalize_in_place(float_vertex)

        # Create face_edge_cart from the vertices
        face_edge_cart = np.array([[vertices[0], vertices[1]],
                                   [vertices[1], vertices[2]],
                                   [vertices[2], vertices[3]],
                                   [vertices[3], vertices[0]]])

        # Check if the North pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', face_edge_cart)
        self.assertTrue(result, "North pole should be inside the polygon")

        # Check if the South pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'South', face_edge_cart)
        self.assertFalse(result, "South pole should not be inside the polygon")

    def test_pole_point_inside_polygon_from_vertice_south(self):
        # Define a face as a list of vertices on the unit sphere
        # Here, we're defining a square like structure around the south pole
        vertices = [[0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [0.0, 0.0, -1.0]]

        # Normalize the vertices to ensure they lie on the unit sphere
        for i, vertex in enumerate(vertices):
            float_vertex = [float(coord) for coord in vertex]
            vertices[i] = ux.grid.coordinates.normalize_in_place(float_vertex)

        # Create face_edge_cart from the vertices, since we are using the south pole, and want retrive the smaller face
        # we need to reverse the order of the vertices
        # Create face_edge_cart from the vertices
        face_edge_cart = np.array([[vertices[0], vertices[1]],
                                   [vertices[1], vertices[2]],
                                   [vertices[2], vertices[0]]])

        # Check if the North pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', face_edge_cart)
        self.assertFalse(result, "North pole should not be inside the polygon")

        # Check if the South pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'South', face_edge_cart)
        self.assertTrue(result, "South pole should be inside the polygon")

    def test_pole_point_inside_polygon_from_vertice_pole(self):
        # Define a face as a list of vertices on the unit sphere
        # Here, we're defining a square like structure that pole is on the edge
        vertices = [[0, 0, 1], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5]]

        # Normalize the vertices to ensure they lie on the unit sphere
        for i, vertex in enumerate(vertices):
            float_vertex = [float(coord) for coord in vertex]
            vertices[i] = ux.grid.coordinates.normalize_in_place(float_vertex)

        # Create face_edge_cart from the vertices
        face_edge_cart = np.array([[vertices[0], vertices[1]],
                                   [vertices[1], vertices[2]],
                                   [vertices[2], vertices[3]],
                                   [vertices[3], vertices[0]]])

        # Check if the North pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', face_edge_cart)
        self.assertTrue(result, "North pole should be inside the polygon")

        # Check if the South pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'South', face_edge_cart)
        self.assertFalse(result, "South pole should not be inside the polygon")

    def test_pole_point_inside_polygon_from_vertice_cross(self):
        # Define a face that crosses the equator and ecompasses the North pole
        vertices = [[0.6, -0.3, 0.5], [0.2, 0.2, -0.2], [-0.5, 0.1, -0.2],
                    [-0.1, -0.2, 0.2]]

        # Normalize the vertices to ensure they lie on the unit sphere
        for i, vertex in enumerate(vertices):
            float_vertex = [float(coord) for coord in vertex]
            vertices[i] = ux.grid.coordinates.normalize_in_place(float_vertex)

        # Create face_edge_cart from the vertices
        face_edge_cart = np.array([[vertices[0], vertices[1]],
                                   [vertices[1], vertices[2]],
                                   [vertices[2], vertices[3]],
                                   [vertices[3], vertices[0]]])

        # Check if the North pole is inside the polygon
        result = ux.grid.geometry._pole_point_inside_polygon(
            'North', face_edge_cart)
        self.assertTrue(result, "North pole should be inside the polygon")
