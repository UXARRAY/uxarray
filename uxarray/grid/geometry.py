import numpy as np
import xarray as xr
import pandas as pd

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

from uxarray.grid.connectivity import close_face_nodes

from shapely import polygons as Polygons
from shapely import Polygon
from spatialpandas.geometry import MultiPolygonArray
from spatialpandas import GeoDataFrame

import antimeridian


def grid_to_polygons(grid):
    """Constructs an array of ``Shapely`` polygons representing the geometry of
    the unstructrued grid, with corrected antimeridian polygons.

     Parameters
    ----------
    grid : uxarray.Grid
        Grid Object

    Returns
    -------
    polygons : np.ndarray
        Array containing Shapely Polygons
    """
    # obtain polygon shells for shapely polygon construction
    polygon_shells = grid.polygon_shells

    # list of shapely Polygons representing each face in our grid
    polygons = Polygons(polygon_shells)

    # TODO: comment
    if grid.antimeridian_face_indices is not None:

        # TODO: comment
        antimeridian_polygons = polygons[grid.antimeridian_face_indices]

        # TODO: comment
        corrected_polygons = [
            antimeridian.fix_polygon(P) for P in antimeridian_polygons
        ]

        # TODO: comment, possibly optimize?
        for i in reversed(grid.antimeridian_face_indices):
            polygons[i] = corrected_polygons.pop()

    return polygons


def _build_polygon_shells(Mesh2_node_x, Mesh2_node_y, Mesh2_face_nodes,
                          nMesh2_face, nMaxMesh2_face_nodes, nNodes_per_face):
    """Constructs the shell of each polygon derived from the closed off face
    nodes.

    Coordinates should be in degrees, with the longitude being in the
    range [-180, 180].
    """

    # close face nodes to construct closed polygons
    closed_face_nodes = close_face_nodes(Mesh2_face_nodes, nMesh2_face,
                                         nMaxMesh2_face_nodes)

    # additional node after closing our faces
    nNodes_per_face_closed = nNodes_per_face + 1

    if Mesh2_node_x.max() > 180:
        Mesh2_node_x = (Mesh2_node_x + 180) % 360 - 180

    polygon_shells = []
    for face_nodes, max_n_nodes in zip(closed_face_nodes,
                                       nNodes_per_face_closed):

        polygon_x = np.empty_like(face_nodes, dtype=Mesh2_node_x.dtype)
        polygon_y = np.empty_like(face_nodes, dtype=Mesh2_node_x.dtype)

        polygon_x[0:max_n_nodes] = Mesh2_node_x[face_nodes[0:max_n_nodes]]
        polygon_y[0:max_n_nodes] = Mesh2_node_y[face_nodes[0:max_n_nodes]]

        polygon_x[max_n_nodes:] = polygon_x[0]
        polygon_y[max_n_nodes:] = polygon_y[0]

        cur_polygon_shell = np.array([polygon_x, polygon_y])
        polygon_shells.append(cur_polygon_shell.T)

    return np.array(polygon_shells)


# TODO: Update this one
def _build_corrected_polygon_shells(polygon_shells):
    """Constructs ``Grid.corrected_polygon_shells`` and
    ``Grid.original_to_corrected), representing the polygon shells, with
    antimeridian polygons split.

     Parameters
    ----------
    grid : uxarray.Grid
        Grid Object

    Returns
    -------
    corrected_polygon_shells : np.ndarray
        Array containing Shapely Polygons
    original_to_corrected : np.ndarray
        Array containing Shapely Polygons
    """

    polygon_shells = polygon_shells

    # list of shapely Polygons representing each Face in our grid
    polygons = [Polygon(shell) for shell in polygon_shells]

    # List of Polygons (non-split) and MultiPolygons (split across antimeridian)
    corrected_polygons = [antimeridian.fix_polygon(P) for P in polygons]

    original_to_corrected = []
    corrected_polygon_shells = []

    for i, polygon in enumerate(corrected_polygons):

        # Convert MultiPolygons into individual Polygon Vertices
        if polygon.geom_type == "MultiPolygon":
            for individual_polygon in polygon.geoms:
                corrected_polygon_shells.append(
                    np.array([
                        individual_polygon.exterior.coords.xy[0],
                        individual_polygon.exterior.coords.xy[1]
                    ]).T)
                original_to_corrected.append(i)

        # Convert Shapely Polygon into Polygon Vertices
        else:
            corrected_polygon_shells.append(
                np.array([
                    polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1]
                ]).T)
            original_to_corrected.append(i)

    original_to_corrected = np.array(original_to_corrected, dtype=INT_DTYPE)

    return corrected_polygon_shells, original_to_corrected


def _build_antimeridian_face_indices(grid):
    """Constructs ``Grid.antimeridian_face_indices``, which represent the
    indicies of faces that cross the antimeridian.

     Parameters
    ----------
    grid : uxarray.Grid
        Grid Object

    Returns
    -------
    antimeridian_face_indices : np.ndarray
        Array containing Shapely Polygons
    """
    antimeridian_face_indices = np.argwhere(
        np.any(np.abs(np.diff(grid.polygon_shells[:, :, 0])) >= 180,
               axis=1)).squeeze()
    return antimeridian_face_indices


def _grid_to_polygon_geodataframe(grid):

    # obtain faces represented as polygons, corrected on the antimeridian
    polygons = grid_to_polygons(grid)

    # prepare geometry for GeoDataFrame
    geometry = MultiPolygonArray(polygons)

    gdf = GeoDataFrame({"geometry": geometry})

    return gdf
