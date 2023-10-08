from __future__ import annotations

import matplotlib
from cartopy import crs as ccrs

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray


def plot(uxda, **kwargs):
    """Default Plotting Method for UxDataArray."""
    return datashade(uxda, **kwargs)


def datashade(uxda: UxDataArray,
              *args,
              method: Optional[str] = "polygon",
              plot_height: Optional[int] = 300,
              plot_width: Optional[int] = 600,
              x_range: Optional[tuple] = (-180, 180),
              y_range: Optional[tuple] = (-90, 90),
              cmap: Optional[str] = "Blues",
              agg: Optional[str] = "mean",
              **kwargs):
    """Visualizes an unstructured grid data variable using data shading
    (rasterization + shading).

    Parameters
    ----------
    method: str, optional
        Selects which method to use for data shading
    plot_width, plot_height : int, optional
       Width and height of the output aggregate in pixels.
    x_range, y_range : tuple, optional
       A tuple representing the bounds inclusive space ``[min, max]`` along
       the axis.
    cmap: str, optional
        Colormap used for shading
    agg : str, optional
        Reduction to compute. Default is "mean", but can be one of "mean" or "sum"
    """
    import datashader as ds
    import datashader.transfer_functions as tf

    cvs = ds.Canvas(plot_width, plot_height, x_range, y_range)
    gdf = uxda.to_geodataframe()

    if agg == "mean":
        _agg = ds.mean
    elif agg == "sum":
        _agg = ds.sum
    else:
        raise ValueError("Invalid agg")

    aggregated = cvs.polygons(gdf, geometry='geometry', agg=_agg(uxda.name))

    # support mpl colormaps
    try:
        _cmap = matplotlib.colormaps[cmap]
    except KeyError:
        _cmap = cmap

    return tf.shade(aggregated, cmap=_cmap, **kwargs)


def rasterize(uxda: UxDataArray,
              *args,
              method="point",
              colorbar=True,
              cmap='coolwarm',
              width=1000,
              height=500,
              tools=['hover'],
              projection: Optional[ccrs] = None,
              aggregator='mean',
              interpolation='linear',
              precompute=True,
              dynamic=False,
              npartitions: Optional[int] = 1,
              **kwargs):
    """Visualizes an unstructured grid data variable using rasterization.

    Parameters
    ----------
    projection: cartopy.crs, optional
            Custom projection to transform the axis coordinates during display. Defaults to None.
    npartitions: int, optional
            Number of partions for Dask DataFrane construction
    """
    import dask.dataframe as dd
    import holoviews as hv
    from holoviews.operation.datashader import rasterize as hds_rasterize

    # Face-centered data
    if uxda._is_face_centered():
        lon = uxda.uxgrid.Mesh2_face_x.values
        lat = uxda.uxgrid.Mesh2_face_y.values
    # Node-centered data
    elif uxda._is_node_centered():
        lon = uxda.uxgrid.Mesh2_node_x.values
        lat = uxda.uxgrid.Mesh2_node_y.values
    else:
        raise ValueError(
            "Issue with data. It is neither face-centered nor node-centered!")

    # Transform axis coords w.r.t projection, if any
    if projection is not None:
        lon, lat, _ = projection.transform_points(ccrs.PlateCarree(), lon,
                                                  lat).T

    if method is "point":
        # Construct a point dictionary
        point_dict = {"lon": lon, "lat": lat, "var": uxda.values}

        # Construct Dask DataFrame
        point_ddf = dd.from_dict(data=point_dict, npartitions=npartitions)

        points = hv.Points(point_ddf, ['lon', 'lat'])

        # Rasterize
        raster = hds_rasterize(points,
                               aggregator=aggregator,
                               interpolation=interpolation,
                               precompute=precompute,
                               dynamic=dynamic,
                               **kwargs)
    elif method is "trimesh":
        tris = _grid_to_hvTriMesh(uxda)

        trimesh = _create_hvTriMesh(uxda, lon, lat, tris, npartitions=npartitions)

        # Rasterize
        raster = hds_rasterize(trimesh,
                               aggregator=aggregator,
                               interpolation=interpolation,
                               precompute=precompute,
                               dynamic=dynamic,
                               **kwargs)
    else:
        raise ValueError("method:" + method + " is not supported")

    return raster.opts(width=width,
                       height=height,
                       tools=tools,
                       colorbar=colorbar,
                       cmap=cmap)


def _grid_to_hvTriMesh(uxda):
    if uxda._is_face_centered():
        tris = uxda.uxgrid.Mesh2_node_faces.values

        # The MPAS connectivity array unfortunately does not seem to guarantee consistent clockwise winding order, which
        # is required by Datashader (and Matplotlib)
        #
        tris = _order_CCW(uxda.uxgrid.Mesh2_face_x, uxda.uxgrid.Mesh2_face_y, tris)

        # Lastly, we need to "unzip" the mesh along a constant line of longitude so that when we project to PCS coordinates
        # cells don't wrap around from east to west. The function below does the job, but it assumes that the
        # central_longitude from the map projection is 0.0. I.e. it will cut the mesh where longitude
        # wraps around from -180.0 to 180.0. We'll need to generalize this
        #
        tris = _unzip_mesh(uxda.uxgrid.Mesh2_face_x, tris, 90.0)
    elif uxda._is_node_centered():
        nNodes_per_face = uxda.uxgrid.nNodes_per_face.values - 1

        # For the dual mesh the data are located on triangle centers, which correspond to cell (polygon) vertices. Here
        # we decompose each cell into triangles
        #
        tris = _triangulate_poly(uxda.uxgrid.Mesh2_face_nodes.values, nNodes_per_face)

        tris = _unzip_mesh(uxda.uxgrid.Mesh2_node_x, tris, 90.0)

    else:
        raise ValueError("Issue with data. It is neither face-centered nor node-centered!")

    return tris


def _create_hvTriMesh(uxda: UxDataArray, x, y, triangle_indices, npartitions=1):
    # Create a Holoviews Triangle Mesh suitable for rendering with Datashader
    #
    # This function returns a Holoviews TriMesh that is created from a list of coordinates, 'x' and 'y',
    # an array of triangle indices that addressess the coordinates in 'x' and 'y', and a data variable 'var'. The
    # data variable's values will annotate the triangle vertices

    import numpy as np
    import pandas as pd
    import dask.dataframe as dd
    import holoviews as hv

    # Declare verts array
    verts = np.column_stack([x, y, uxda])

    # Convert to pandas
    verts_df = pd.DataFrame(verts, columns=['x', 'y', 'z'])
    tris_df = pd.DataFrame(triangle_indices, columns=['v0', 'v1', 'v2'])

    # Convert to dask
    verts_ddf = dd.from_pandas(verts_df, npartitions=npartitions)
    tris_ddf = dd.from_pandas(tris_df, npartitions=npartitions)

    # Declare HoloViews element
    tri_nodes = hv.Nodes(verts_ddf, ['x', 'y', 'index'], ['z'])
    trimesh = hv.TriMesh((tris_ddf, tri_nodes))

    return (trimesh)


def _order_CCW(x, y, tris):
    # Reorder triangles as necessary so they all have counter clockwise winding order. CCW is what Datashader and MPL
    # require.

    tris[_triArea(x, y, tris) < 0.0, :] = tris[_triArea(x, y, tris) < 0.0, ::-1]
    return (tris)


def _triArea(x, y, tris):
    # Compute the signed area of a triangle

    return ((x[tris[:, 1]] - x[tris[:, 0]]) * (y[tris[:, 2]] - y[tris[:, 0]])) - (
                (x[tris[:, 2]] - x[tris[:, 0]]) * (y[tris[:, 1]] - y[tris[:, 0]]))


# Triangulate MPAS primary mesh:
#
# Triangulate each polygon in a heterogenous mesh of n-gons by connecting
# each internal polygon vertex to the first vertex. Uses the MPAS
# auxilliary variables verticesOnCell, and nEdgesOnCell.
#
# The function is decorated with Numba's just-in-time compiler so that it is translated into
# optimized machine code for better peformance
#

# from numba import jit
# @jit(nopython=True)
def _triangulate_poly(verticesOnCell, nEdgesOnCell):
    # Calculate the number of triangles. nEdgesOnCell gives the number of vertices for each cell (polygon)
    # The number of triangles per polygon is the number of vertices minus 2.
    #
    import numpy as np

    nTriangles = np.sum(nEdgesOnCell - 2)

    triangles = np.ones((nTriangles, 3), dtype=np.int64)
    nCells = verticesOnCell.shape[0]
    triIndex = 0
    for j in range(nCells):
        for i in range(nEdgesOnCell[j] - 2):
            triangles[triIndex][0] = verticesOnCell[j][0]
            triangles[triIndex][1] = verticesOnCell[j][i + 1]
            triangles[triIndex][2] = verticesOnCell[j][i + 2]
            triIndex += 1

    return triangles


def _unzip_mesh(x, tris, t):
    # This funtion splits a global mesh along longitude
    #
    # Examine the X coordinates of each triangle in 'tris'. Return an array of 'tris' where only those triangles
    # with legs whose length is less than 't' are returned.

    import numpy as np

    return tris[(np.abs((x[tris[:, 0]]) - (x[tris[:, 1]])) < t) & (np.abs((x[tris[:, 0]]) - (x[tris[:, 2]])) < t)]
