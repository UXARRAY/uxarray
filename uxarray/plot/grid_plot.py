from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.grid import Grid

import cartopy.crs as ccrs

import numpy as np

import holoviews as hv
import pandas as pd


def plot(grid: Grid):
    pass


def mesh(grid: Grid, projection=None, **kwargs):

    node_lon = grid.Mesh2_node_x.values
    node_lat = grid.Mesh2_node_y.values

    if node_lon.max() > 180:
        node_lon = (node_lon + 180) % 360 - 180

    edge_nodes = grid.Mesh2_edge_nodes.values

    edge_x = node_lon[edge_nodes]
    edge_y = node_lat[edge_nodes]

    edge_x_list = []
    edge_y_list = []

    for edge, x, y in zip(edge_nodes, edge_x, edge_y):
        if abs(x[0] - x[1]) >= 180:
            # correct antimeridian edges
            if x[0] <= 0 and x[1] >= 0:

                edge_x_list.extend([x[0], -180, float("nan")])
                edge_y_list.extend([y[0], y[1], float("nan")])

                edge_x_list.extend([x[1], 180, float("nan")])
                edge_y_list.extend([y[1], y[0], float("nan")])
            elif x[0] >= 0 and x[1] <= 0:
                edge_x_list.extend([x[1], -180, float("nan")])
                edge_y_list.extend([y[1], y[0], float("nan")])

                edge_x_list.extend([x[0], 180, float("nan")])
                edge_y_list.extend([y[0], y[1], float("nan")])
        else:
            edge_x_list.extend([x[0], x[1], float("nan")])
            edge_y_list.extend([y[0], y[1], float("nan")])
            continue

    if projection is not None:
        edge_lon_transformed, edge_lat_transformed, _ = projection.transform_points(
            ccrs.PlateCarree(), np.array(edge_x_list), np.array(edge_y_list)).T
    else:
        edge_lon_transformed = np.array(edge_x_list)
        edge_lat_transformed = np.array(edge_y_list)

    ec = pd.DataFrame(np.array([edge_lon_transformed, edge_lat_transformed]).T,
                      columns=['x', 'y'])

    hv.extension("bokeh")

    return hv.Path(ec).opts(**kwargs)
