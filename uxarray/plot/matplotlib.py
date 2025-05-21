from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from uxarray import UxDataArray


def plot(data: UxDataArray, ax=None, **kwargs):
    fig = ax.get_figure()
    fig.canvas.draw()
    res = _sample_grid(data, ax)

    im = ax.imshow(
        res,
        origin="lower",
        extent=ax.get_xlim() + ax.get_ylim(),
        **kwargs,
    )

    return im


def contour(data: UxDataArray, ax=None, **kwargs):
    fig = ax.get_figure()
    fig.canvas.draw()
    res = _sample_grid(data, ax)

    im = ax.contour(
        res,
        origin="lower",
        extent=ax.get_xlim() + ax.get_ylim(),
        **kwargs,
    )

    return im


def contourf(data: UxDataArray, ax=None, **kwargs):
    fig = ax.get_figure()
    fig.canvas.draw()
    res = _sample_grid(data, ax)

    im = ax.contourf(
        res,
        origin="lower",
        extent=ax.get_xlim() + ax.get_ylim(),
        **kwargs,
    )

    return im


def _get_points_from_axis(ax):
    import cartopy.crs as ccrs

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    _, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    x = np.linspace(x0 + dx / 2, x1 - dx / 2, nx)
    y = np.linspace(y0 + dy / 2, y1 - dy / 2, ny)
    x2d, y2d = np.meshgrid(x, y)

    lonlat = ccrs.PlateCarree().transform_points(ax.projection, x2d, y2d)

    valid = np.isfinite(lonlat[..., 0]) & np.isfinite(lonlat[..., 1])

    lons = np.deg2rad(lonlat[..., 0][valid])
    lats = np.deg2rad(lonlat[..., 1][valid])

    xs = np.cos(lats) * np.cos(lons)
    ys = np.cos(lats) * np.sin(lons)
    zs = np.sin(lats)
    pts = np.vstack((xs, ys, zs)).T

    return pts, valid, nx, ny


def _sample_grid(
    data: UxDataArray,
    ax=None,
):
    pts, valid, nx, ny = _get_points_from_axis(ax)
    face_indices, counts = data.uxgrid.get_faces_containing_point(
        point_xyz=pts,
        return_counts=True,
    )

    # TODO: instead of picking the first face, consider an average

    # pick the first face
    first_face = face_indices[:, 0]
    first_face[counts == 0] = -1

    # build an array of values for each valid point
    flat_vals = np.full(first_face.shape, np.nan, dtype=float)
    mask_has_face = first_face >= 0
    flat_vals[mask_has_face] = data.values[first_face[mask_has_face]]

    # scatter back into a full raster via the valid mask
    res = np.full((ny, nx), np.nan, dtype=float)
    res.flat[np.flatnonzero(valid.ravel())] = flat_vals

    return res
