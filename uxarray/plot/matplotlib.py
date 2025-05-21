from __future__ import annotations

from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import matplotlib.pylab as plt
import numpy as np
from cartopy.mpl import geoaxes

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes

    from uxarray import UxDataArray


def imshow(
    data: UxDataArray,
    ax: GeoAxes = None,
    *,
    projection: ccrs.CRS = ccrs.PlateCarree(),
    **kwargs,
) -> plt.AxesImage:
    """
    Render a UxDataArray onto a GeoAxes by sampling the unstructured grid onto a screen-space grid.

    If `ax` is None or not a GeoAxes, a new figure/GeoAxes will be created with the given projection.

    Parameters
    ----------
    data
        Unstructured-grid data array to render.
    ax
        A Cartopy GeoAxes (or None to create one).
    projection
        The map projection to use if creating a new axes.
    **kwargs
        Passed through to `ax.imshow` (e.g., cmap, vmin, vmax).

    Returns
    -------
    The AxesImage returned by `ax.imshow`.
    """
    ax = _ensure_geoaxes(ax, projection=projection)
    fig = ax.get_figure()
    fig.canvas.draw()

    sampled = _nearest_neighbor_resample(data, ax)
    im = ax.imshow(
        sampled,
        origin="lower",
        extent=ax.get_xlim() + ax.get_ylim(),
        **kwargs,
    )

    return im


def _ensure_geoaxes(
    ax,
    *,
    projection,
) -> GeoAxes:
    """
    Return a GeoAxes. If `ax` is already a GeoAxes, return it; otherwise
    create a new figure & GeoAxes with the given projection.
    """
    if isinstance(ax, geoaxes.GeoAxes):
        return ax

    fig, ax = plt.subplots(subplot_kw={"projection": projection})
    return ax


def _get_points_from_axis(ax):
    """
    Compute 3D Cartesian coordinates for each pixel center in an Axes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The map axes defining the projection and bounds for sampling.

    Returns
    -------
    pts : numpy.ndarray, shape (N, 3)
        Cartesian coordinates on the unit sphere corresponding to valid pixel centers.
    valid : numpy.ndarray of bool, shape (ny, nx)
        Boolean mask indicating which pixels correspond to finite longitude and latitude values.
    nx : int
        Number of columns (width) in the pixel grid.
    ny : int
        Number of rows (height) in the pixel grid.
    """
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


def _nearest_neighbor_resample(
    data: UxDataArray,
    ax=None,
):
    """
    Resample a UxDataArray onto screen-space grid using nearest-neighbor rasterization.

    Parameters
    ----------
    data : UxDataArray
        Unstructured-grid data to rasterize.
    ax : matplotlib.axes.Axes
        The target axes defining the sampling grid.

    Returns
    -------
    res : numpy.ndarray, shape (ny, nx)
        Array of resampled data values corresponding to each pixel.

    Notes
    -----
    This function determines which face on the grid contains each pixel center and assigns
    the data value of the nearest face to that pixel.
    """
    pts, valid, nx, ny = _get_points_from_axis(ax)
    face_indices, counts = data.uxgrid.get_faces_containing_point(pts)

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
