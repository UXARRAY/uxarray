from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes

    from uxarray import UxDataArray, UxDataset


def _ensure_dimensions(data: UxDataArray) -> UxDataArray:
    """
    Ensures that the data array passed into the plotting routine is exactly one-dimensional over
    the faces of the unstructured grid.

    Raises
    ------
    ValueError
        If the DataArray has more or fewer than one dimension.
    ValueError
        If the sole dimension is not named "n_face".
    """
    # Check dimensionality
    if data.ndim != 1:
        raise ValueError(
            f"Expected a 1D DataArray over 'n_face', but got {data.ndim} dimensions: {data.dims}"
        )

    # Check dimension name
    if data.dims[0] != "n_face":
        raise ValueError(f"Expected dimension 'n_face', but got '{data.dims[0]}'")

    return data


def _get_points_from_axis(ax: GeoAxes, *, pixel_ratio: float = 1):
    """
    Compute 3D Cartesian coordinates for each pixel center in an Axes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The map axes defining the projection and bounds for sampling.
    pixel_ratio : float, default=1.0
        A scaling factor to adjust the resolution of the rasterization.

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
    nx = int(nx * pixel_ratio)
    ny = int(ny * pixel_ratio)

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


def _get_raster_pixel_to_face_mapping(
    obj: UxDataArray | UxDataset,
    ax: GeoAxes,
    *,
    pixel_ratio: float = 1,
):
    """
    Compute a mapping from pixels within a Cartopy GeoAxes to nearest grid face index.

    Parameters
    ----------
    obj : UxDataArray or UxDataset
        Unstructured grid to rasterize.
    ax : cartopy.mpl.geoaxes.GeoAxes
        The target axes defining the sampling grid.
    pixel_ratio : float, default=1.0
        A scaling factor to adjust the resolution of the rasterization.

    Returns
    -------
    pixel_mapping : numpy.ndarray, shape (n,)
        Indices of the first (nearest) grid face containing each pixel center
        within the Cartopy GeoAxes boundary.
        Pixels in the boundary but not contained in any grid face are marked with -1.
    """
    pts, *_ = _get_points_from_axis(ax, pixel_ratio=pixel_ratio)
    face_indices, counts = obj.uxgrid.get_faces_containing_point(pts)

    # pick the first face
    first_face = face_indices[:, 0]
    first_face[counts == 0] = -1

    return first_face


def _nearest_neighbor_resample(
    data: UxDataArray,
    ax: GeoAxes,
    *,
    pixel_ratio: float = 1,
    pixel_mapping: np.ndarray | None = None,
):
    """
    Resample a UxDataArray onto screen-space grid using nearest-neighbor rasterization.

    Parameters
    ----------
    data : UxDataArray
        Unstructured-grid data to rasterize.
    ax : cartopy.mpl.geoaxes.GeoAxes
        The target axes defining the sampling grid.
    pixel_ratio : float, default=1.0
        A scaling factor to adjust the resolution of the rasterization.
    pixel_mapping : numpy.ndarray, optional
        Pre-computed indices of the first (nearest) face containing each pixel center.

    Returns
    -------
    res : numpy.ndarray, shape (ny, nx)
        Array of resampled data values corresponding to each pixel.

    Notes
    -----
    This function determines which face on the grid contains each pixel center and assigns
    the data value of the nearest face to that pixel.
    """
    pts, valid, nx, ny = _get_points_from_axis(ax, pixel_ratio=pixel_ratio)
    if pixel_mapping is None:
        face_indices, counts = data.uxgrid.get_faces_containing_point(pts)

        # pick the first face
        first_face = face_indices[:, 0]
        first_face[counts == 0] = -1
    else:
        first_face = pixel_mapping

    # build an array of values for each valid point
    flat_vals = np.full(first_face.shape, np.nan, dtype=float)
    mask_has_face = first_face >= 0
    flat_vals[mask_has_face] = data.values[first_face[mask_has_face]]

    # scatter back into a full raster via the valid mask
    res = np.full((ny, nx), np.nan, dtype=float)
    res.flat[np.flatnonzero(valid.ravel())] = flat_vals

    return res
