from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

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
    # Allow extra singleton dimensions as long as there's exactly one non-singleton dim
    non_trivial_dims = [dim for dim, size in zip(data.dims, data.shape) if size != 1]

    if len(non_trivial_dims) != 1:
        raise ValueError(
            "Expected data with a single dimension (other axes may be length 1), "
            f"but got dims {data.dims} with shape {data.shape}"
        )

    sole_dim = non_trivial_dims[0]
    if sole_dim != "n_face":
        raise ValueError(f"Expected dimension 'n_face', but got '{sole_dim}'")

    # Squeeze any singleton axes to ensure we return a true 1D array over n_face
    return data.squeeze()


class _RasterAxAttrs(NamedTuple):
    projection: str
    """``str(ax.projection)`` (e.g. a PROJ string)."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]

    shape: tuple[int, int]
    """``(ny, nx)`` shape of the raster grid in pixels.
    Computed using the ``ax`` bbox and ``pixel_ratio``.
    """

    pixel_ratio: float = 1

    @classmethod
    def from_ax(cls, ax: GeoAxes, *, pixel_ratio: float = 1) -> _RasterAxAttrs:
        _, _, w, h = ax.bbox.bounds
        nx = int(w * pixel_ratio)
        ny = int(h * pixel_ratio)
        return cls(
            projection=str(ax.projection),
            xlim=ax.get_xlim(),
            ylim=ax.get_ylim(),
            shape=(ny, nx),
            pixel_ratio=pixel_ratio,
        )

    def to_xr_attrs(self) -> dict[str, Any]:
        """Convert instance to a DataArray attrs dict suitable for saving to nc."""
        return {
            "projection": self.projection,
            "ax_xlim": np.asarray(self.xlim),
            "ax_ylim": np.asarray(self.ylim),
            "ax_shape": np.asarray(self.shape),
            "pixel_ratio": np.float64(self.pixel_ratio),
        }

    @classmethod
    def from_xr_attrs(cls, attrs: dict[str, Any]) -> _RasterAxAttrs:
        """Create instance from the :meth:`to_xr_attrs` attrs dict."""
        return cls(
            projection=attrs["projection"],
            xlim=tuple(x.item() for x in np.asarray(attrs["ax_xlim"])),
            ylim=tuple(x.item() for x in np.asarray(attrs["ax_ylim"])),
            shape=tuple(x.item() for x in np.asarray(attrs["ax_shape"])),
            pixel_ratio=np.float64(attrs["pixel_ratio"]).item(),
        )

    def _value_comparison_message(self, other: _RasterAxAttrs) -> str:
        """Generate a human-readable message describing differences in field values.

        For example: ``'shape (2, 3) != (400, 300). pixel_ratio 2.0 != 1.0.'``,
        where the `other` value is on the LHS.
        """
        parts = []
        for (k, v_self), (_, v_other) in zip(
            self._asdict().items(), other._asdict().items()
        ):
            if v_self != v_other:
                parts.append(f"{k} {v_other} != {v_self}.")
        return " ".join(parts)


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

    ax_attrs = _RasterAxAttrs.from_ax(ax, pixel_ratio=pixel_ratio)

    x0, x1 = ax_attrs.xlim
    y0, y1 = ax_attrs.ylim
    ny, nx = ax_attrs.shape

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
        Pre-computed indices of the first (nearest) face containing each pixel center
        within the Cartopy GeoAxes boundary.
        Pixels in the boundary but not contained in any grid face are marked with -1.

    Returns
    -------
    res : numpy.ndarray, shape (ny, nx)
        Array of resampled data values corresponding to each pixel.
    pixel_mapping : numpy.ndarray, shape (n,)
        Computed using :meth:`~uxarray.Grid.get_faces_containing_point`,
        or the one you passed in.

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
    res.flat[np.flatnonzero(valid)] = flat_vals

    return res, first_face
