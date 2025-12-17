from __future__ import annotations

from warnings import warn

import numpy as np
import xarray as xr

from uxarray.constants import INT_DTYPE

from .sample import (
    _fill_numba,
    sample_constant_latitude,
    sample_constant_longitude,
    sample_geodesic,
)


class UxDataArrayCrossSectionAccessor:
    def __init__(self, uxda) -> None:
        self.uxda = uxda

    def __call__(
        self,
        *,
        start: tuple[float, float] | None = None,
        end: tuple[float, float] | None = None,
        lat: float | None = None,
        lon: float | None = None,
        steps: int = 100,
    ) -> xr.DataArray:
        """
        Extracts a cross-section sampled along an arbitrary great-circle arc (GCA) or line of constant latitude/longitude.

        Exactly one mode must be specified:

        - **Great‐circle**: supply both `start` and `end` (lon, lat) tuples,
          samples a geodesic arc.
        - **Constant latitude**: supply `lat` alone (float),
          returns points along that latitude.
        - **Constant longitude**: supply `lon` alone (float),
          returns points along that longitude.

        Parameters
        ----------
        start : tuple[float, float], optional
            (lon, lat) of the geodesic start point.
        end : tuple[float, float], optional
            (lon, lat) of the geodesic end point.
        lat : float, optional
            Latitude for a constant‐latitude slice.
        lon : float, optional
            Longitude for a constant‐longitude slice.
        steps : int, default 100
            Number of sample points (including endpoints).

        Returns
        -------
        xr.DataArray
            A DataArray with a new `"steps"` dimension, plus `"lat"` and `"lon"`
            coordinate variables giving the sampling positions.


        Examples
        --------

        Cross-section between two points (lon, lat)

        >>> uxda.cross_section(start=(-45, -45), end=(45, 45))

        Constant latitude cross-section

        >>> uxda.cross_section(lat=45)

        Constant longitude cross-section

        >>> uxda.cross_section(lon=0)

        Constant longitude cross-section with custom number of steps

        >>> uxda.cross_section(lon=0, steps=200)
        """

        if not self.uxda._face_centered():
            raise ValueError("Data variable must be face-centered")

        if steps < 2:
            raise ValueError("steps must be at least 2")

        great_circle = start is not None or end is not None
        const_lon = lon is not None
        const_lat = lat is not None

        if great_circle and (start is None or end is None):
            raise ValueError(
                "Both 'start' and 'end' must be provided for great-circle mode."
            )

        # exactly one mode
        if sum([great_circle, const_lon, const_lat]) != 1:
            raise ValueError(
                "Must specify exactly one mode (keyword-only): start & end, OR lon, OR lat."
            )

        if great_circle:
            points_xyz, points_latlon = sample_geodesic(start, end, steps)
        elif const_lat:
            points_xyz, points_latlon = sample_constant_latitude(lat, steps)
        else:
            points_xyz, points_latlon = sample_constant_longitude(lon, steps)

        # Find the nearest face for each sample (–1 if no face)
        faces = self.uxda.uxgrid.get_faces_containing_point(
            points_xyz, return_counts=False
        )
        face_idx = np.array([row[0] if row else -1 for row in faces], dtype=INT_DTYPE)

        orig_dims = list(self.uxda.dims)
        face_axis = orig_dims.index("n_face")
        new_dim = "steps"
        new_dims = [new_dim if d == "n_face" else d for d in orig_dims]
        dim_axis = new_dims.index(new_dim)

        arr = np.moveaxis(self.uxda.compute().values, face_axis, -1)
        M, Nf = arr.reshape(-1, arr.shape[-1]).shape
        flat_orig = arr.reshape(M, Nf)

        # Fill along the arc with nearest‐neighbor
        flat_filled = _fill_numba(flat_orig, face_idx, Nf, steps)
        filled = flat_filled.reshape(*arr.shape[:-1], steps)

        # Move steps axis back to its proper position
        data = np.moveaxis(filled, -1, dim_axis)

        # Build coords dict: keep everything except 'n_face'
        coords = {
            name: self.uxda.coords[name]
            for name in self.uxda.coords
            if name != "n_face" and "n_face" not in self.uxda.coords[name].dims
        }
        # index along the arc
        coords[new_dim] = np.arange(steps)

        # attach lat/lon vectors
        coords["lat"] = (new_dim, points_latlon[:, 1])
        coords["lon"] = (new_dim, points_latlon[:, 0])

        return xr.DataArray(
            data,
            dims=new_dims,
            coords=coords,
            name=self.uxda.name,
            attrs=self.uxda.attrs,
        )

    __doc__ = __call__.__doc__

    def constant_latitude(self, *args, **kwargs):
        warn(
            "The ‘.cross_section.constant_latitude’ method is deprecated and will be removed in a future release; "
            "please use the `.subset.constant_latitude` accessor instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.uxda.subset.constant_latitude(*args, **kwargs)

    def constant_longitude(self, *args, **kwargs):
        warn(
            "The ‘.cross_section.constant_longitude’ method is deprecated and will be removed in a future release; "
            "please use the `.subset.constant_longitude` accessor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.uxda.subset.constant_longitude(*args, **kwargs)

    def constant_latitude_interval(self, *args, **kwargs):
        warn(
            "The ‘.cross_section.constant_latitude_interval’ method is deprecated and will be removed in a future release; "
            "please use the `.subset.constant_latitude_interval` accessor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.uxda.subset.constant_latitude_interval(*args, **kwargs)

    def constant_longitude_interval(self, *args, **kwargs):
        warn(
            "The ‘.cross_section.constant_longitude_interval’ method is deprecated and will be removed in a future release; "
            "please use the `.subset.constant_longitude_interval` accessor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.uxda.subset.constant_longitude_interval(*args, **kwargs)
