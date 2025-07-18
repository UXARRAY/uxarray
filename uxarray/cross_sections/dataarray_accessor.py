from __future__ import annotations

from typing import TYPE_CHECKING, List, Set, Union

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    pass


class UxDataArrayCrossSectionAccessor:
    """Accessor for cross-section operations on a ``UxDataArray``.

    Provides methods to extract cross-sections of data along constant latitude or longitude lines, as well as within latitude or longitude intervals. Supports both regular sampling and face-based extraction for face-centered data variables.
    """

    def __init__(self, uxda) -> None:
        self.uxda = uxda

    def __repr__(self):
        prefix = "<uxarray.UxDataArray.cross_section>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "Cross-Sections:\n"
        methods_heading += "  * constant_latitude(lat, n_samples=None, lon_range=(-180, 180), method='structured'|'faces', ...)\n"
        methods_heading += "  * constant_longitude(lon, n_samples=None, lat_range=(-90, 90), method='structured'|'faces', ...)\n"
        methods_heading += "  * constant_latitude_interval(lats, inverse_indices)\n"
        methods_heading += "  * constant_longitude_interval(lons, inverse_indices)\n"

        methods_heading += "\nNotes:\n"
        methods_heading += "  - Creates regular sampling along cross-section lines\n"
        methods_heading += "  - Returns coordinate-based data suitable for plotting\n"
        methods_heading += "  - Works with any dimensionality (2D, 3D, 4D, etc.)\n"

        return prefix + methods_heading

    def _check_vertical_coord_exists(self, vertical_coord):
        """Check if a specified vertical coordinate exists in the data dimensions."""
        if vertical_coord not in self.uxda.dims:
            available_dims = list(self.uxda.dims)
            raise ValueError(
                f"Vertical coordinate '{vertical_coord}' not found in data dimensions. "
                f"Available dimensions: {available_dims}"
            )
        return vertical_coord

    def _get_vertical_coord_name(self, vertical_coord):
        """Validate the vertical coordinate dimension name."""
        if vertical_coord is None:
            raise ValueError(
                "A vertical coordinate must be explicitly specified. "
                "Use the 'vertical_coord' parameter to specify which dimension represents depth/height."
            )
        return self._check_vertical_coord_exists(vertical_coord)

    def constant_latitude(
        self,
        lat: float,
        sample: bool = False,
        n_samples: int = None,
        lon_range: tuple = (-180, 180),
        inverse_indices: "Union[List[str], Set[str], bool]" = False,
    ):
        """
        Extract a cross-section of the data array along a line of constant latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section, in degrees. Must be between -90.0 and 90.0.
        sample : bool, optional
            If True, creates a regularly sampled cross-section along the latitude using `n_samples` points. If False, extracts faces intersecting the latitude. Default is False.
        n_samples : int, optional
            Number of sample points to create along the longitude direction. Used only if `sample` is True. If not provided, defaults to the number of faces intersecting the latitude.
        lon_range : tuple, optional
            Longitude range (min_lon, max_lon) for sampling. Default is (-180, 180).
        inverse_indices : list, set, or bool, optional
            Controls storage of original grid indices. See documentation for options. Default is False.

        Returns
        -------
        xarray.DataArray
            A DataArray with dimensions including 'sample' (if sampled) or 'n_face' (if not), and coordinates including 'lon', 'lat', and any other dimensions from the original data.

        Raises
        ------
        ValueError
            If the data variable is not face-centered.

        Examples
        --------
        >>> # Extract cross-section at 45°N latitude with 100 sample points
        >>> cross_section = uxda.cross_section.constant_latitude(
        ...     lat=45.0, sample=True, n_samples=100
        ... )

        >>> # Extract cross-section with custom longitude range
        >>> cross_section = uxda.cross_section.constant_latitude(
        ...     lat=0.0, sample=True, n_samples=50, lon_range=(-90, 90)
        ... )

        Notes
        -----
        - Creates regular sampling along the latitude line if `sample` is True
        - Uses nearest-neighbor interpolation from unstructured grid
        - Works with any dimensionality (2D, 3D, 4D, etc.)
        - The initial execution time may be longer due to Numba compilation
        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )
        faces = self.uxda.uxgrid.get_faces_at_constant_latitude(lat)
        # If n_samples is provided, treat as structured sampling
        if n_samples is not None:
            sample = True
        if sample:
            return self._structured_constant_latitude(
                lat, lon_range, n_samples=n_samples or len(faces)
            )
        else:
            return self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

    def constant_longitude(
        self,
        lon: float,
        sample: bool = False,
        n_samples: int = None,
        lat_range: tuple = (-90, 90),
        inverse_indices: "Union[List[str], Set[str], bool]" = False,
    ):
        """
        Extract a cross-section of the data array along a line of constant longitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section, in degrees. Must be between -180.0 and 180.0.
        sample : bool, optional
            If True, creates a regularly sampled cross-section along the longitude using `n_samples` points. If False, extracts faces intersecting the longitude. Default is False.
        n_samples : int, optional
            Number of sample points to create along the latitude direction. Used only if `sample` is True. If not provided, defaults to the number of faces intersecting the longitude.
        lat_range : tuple, optional
            Latitude range (min_lat, max_lat) for sampling. Default is (-90, 90).
        inverse_indices : list, set, or bool, optional
            Controls storage of original grid indices. See documentation for options. Default is False.

        Returns
        -------
        xarray.DataArray
            A DataArray with dimensions including 'sample' (if sampled) or 'n_face' (if not), and coordinates including 'lon', 'lat', and any other dimensions from the original data.

        Raises
        ------
        ValueError
            If the data variable is not face-centered.

        Examples
        --------
        >>> # Extract cross-section at 0° longitude with 100 sample points
        >>> cross_section = uxda.cross_section.constant_longitude(
        ...     lon=0.0, sample=True, n_samples=100
        ... )

        >>> # Extract cross-section with custom latitude range
        >>> cross_section = uxda.cross_section.constant_longitude(
        ...     lon=90.0, sample=True, n_samples=50, lat_range=(-45, 45)
        ... )

        Notes
        -----
        - Creates regular sampling along the longitude line if `sample` is True
        - Uses nearest-neighbor interpolation from unstructured grid
        - Works with any dimensionality (2D, 3D, 4D, etc.)
        - The initial execution time may be longer due to Numba compilation
        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )
        faces = self.uxda.uxgrid.get_faces_at_constant_longitude(lon)
        # If n_samples is provided, treat as structured sampling
        if n_samples is not None:
            sample = True
        if sample:
            return self._structured_constant_longitude(
                lon, lat_range, n_samples=n_samples or len(faces)
            )
        else:
            return self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

    def constant_latitude_interval(
        self,
        lats: tuple,
        inverse_indices: "Union[List[str], Set[str], tuple, bool]" = False,
    ):
        """
        Extract a horizontal cross-section of data by selecting all faces that are within a specified latitude interval.

        Parameters
        ----------
        lats : tuple
            The latitude interval (min_lat, max_lat) at which to extract the cross-section, in degrees. Values must be between -90.0 and 90.0.
        inverse_indices : list, set, tuple, or bool, optional
            Controls storage of original grid indices. See documentation for options. Default is False.

        Returns
        -------
        xarray.DataArray
            A subset of the original data array containing only the faces that are within a specified latitude interval.

        Raises
        ------
        ValueError
            If no faces are found within the specified latitude interval.

        Examples
        --------
        >>> # Extract horizontal slice between 30°S and 30°N latitude
        >>> cross_section = uxda.cross_section.constant_latitude_interval(
        ...     lats=(-30.0, 30.0)
        ... )

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        faces = self.uxda.uxgrid.get_faces_between_latitudes(lats)
        return self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

    def constant_longitude_interval(
        self,
        lons: tuple,
        inverse_indices: "Union[List[str], Set[str], tuple, bool]" = False,
    ):
        """
        Extract a horizontal cross-section of data by selecting all faces that are within a specified longitude interval.

        Parameters
        ----------
        lons : tuple
            The longitude interval (min_lon, max_lon) at which to extract the cross-section, in degrees. Values must be between -180.0 and 180.0.
        inverse_indices : list, set, tuple, or bool, optional
            Controls storage of original grid indices. See documentation for options. Default is False.

        Returns
        -------
        xarray.DataArray
            A subset of the original data array containing only the faces that intersect with the specified longitude interval.

        Raises
        ------
        ValueError
            If no faces are found within the specified longitude interval.

        Examples
        --------
        >>> # Extract horizontal slice between 0° and 45° longitude
        >>> cross_section = uxda.cross_section.constant_longitude_interval(
        ...     lons=(0.0, 45.0)
        ... )

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        faces = self.uxda.uxgrid.get_faces_between_longitudes(lons)
        return self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

    def _structured_constant_latitude(
        self,
        lat: float,
        lon_range: tuple,
        n_samples=None,
    ):
        """
        Create a structured cross-section at constant latitude.

        Generates a regularly sampled cross-section along a specified latitude, using `n_samples` points between the given longitude range. Returns a DataArray with a 'sample' dimension and corresponding coordinates.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section.
        lon_range : tuple
            Longitude range (min_lon, max_lon) for sampling.
        n_samples : int
            Number of sample points to create along the longitude direction.

        Returns
        -------
        xarray.DataArray
            DataArray with a 'sample' dimension and coordinates 'lon' and 'lat'.
        """
        # Create sample points along longitude
        lons = np.linspace(lon_range[0], lon_range[1], n_samples)
        lats = np.ones_like(lons) * lat

        # Find which face(s) each point hits
        faces_along_lat = self.uxda.uxgrid.get_faces_containing_point(
            np.column_stack((lons, lats)), return_counts=False
        )

        # Collapse to a 1D index: use -1 where no face hits
        face_idx = np.array(
            [row[0] if row else -1 for row in faces_along_lat], dtype=int
        )

        # Get the shape and dimensions of the original data
        original_shape = self.uxda.shape
        original_dims = list(self.uxda.dims)
        n_face_dim = original_dims.index("n_face")

        # Create new shape and dimensions with samples replacing n_face
        new_shape = list(original_shape)
        new_shape[n_face_dim] = n_samples
        new_dims = original_dims.copy()
        new_dims[n_face_dim] = "sample"

        # Pre-allocate with NaN
        data = np.full(new_shape, fill_value=np.nan, dtype=self.uxda.dtype)
        valid = face_idx >= 0

        # Only process if we have valid indices
        if np.any(valid):
            valid_face_idx = face_idx[valid]

            # Use numpy's advanced indexing to handle arbitrary dimensions
            # Create a list of indices for each dimension
            indices = []
            for i, dim_size in enumerate(original_shape):
                if i == n_face_dim:
                    indices.append(valid_face_idx)
                else:
                    indices.append(slice(None))

            # Create output indices
            out_indices = []
            for i, dim_size in enumerate(new_shape):
                if i == n_face_dim:
                    out_indices.append(valid)
                else:
                    out_indices.append(slice(None))

            # Extract and assign data
            data[tuple(out_indices)] = self.uxda.values[tuple(indices)]
        # If all face_idx are -1, data remains all-NaN (valid)

        # Create coordinates
        coords = {}
        coords["sample"] = np.arange(n_samples)
        coords["lon"] = ("sample", lons)
        coords["lat"] = ("sample", lats)

        # Copy over other coordinates that don't depend on n_face
        for coord_name, coord in self.uxda.coords.items():
            if "n_face" not in coord.dims:
                coords[coord_name] = coord

        # Wrap in DataArray with proper coordinates
        da = xr.DataArray(
            data,
            dims=new_dims,
            coords=coords,
            attrs=self.uxda.attrs.copy(),
        )

        return da

    def _structured_constant_longitude(
        self,
        lon: float,
        lat_range: tuple,
        n_samples=None,
    ):
        """
        Create a structured cross-section at constant longitude.

        Generates a regularly sampled cross-section along a specified longitude, using `n_samples` points between the given latitude range. Returns a DataArray with a 'sample' dimension and corresponding coordinates.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section.
        lat_range : tuple
            Latitude range (min_lat, max_lat) for sampling.
        n_samples : int
            Number of sample points to create along the latitude direction.

        Returns
        -------
        xarray.DataArray
            DataArray with a 'sample' dimension and coordinates 'lon' and 'lat'.
        """
        # Create sample points along latitude
        lats = np.linspace(lat_range[0], lat_range[1], n_samples)
        lons = np.ones_like(lats) * lon

        # Find which face(s) each point hits
        faces_along_lon = self.uxda.uxgrid.get_faces_containing_point(
            np.column_stack((lons, lats)), return_counts=False
        )

        # Collapse to a 1D index: use -1 where no face hits
        face_idx = np.array(
            [row[0] if row else -1 for row in faces_along_lon], dtype=int
        )

        # Get the shape and dimensions of the original data
        original_shape = self.uxda.shape
        original_dims = list(self.uxda.dims)
        n_face_dim = original_dims.index("n_face")

        # Create new shape and dimensions with samples replacing n_face
        new_shape = list(original_shape)
        new_shape[n_face_dim] = n_samples
        new_dims = original_dims.copy()
        new_dims[n_face_dim] = "sample"

        # Pre-allocate with NaN
        data = np.full(new_shape, fill_value=np.nan, dtype=self.uxda.dtype)
        valid = face_idx >= 0

        # Only process if we have valid indices
        if np.any(valid):
            valid_face_idx = face_idx[valid]

            # Use numpy's advanced indexing to handle arbitrary dimensions
            # Create a list of indices for each dimension
            indices = []
            for i, dim_size in enumerate(original_shape):
                if i == n_face_dim:
                    indices.append(valid_face_idx)
                else:
                    indices.append(slice(None))

            # Create output indices
            out_indices = []
            for i, dim_size in enumerate(new_shape):
                if i == n_face_dim:
                    out_indices.append(valid)
                else:
                    out_indices.append(slice(None))

            # Extract and assign data
            data[tuple(out_indices)] = self.uxda.values[tuple(indices)]
        # If all face_idx are -1, data remains all-NaN (valid)

        # Create coordinates
        coords = {}
        coords["sample"] = np.arange(n_samples)
        coords["lon"] = ("sample", lons)
        coords["lat"] = ("sample", lats)

        # Copy over other coordinates that don't depend on n_face
        for coord_name, coord in self.uxda.coords.items():
            if "n_face" not in coord.dims:
                coords[coord_name] = coord

        # Wrap in DataArray with proper coordinates
        da = xr.DataArray(
            data,
            dims=new_dims,
            coords=coords,
            attrs=self.uxda.attrs.copy(),
        )

        return da
