from __future__ import annotations

from typing import TYPE_CHECKING, List, Set, Tuple, Union

if TYPE_CHECKING:
    from xarray import DataArray

    from uxarray import UxDataArray


class UxDataArrayCrossSectionAccessor:
    """Accessor for cross-section operations on a ``UxDataArray``"""

    def __init__(self, uxda) -> None:
        self.uxda = uxda

    def __repr__(self):
        prefix = "<uxarray.UxDataArray.cross_section>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "  * constant_latitude(lat, inverse_indices)\n"
        methods_heading += "  * constant_longitude(lon, inverse_indices)\n"
        methods_heading += "  * constant_latitude_interval(lats, inverse_indices)\n"
        methods_heading += "  * constant_longitude_interval(lons, inverse_indices)\n"

        return prefix + methods_heading

    def constant_latitude(
        self,
        lat: float,
        inverse_indices: Union[List[str], Set[str], bool] = False,
        lon_range: Tuple[float, float] = (-180, 180),
    ) -> UxDataArray | DataArray:
        """Extracts a cross-section of the data array across a line of constant-latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section, in degrees.
            Must be between -90.0 and 90.0
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)
        lon_range: Tuple[float, float], optional
            `(min_lon, max_lon)` longitude values to perform the cross-section. Values must lie in [-180, 180]. Default is `(-180, 180)`.

        Returns
        -------
        uxarray.UxDataArray
            In **grid-based** mode, a subset of the original data array containing only the faces that intersect
            with the specified line of constant latitude.
        xarray.DataArray
            In **interpolated** mode (`interpolate=True`), a new Xarray DataArray with data sampled along the line of constant latitude,
            including longitude and latitude coordinates for each sample.

        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude or the data variable is not face-centered.

        Examples
        --------
        >>> # Extract data at 15.5°S latitude
        >>> cross_section = uxda.cross_section.constant_latitude(lat=-15.5)

        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )

        # TODO: Extend to support constrained ranges
        faces = self.uxda.uxgrid.get_faces_at_constant_latitude(lat)

        if len(faces) == 0:
            raise ValueError(
                f"No faces found that intersect a line of constant latitude at {lat} degrees between {lon_range[0]} and {lon_range[1]} degrees longitude."
            )

        da = self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

        da = da.assign_attrs({"cross_section": True, "constant_latitude": lat})

        return da

    def constant_longitude(
        self,
        lon: float,
        inverse_indices: Union[List[str], Set[str], bool] = False,
        lat_range: Tuple[float, float] = (-90, 90),
    ) -> UxDataArray | DataArray:
        """Extracts a cross-section of the data array across a line of constant-longitude.

        This method supports two modes:
          - **grid‐based** (`interpolate=False`, the default): returns exactly those faces
            which intersect the line of constant longitude, with a new Grid containing those faces.
          - **interpolated** (`interpolate=True`): generates `n_samples` equally‐spaced points
            between `lon_range[0]` and `lon_range[1]` and picks whichever face contains each sample point.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section, in degrees.
            Must be between -180.0 and 180.0
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)
        lat_range: Tuple[float, float], optional
            `(min_lat, max_lat)` latitude values to perform the cross-section. Values must lie in [-90, 90]. Default is `(-90, 90)`.

        Returns
        -------
        uxarray.UxDataArray
            In **grid-based** mode, a subset of the original data array containing only the faces that intersect
            with the specified line of constant longitude.
        xarray.DataArray
            In **interpolated** mode (`interpolate=True`), a new Xarray DataArray with data sampled along the line of constant longitude,
            including longitude and latitude coordinates for each sample.

        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude or the data variable is not face-centered.

        Examples
        --------
        >>> # Extract data at 0° longitude
        >>> cross_section = uxda.cross_section.constant_latitude(lon=0.0)
        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )

        # TODO: Extend to support constrained ranges
        faces = self.uxda.uxgrid.get_faces_at_constant_longitude(
            lon,
        )

        if len(faces) == 0:
            raise ValueError(
                f"No faces found that intersect a line of constant longitude at {lon} degrees between {lat_range[0]} and {lat_range[1]} degrees latitude."
            )

        da = self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

        da = da.assign_attrs({"cross_section": True, "constant_longitude": lon})

        return da

    def constant_latitude_interval(
        self,
        lats: Tuple[float, float],
        inverse_indices: Union[List[str], Set[str], bool] = False,
    ):
        """Extracts a cross-section of data by selecting all faces that
        are within a specified latitude interval.

        Parameters
        ----------
        lats : Tuple[float, float]
            The latitude interval (min_lat, max_lat) at which to extract the cross-section,
            in degrees. Values must be between -90.0 and 90.0
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.UxDataArray
            A subset of the original data array containing only the faces that are within a specified latitude interval.

        Raises
        ------
        ValueError
            If no faces are found within the specified latitude interval.

        Examples
        --------
        >>> # Extract data between 30°S and 30°N latitude
        >>> cross_section = uxda.cross_section.constant_latitude_interval(
        ...     lats=(-30.0, 30.0)
        ... )
        """
        faces = self.uxda.uxgrid.get_faces_between_latitudes(lats)

        return self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

    def constant_longitude_interval(
        self,
        lons: Tuple[float, float],
        inverse_indices: Union[List[str], Set[str], bool] = False,
    ):
        """Extracts a cross-section of data by selecting all faces are within a specifed longitude interval.

        Parameters
        ----------
        lons : Tuple[float, float]
            The longitude interval (min_lon, max_lon) at which to extract the cross-section,
            in degrees. Values must be between -180.0 and 180.0
        inverse_indices : Union[List[str], Set[str], bool], optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - List/Set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.UxDataArray
            A subset of the original data array containing only the faces that intersect
            with the specified longitude interval.

        Raises
        ------
        ValueError
            If no faces are found within the specified longitude interval.

        Examples
        --------
        >>> # Extract data between 0° and 45° longitude
        >>> cross_section = uxda.cross_section.constant_longitude_interval(
        ...     lons=(0.0, 45.0)
        ... )
        """
        faces = self.uxda.uxgrid.get_faces_between_longitudes(lons)

        return self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)
