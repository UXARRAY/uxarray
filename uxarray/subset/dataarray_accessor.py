from __future__ import annotations

import numpy as np


class DataArraySubsetAccessor:
    """Accessor for performing unstructured grid subsetting with a data
    variable, accessed through ``UxDataArray.subset``"""

    def __init__(self, uxda) -> None:
        self.uxda = uxda

    def __repr__(self):
        prefix = "<uxarray.UxDataArray.subset>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "  * nearest_neighbor(center_coord, k, element)\n"
        methods_heading += "  * bounding_circle(center_coord, r, element)\n"
        methods_heading += "  * bounding_box(lon_bounds, lat_bounds)\n"
        methods_heading += "  * constant_latitude(lat, lon_range)\n"
        methods_heading += "  * constant_longitude(lon, lat_range)\n"
        methods_heading += "  * constant_latitude_interval(lats)\n"
        methods_heading += "  * constant_longitude_interval(lons)\n"

        return prefix + methods_heading

    def bounding_box(
        self,
        lon_bounds: tuple | list | np.ndarray,
        lat_bounds: tuple | list | np.ndarray,
        inverse_indices: list[str] | set[str] | bool = False,
    ):
        """Subsets an unstructured grid between two latitude and longitude
        points which form a bounding box.

        A bounding box may span the antimeridian, when the pair of longitude points is given in descending order (
        i.e. the first longitude point is greater than the second).

        Parameters
        ----------
        lon_bounds: tuple, list, np.ndarray
            (lon_left, lon_right) where lon_left < lon_right when the bounding box does not span
            the antimeridian, otherwise lon_left > lon_right, both between [-180, 180]
        lat_bounds: tuple, list, np.ndarray
            (lat_bottom, lat_top) where lat_top > lat_bottom and between [-90, 90]
        method: str
            Bounding Box Method, currently supports 'coords', which ensures the coordinates of the corner nodes,
            face centers, or edge centers lie within the bounds.
        element: str
            Element for use with `coords` comparison, one of `nodes`, `face centers`, or `edge centers`
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)
        """
        grid = self.uxda.uxgrid.subset.bounding_box(
            lon_bounds, lat_bounds, inverse_indices=inverse_indices
        )

        return self.uxda._slice_from_grid(grid)

    def bounding_circle(
        self,
        center_coord: tuple | list | np.ndarray,
        r: float | int,
        element: str | None = "face centers",
        inverse_indices: list[str] | set[str] | bool = False,
        **kwargs,
    ):
        """Subsets an unstructured grid by returning all elements within some
        radius (in degrees) from a center coord.

        Parameters
        ----------
        center_coord : tuple, list, np.ndarray
            Longitude and latitude of the center of the bounding circle
        r: scalar, int, float
            Radius of bounding circle (in degrees)
        element: str
            Element for use with `coords` comparison, one of `nodes`, `face centers`, or `edge centers`
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)
        """
        grid = self.uxda.uxgrid.subset.bounding_circle(
            center_coord, r, element, inverse_indices=inverse_indices, **kwargs
        )
        return self.uxda._slice_from_grid(grid)

    def nearest_neighbor(
        self,
        center_coord: tuple | list | np.ndarray,
        k: int,
        element: str | None = "face centers",
        inverse_indices: list[str] | set[str] | bool = False,
        **kwargs,
    ):
        """Subsets an unstructured grid by returning the ``k`` closest
        neighbors from a center coordinate.

        Parameters
        ----------
        center_coord : tuple, list, np.ndarray
            Longitude and latitude of the center of the bounding circle
        k: int
            Number of neighbors to query
        element: str
            Element for use with `coords` comparison, one of `nodes`, `face centers`, or `edge centers`
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)
        """

        grid = self.uxda.uxgrid.subset.nearest_neighbor(
            center_coord, k, element, inverse_indices=inverse_indices, **kwargs
        )

        return self.uxda._slice_from_grid(grid)

    def constant_latitude(
        self,
        lat: float,
        inverse_indices: list[str] | set[str] | bool = False,
    ):
        """Extracts a subset of the data array across a line of constant-latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the subset, in degrees.
            Must be between -90.0 and 90.0
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.UxDataArray
            A new UxDataArray containing elements that intersect the line of constant latitude.

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

        faces = self.uxda.uxgrid.get_faces_at_constant_latitude(lat)

        if len(faces) == 0:
            raise ValueError(
                f"No faces found that intersect a line of constant latitude at {lat} degrees."
            )

        da = self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

        da = da.assign_attrs({"cross_section": True, "constant_latitude": lat})

        return da

    def constant_longitude(
        self,
        lon: float,
        inverse_indices: list[str] | set[str] | bool = False,
    ):
        """Extracts a subset of the data array across a line of constant-longitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the subset, in degrees.
            Must be between -180.0 and 180.0
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.UxDataArray
            A subset of the original data array containing only the faces that intersect
            with the line of constant longitude.


        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude or the data variable is not face-centered.

        Examples
        --------
        >>> # Extract data at 0° longitude
        >>> cross_section = uxda.cross_section.constant_longitude(lon=0.0)
        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )

        faces = self.uxda.uxgrid.get_faces_at_constant_longitude(
            lon,
        )

        if len(faces) == 0:
            raise ValueError(
                f"No faces found that intersect a line of constant longitude at {lon} degrees."
            )

        da = self.uxda.isel(n_face=faces, inverse_indices=inverse_indices)

        da = da.assign_attrs({"cross_section": True, "constant_longitude": lon})

        return da

    def constant_latitude_interval(
        self,
        lats: tuple[float, float],
        inverse_indices: list[str] | set[str] | bool = False,
    ):
        """Extracts a subset of data by selecting all faces that
        are within a specified latitude interval.

        Parameters
        ----------
        lats : tuple[float, float]
            The latitude interval (min_lat, max_lat) at which to extract the subset,
            in degrees. Values must be between -90.0 and 90.0
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
            - False: No index storage (default)

        Returns
        -------
        uxarray.UxDataArray
            A subset of the original data array containing only the faces that intersect
            with the line of constant latitude.

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
        lons: tuple[float, float],
        inverse_indices: list[str] | set[str] | bool = False,
    ):
        """Extracts a subset of data by selecting all faces that are within a specified longitude interval.

        Parameters
        ----------
        lons : tuple[float, float]
            The longitude interval (min_lon, max_lon) at which to extract the subset,
            in degrees. Values must be between -180.0 and 180.0
        inverse_indices : list[str] | set[str] | bool, optional
            Controls storage of original grid indices. Options:
            - True: Stores original face indices
            - list/set of strings: Stores specified index types (valid values: "face", "edge", "node")
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
