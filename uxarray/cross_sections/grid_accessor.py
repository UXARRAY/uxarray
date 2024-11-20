from __future__ import annotations


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.grid import Grid


class GridCrossSectionAccessor:
    """Accessor for cross-section operations on a ``Grid``"""

    def __init__(self, uxgrid: Grid) -> None:
        self.uxgrid = uxgrid

    def __repr__(self):
        prefix = "<uxarray.Grid.cross_section>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "  * constant_latitude(lat, return_face_indices)\n"
        methods_heading += "  * constant_longitude(lon, return_face_indices)\n"
        return prefix + methods_heading

    def constant_latitude(
        self,
        lat: float,
        return_face_indices: bool = False,
    ):
        """Extracts a cross-section of the grid by selecting all faces that
        intersect with a specified line of constant latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section, in degrees.
            Must be between -90.0 and 90.0
        return_face_indices : bool, optional
            If True, also returns the indices of the faces that intersect with the line of constant latitude.

        Returns
        -------
        uxarray.Grid
            A subset of the original grid containing only the faces that intersect
            with the specified latitude.

        Raises
        ------
        ValueError
            If no intersections are found at the specified latitude.

        Examples
        --------
        >>> # Extract data at 15.5°S latitude
        >>> cross_section = grid.constant_latitude(lat=-15.5)

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """

        faces = self.uxgrid.get_faces_at_constant_latitude(
            lat,
        )

        if len(faces) == 0:
            raise ValueError(f"No intersections found at lat={lat}.")

        grid_at_constant_lat = self.uxgrid.isel(n_face=faces)

        if return_face_indices:
            return grid_at_constant_lat, faces
        else:
            return grid_at_constant_lat

    def constant_longitude(
        self,
        lon: float,
        return_face_indices: bool = False,
    ):
        """Extracts a cross-section of the grid by selecting all faces that
        intersect with a specified line of constant longitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section, in degrees.
            Must be between -90.0 and 90.0
        return_face_indices : bool, optional
            If True, also returns the indices of the faces that intersect with the line of constant longitude.

        Returns
        -------
        uxarray.Grid
            A subset of the original grid containing only the faces that intersect
            with the specified longitude.

        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude.

        Examples
        --------
        >>> # Extract data at 0° longitude
        >>> cross_section = grid.constant_latitude(lon=0.0)

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        faces = self.uxgrid.get_faces_at_constant_longitude(
            lon,
        )

        if len(faces) == 0:
            raise ValueError(f"No intersections found at lon={lon}")

        grid_at_constant_lon = self.uxgrid.isel(n_face=faces)

        if return_face_indices:
            return grid_at_constant_lon, faces
        else:
            return grid_at_constant_lon

    def gca(self, *args, **kwargs):
        raise NotImplementedError

    def bounded_latitude(self, *args, **kwargs):
        raise NotImplementedError

    def bounded_longitude(self, *args, **kwargs):
        raise NotImplementedError

    def gca_gca(self, *args, **kwargs):
        raise NotImplementedError
