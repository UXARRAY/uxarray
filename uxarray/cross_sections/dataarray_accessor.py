from __future__ import annotations


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class UxDataArrayCrossSectionAccessor:
    """Accessor for cross-section operations on a ``UxDataArray``"""

    def __init__(self, uxda) -> None:
        self.uxda = uxda

    def __repr__(self):
        prefix = "<uxarray.UxDataArray.cross_section>\n"
        methods_heading = "Supported Methods:\n"

        methods_heading += "  * constant_latitude(lat)\n"
        methods_heading += "  * constant_longitude(lon)\n"

        return prefix + methods_heading

    def constant_latitude(self, lat: float):
        """Extracts a cross-section of the data array by selecting all faces that
        intersect with a specified line of constant latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section, in degrees.
            Must be between -90.0 and 90.0

        Returns
        -------
        uxarray.UxDataArray
            A subset of the original data array containing only the faces that intersect
            with the specified latitude.

        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude or the data variable is not face-centered.

        Examples
        --------
        >>> # Extract data at 15.5°S latitude
        >>> cross_section = uxda.constant_latitude(lat=-15.5)

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )

        faces = self.uxda.uxgrid.get_faces_at_constant_latitude(lat)

        return self.uxda.isel(n_face=faces)

    def constant_longitude(self, lon: float):
        """Extracts a cross-section of the data array by selecting all faces that
        intersect with a specified line of constant longitude.

        Parameters
        ----------
        lon : float
            The latitude at which to extract the cross-section, in degrees.
            Must be between -180.0 and 180.0

        Returns
        -------
        uxarray.UxDataArray
            A subset of the original data array containing only the faces that intersect
            with the specified longitude.

        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude or the data variable is not face-centered.

        Examples
        --------
        >>> # Extract data at 0° longitude
        >>> cross_section = uxda.constant_latitude(lon=0.0)

        Notes
        -----
        The initial execution time may be significantly longer than subsequent runs
        due to Numba's just-in-time compilation. Subsequent calls will be faster due to caching.
        """
        if not self.uxda._face_centered():
            raise ValueError(
                "Cross sections are only supported for face-centered data variables."
            )

        faces = self.uxda.uxgrid.get_faces_at_constant_longitude(
            lon,
        )

        return self.uxda.isel(n_face=faces)

    def gca(self, *args, **kwargs):
        raise NotImplementedError

    def bounded_latitude(self, *args, **kwargs):
        raise NotImplementedError

    def bounded_longitude(self, *args, **kwargs):
        raise NotImplementedError

    def gca_gca(self, *args, **kwargs):
        raise NotImplementedError
