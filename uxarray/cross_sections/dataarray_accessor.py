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

        methods_heading += "  * constant_latitude(center_coord, k, element, **kwargs)\n"

        return prefix + methods_heading

    def constant_latitude(self, lat: float, method="fast"):
        """Extracts a cross-section of the data array at a specified constant
        latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section, in degrees.
        method : str, optional
            The internal method to use when identifying faces at the constant latitude.
            Options are:
            - 'fast': Uses a faster but potentially less accurate method for face identification.
            - 'accurate': Uses a slower but more accurate method.
            Default is 'fast'.

        Raises
        ------
        ValueError
            If no intersections are found at the specified latitude, a ValueError is raised.

        Examples
        --------
        >>> uxda.constant_latitude_cross_section(lat=-15.5)

        Notes
        -----
        The accuracy and performance of the function can be controlled using the `method` parameter.
        For higher precision requreiments, consider using method='acurate'.
        """
        faces = self.uxda.uxgrid.get_faces_at_constant_latitude(lat, method)

        return self.uxda.isel(n_face=faces)

    def constant_longitude(self, lon: float, method="fast"):
        """Extracts a cross-section of the data array at a specified constant
        longitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section, in degrees.
        method : str, optional
            The internal method to use when identifying faces at the constant longitude.
            Options are:
            - 'fast': Uses a faster but potentially less accurate method for face identification.
            - 'accurate': Uses a slower but more accurate method.
            Default is 'fast'.

        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude, a ValueError is raised.

        Examples
        --------
        >>> uxda.constant_longitude_cross_section(lon=-15.5)

        Notes
        -----
        The accuracy and performance of the function can be controlled using the `method` parameter.
        For higher precision requreiments, consider using method='acurate'.
        """
        faces = self.uxda.uxgrid.get_faces_at_constant_longitude(lon, method)

        return self.uxda.isel(n_face=faces)

    def gca(self, *args, **kwargs):
        raise NotImplementedError

    def bounded_latitude(self, *args, **kwargs):
        raise NotImplementedError

    def bounded_longitude(self, *args, **kwargs):
        raise NotImplementedError

    def gca_gca(self, *args, **kwargs):
        raise NotImplementedError
