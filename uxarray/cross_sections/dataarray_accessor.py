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

    def constant_latitude(self, lat: float, use_spherical_bounding_box=False):
        """Extracts a cross-section of the data array at a specified constant
        latitude.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section, in degrees.
        use_spherical_bounding_box : bool, optional
            If True, uses a spherical bounding box for intersection calculations.

        Raises
        ------
        ValueError
            If no intersections are found at the specified latitude, a ValueError is raised.

        Examples
        --------
        >>> uxda.constant_latitude_cross_section(lat=-15.5)
        """
        faces = self.uxda.uxgrid.get_faces_at_constant_latitude(
            lat, use_spherical_bounding_box
        )

        return self.uxda.isel(n_face=faces)

    def constant_longitude(self, lon: float, use_spherical_bounding_box=False):
        """Extracts a cross-section of the data array at a specified constant
        longitude.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section, in degrees.
        use_spherical_bounding_box : bool, optional
            If True, uses a spherical bounding box for intersection calculations.

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
        faces = self.uxda.uxgrid.get_faces_at_constant_longitude(
            lon, use_spherical_bounding_box
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
