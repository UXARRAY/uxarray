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

        methods_heading += "  * constant_latitude(lat, )\n"
        return prefix + methods_heading

    def constant_latitude(self, lat: float, return_face_indices=False, method="fast"):
        """Extracts a cross-section of the grid at a specified constant
        latitude.

        This method identifies and returns all faces (or grid elements) that intersect
        with a given latitude. The returned cross-section can include either just the grid
        or both the grid elements and the corresponding face indices, depending
        on the `return_face_indices` parameter.

        Parameters
        ----------
        lat : float
            The latitude at which to extract the cross-section, in degrees.
        return_face_indices : bool, optional
            If True, returns both the grid at the specified latitude and the indices
            of the intersecting faces. If False, only the grid is returned.
            Default is False.
        method : str, optional
            The internal method to use when identifying faces at the constant latitude.
            Options are:
            - 'fast': Uses a faster but potentially less accurate method for face identification.
            - 'accurate': Uses a slower but more accurate method.
            Default is 'fast'.

        Returns
        -------
        grid_at_constant_lat : Grid
            The grid with faces that interesected at a given lattitude
        faces : array, optional
            The indices of the faces that intersect with the specified latitude. This is only
            returned if `return_face_indices` is set to True.

        Raises
        ------
        ValueError
            If no intersections are found at the specified latitude, a ValueError is raised.

        Examples
        --------
        >>> grid, indices = grid.cross_section.constant_latitude(lat=30.0, return_face_indices=True)
        >>> grid = grid.cross_section.constant_latitude(lat=-15.5)

        Notes
        -----
        The accuracy and performance of the function can be controlled using the `method` parameter.
        For higher precision requreiments, consider using method='acurate'.
        """
        faces = self.uxgrid.get_faces_at_constant_latitude(lat, method)

        if len(faces) == 0:
            raise ValueError(f"No intersections found at lat={lat}.")

        grid_at_constant_lat = self.uxgrid.isel(n_face=faces)

        if return_face_indices:
            return grid_at_constant_lat, faces
        else:
            return grid_at_constant_lat

    def constant_longitude(self, lon: float, return_face_indices=False, method="fast"):
        """Extracts a cross-section of the grid at a specified constant
        longitude.

        This method identifies and returns all faces (or grid elements) that intersect
        with a given longitude. The returned cross-section can include either just the grid
        or both the grid elements and the corresponding face indices, depending
        on the `return_face_indices` parameter.

        Parameters
        ----------
        lon : float
            The longitude at which to extract the cross-section, in degrees.
        return_face_indices : bool, optional
            If True, returns both the grid at the specified longitude and the indices
            of the intersecting faces. If False, only the grid is returned.
            Default is False.
        method : str, optional
            The internal method to use when identifying faces at the constant longitude.
            Options are:
            - 'fast': Uses a faster but potentially less accurate method for face identification.
            - 'accurate': Uses a slower but more accurate method.
            Default is 'fast'.

        Returns
        -------
        grid_at_constant_lon : Grid
            The grid with faces that interesected at a given longitudes
        faces : array, optional
            The indices of the faces that intersect with the specified longitude. This is only
            returned if `return_face_indices` is set to True.

        Raises
        ------
        ValueError
            If no intersections are found at the specified longitude, a ValueError is raised.

        Examples
        --------
        >>> grid, indices = grid.cross_section.constant_longitude(lat=0.0, return_face_indices=True)
        >>> grid = grid.cross_section.constant_longitude(lat=20.0)

        Notes
        -----
        The accuracy and performance of the function can be controlled using the `method` parameter.
        For higher precision requreiments, consider using method='acurate'.
        """
        faces = self.uxgrid.get_faces_at_constant_longitude(lon, method)

        if len(faces) == 0:
            raise ValueError(f"No intersections found at lon={lon}.")

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
