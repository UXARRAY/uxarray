"""
Interval tree spatial indexing for latitude/longitude bounds.

This module provides efficient spatial indexing for unstructured meshes
using separate interval trees for latitude and longitude components of face bounds.
Handles antimeridian crossings through interval splitting.
"""

from typing import List, Set

import numpy as np
from intervaltree import IntervalTree as _IntervalTree


class IntervalTree:
    """
    Interval tree spatial index for lat/lon face bounds.

    This index provides efficient range queries on face bounds by using
    separate interval trees for latitude and longitude components.
    Antimeridian crossings are handled by splitting crossing intervals
    into two normal intervals.

    Parameters
    ----------
    grid : ux.Grid
        UXarray grid to index

    Attributes
    ----------
    lat_tree : intervaltree.IntervalTree
        Interval tree for latitude bounds
    lon_tree : intervaltree.IntervalTree
        Interval tree for longitude bounds
    crossing_faces : set
        Set of face IDs that cross the antimeridian
    """

    def __init__(self, grid):
        self.grid = grid
        self.lat_tree = _IntervalTree()
        self.lon_tree = _IntervalTree()
        self.crossing_faces = set()
        self._build_index()

    def _build_index(self):
        """Build the interval trees from grid face bounds."""

        # Get face bounds using UXarray's interface
        # Note: face_bounds_lat and face_bounds_lon return values in degrees
        lat_bounds = self.grid.face_bounds_lat.values  # Shape: (n_face, 2)
        lon_bounds = self.grid.face_bounds_lon.values  # Shape: (n_face, 2)

        for face_id in range(self.grid.n_face):
            # Get bounds for this face (already in degrees)
            lat_min, lat_max = lat_bounds[face_id]
            lon_min, lon_max = lon_bounds[face_id]

            # Insert latitude interval (always straightforward)
            if lat_max > lat_min:
                self.lat_tree[lat_min:lat_max] = face_id
            elif lat_max == lat_min:
                # Point interval - add small epsilon
                self.lat_tree[lat_min : (lat_min + 1e-6)] = face_id

            # Handle longitude with antimeridian crossing detection
            if lon_max < lon_min:  # Crosses antimeridian
                self.crossing_faces.add(face_id)

                # Split into two intervals: [lon_min, 180°] and [-180°, lon_max]
                # Work directly with [-180, 180] range
                if lon_min < 180.0:
                    self.lon_tree[lon_min:180.0] = face_id
                if lon_max > -180.0:
                    self.lon_tree[-180.0:lon_max] = face_id
            else:
                # Normal case: insert interval directly
                if lon_max > lon_min:
                    self.lon_tree[lon_min:lon_max] = face_id
                elif lon_max == lon_min:
                    # Point interval - add small epsilon
                    self.lon_tree[lon_min : (lon_min + 1e-6)] = face_id

    def query_point(self, lat: float, lon: float) -> Set[int]:
        """
        Find all faces containing a point.

        Parameters
        ----------
        lat : float
            Latitude in degrees [-90, 90]
        lon : float
            Longitude in degrees [-180, 180]

        Returns
        -------
        Set[int]
            Set of face IDs containing the point
        """
        # Query latitude tree directly
        lat_candidates = {iv.data for iv in self.lat_tree[lat]}

        # Query longitude tree directly
        lon_candidates = {interval.data for interval in self.lon_tree[lon]}

        # Return intersection
        return lat_candidates & lon_candidates

    def query_lat_band(self, lat_min: float, lat_max: float) -> Set[int]:
        """
        Find all faces completely within a latitude band.

        This matches UXarray's get_faces_between_latitudes() semantics:
        faces must be entirely contained within [lat_min, lat_max].
        Perfect for zonal averaging and regional analysis.

        Parameters
        ----------
        lat_min, lat_max : float
            Latitude bounds in degrees [-90, 90]

        Returns
        -------
        Set[int]
            Set of face IDs completely within the latitude band
        """
        # Get all intersecting faces first
        intersecting_faces = {iv.data for iv in self.lat_tree[lat_min:lat_max]}

        # Filter to only those completely within bounds
        within_faces = set()
        face_bounds_lat = self.grid.face_bounds_lat.values

        for face_id in intersecting_faces:
            face_lat_min, face_lat_max = face_bounds_lat[face_id]
            if face_lat_min >= lat_min and face_lat_max <= lat_max:
                within_faces.add(face_id)

        return within_faces

    def query_region(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> Set[int]:
        """
        Find all faces intersecting a rectangular region.

        Parameters
        ----------
        lat_min, lat_max : float
            Latitude bounds in degrees [-90, 90]
        lon_min, lon_max : float
            Longitude bounds in degrees [-180, 180]

        Returns
        -------
        Set[int]
            Set of face IDs intersecting the region
        """
        # Query latitude band
        lat_candidates = {iv.data for iv in self.lat_tree[lat_min:lat_max]}

        # Handle longitude range (may cross antimeridian)
        if lon_max < lon_min:  # Crosses antimeridian
            # Query two ranges: [lon_min, 180°] and [-180°, lon_max]
            # Work directly with [-180, 180] range
            lon_faces = set()
            if lon_min < 180.0:
                lon_faces_1 = {
                    interval.data for interval in self.lon_tree[lon_min:180.0]
                }
                lon_faces.update(lon_faces_1)
            if lon_max > -180.0:
                lon_faces_2 = {
                    interval.data for interval in self.lon_tree[-180.0:lon_max]
                }
                lon_faces.update(lon_faces_2)
        else:
            # Normal case: query directly
            lon_faces = {interval.data for interval in self.lon_tree[lon_min:lon_max]}

        return lat_candidates & lon_faces

    def query_faces_containing_points(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> List[Set[int]]:
        """
        Find faces containing multiple points efficiently.

        Parameters
        ----------
        lats : array-like
            Latitudes in degrees [-90, 90]
        lons : array-like
            Longitudes in degrees [-180, 180]

        Returns
        -------
        List[Set[int]]
            List of sets, each containing face IDs for corresponding point
        """
        results = []
        for lat, lon in zip(lats, lons):
            results.append(self.query_point(lat, lon))
        return results

    @property
    def n_crossing_faces(self) -> int:
        """Number of faces that cross the antimeridian."""
        return len(self.crossing_faces)

    @property
    def crossing_percentage(self) -> float:
        """Percentage of faces that cross the antimeridian."""
        return 100.0 * len(self.crossing_faces) / self.grid.n_face

    def __repr__(self) -> str:
        return (
            f"IntervalTree(n_faces={self.grid.n_face}, "
            f"crossing_faces={len(self.crossing_faces)} "
            f"({self.crossing_percentage:.1f}%))"
        )


def build_interval_tree(grid) -> IntervalTree:
    """
    Build an interval tree index for the given grid.

    Parameters
    ----------
    grid : ux.Grid
        UXarray grid to index

    Returns
    -------
    IntervalTree
        Spatial index for efficient range queries on face bounds
    """
    return IntervalTree(grid)
