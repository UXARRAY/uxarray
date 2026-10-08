"""Contour lines for data on an unstructured grid, without regridding.

Two methods are provided:

- ``"interpolated"``: the data is treated as values at the corners of a
  triangulation of the grid (the grid's own faces for node-centered data, its
  dual mesh for face-centered data) and contoured with linear interpolation
  inside each triangle.
- ``"edges"``: a contour is made of the grid edges that separate faces above a
  level from faces at or below it. No interpolation is done.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Sequence

import numpy as np

from uxarray.errors import DataCenteringError, DimensionError
from uxarray.utils.imports import _raise_hint_if_optional_deps_missing

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.grid import Grid

# Triangles or edges wider than this in longitude wrap around the antimeridian
_MAX_LON_SPAN = 180.0


def _spatial_values(uxda: UxDataArray) -> tuple[np.ndarray, str]:
    """Returns the data as a 1-D array over the faces or nodes of the grid,
    together with the name of that dimension."""
    non_trivial_dims = [dim for dim, size in zip(uxda.dims, uxda.shape) if size != 1]
    if len(non_trivial_dims) != 1:
        raise DimensionError(
            "Expected data with a single dimension (other axes may be length 1), "
            f"but got dims {uxda.dims} with shape {uxda.shape}"
        )
    dim = non_trivial_dims[0]
    if dim not in ("n_face", "n_node"):
        raise DataCenteringError(
            "Contours are only supported for face-centered or node-centered data, "
            f"but got dimension '{dim}'"
        )
    return np.asarray(uxda.squeeze().values, dtype=float), dim


def _wrap_longitude(lon: np.ndarray) -> np.ndarray:
    return (lon + 180.0) % 360.0 - 180.0


def _fan_triangles(
    face_node_connectivity: np.ndarray, n_nodes_per_face: np.ndarray
) -> np.ndarray:
    """Splits each face into triangles that share the face's first node."""
    triangles = []
    for k in range(1, face_node_connectivity.shape[1] - 1):
        has_triangle = n_nodes_per_face > k + 1
        triangles.append(
            face_node_connectivity[has_triangle][:, [0, k, k + 1]],
        )
    if not triangles:
        return np.empty((0, 3), dtype=face_node_connectivity.dtype)
    return np.concatenate(triangles)


def _triangulation(uxgrid: Grid, dim: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangles whose corners are the locations of the data.

    For node-centered data the faces of the grid are split into triangles. For
    face-centered data the faces of the dual mesh are used instead, since its
    nodes are the face centers of the grid.

    Returns
    -------
    lon, lat : np.ndarray
        Coordinates of the triangle corners, in degrees.
    triangles : np.ndarray
        Indices into ``lon`` and ``lat``, with shape ``(n_triangle, 3)``.
        Triangles that cross the antimeridian are not included.
    """
    source = uxgrid.get_dual() if dim == "n_face" else uxgrid

    lon = _wrap_longitude(source.node_lon.values)
    lat = source.node_lat.values
    triangles = _fan_triangles(
        source.face_node_connectivity.values, source.n_nodes_per_face.values
    )

    usable = np.ptp(lon[triangles], axis=1) < _MAX_LON_SPAN
    usable &= np.isfinite(lon[triangles]).all(axis=1)
    usable &= np.isfinite(lat[triangles]).all(axis=1)

    return lon, lat, triangles[usable]


def _contour_levels(values: np.ndarray, levels: int | Sequence[float]) -> np.ndarray:
    """Returns the contour levels that fall inside the range of the data.

    An integer requests roughly that many levels at "nice" values, chosen the
    way Matplotlib chooses them.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.empty(0)
    vmin, vmax = finite.min(), finite.max()

    if isinstance(levels, (int, np.integer)):
        if levels < 1:
            raise ValueError(f"levels must be at least 1, but got {levels}")
        from matplotlib.ticker import MaxNLocator

        levels = MaxNLocator(levels + 1, min_n_ticks=1).tick_values(vmin, vmax)
    else:
        levels = np.sort(np.asarray(levels, dtype=float))

    return levels[(levels > vmin) & (levels < vmax)]


def _interpolated_contours(
    values: np.ndarray, uxgrid: Grid, dim: str, levels: np.ndarray
) -> list[tuple[float, np.ndarray]]:
    """Contour lines from linear interpolation on a triangulation of the grid."""
    from matplotlib.figure import Figure
    from matplotlib.tri import Triangulation

    lon, lat, triangles = _triangulation(uxgrid, dim)

    # Triangles with a missing value at a corner cannot be contoured
    finite = np.isfinite(values)
    triangles = triangles[finite[triangles].all(axis=1)]
    if len(triangles) == 0 or len(levels) == 0:
        return []

    # The contours are computed on an axes that is never displayed
    ax = Figure().add_subplot()
    contour_set = ax.tricontour(
        Triangulation(lon, lat, triangles), np.where(finite, values, 0.0), levels=levels
    )

    return [
        (float(level), line)
        for level, lines in zip(contour_set.levels, contour_set.allsegs)
        for line in lines
        if len(line) > 1
    ]


def _join_segments(start: np.ndarray, end: np.ndarray) -> list[list[int]]:
    """Joins segments, given by the indices of their two end nodes, into
    continuous lines. Returns the node indices along each line."""
    start, end = start.tolist(), end.tolist()
    segments_at = defaultdict(list)
    for i, (a, b) in enumerate(zip(start, end)):
        segments_at[a].append(i)
        segments_at[b].append(i)

    used = [False] * len(start)

    def walk(node):
        line = [node]
        while True:
            unused = [i for i in segments_at[node] if not used[i]]
            if not unused:
                return line
            used[unused[0]] = True
            node = end[unused[0]] if start[unused[0]] == node else start[unused[0]]
            line.append(node)

    lines = []
    # Open lines first, starting from their ends, then closed loops
    for only_line_ends in (True, False):
        for node, segments in segments_at.items():
            if only_line_ends and len(segments) % 2 == 0:
                continue
            while any(not used[i] for i in segments):
                lines.append(walk(node))
    return lines


def _edge_contours(
    values: np.ndarray, uxgrid: Grid, levels: np.ndarray
) -> list[tuple[float, np.ndarray]]:
    """Contour lines made of the grid edges between faces on either side of a
    level."""
    edge_faces = uxgrid.edge_face_connectivity.values
    edge_nodes = uxgrid.edge_node_connectivity.values
    lon = _wrap_longitude(uxgrid.node_lon.values)
    lat = uxgrid.node_lat.values

    # Edges with a face on both sides, both with data, that do not wrap around
    has_two_faces = (edge_faces >= 0).all(axis=1)
    first, second = edge_faces[:, 0].clip(0), edge_faces[:, 1].clip(0)
    finite = np.isfinite(values)
    usable = has_two_faces & finite[first] & finite[second]
    usable &= np.abs(lon[edge_nodes[:, 0]] - lon[edge_nodes[:, 1]]) < _MAX_LON_SPAN

    contours = []
    for level in levels:
        above = values > level
        on_contour = usable & (above[first] != above[second])
        nodes = edge_nodes[on_contour]
        for line in _join_segments(nodes[:, 0], nodes[:, 1]):
            contours.append((float(level), np.column_stack([lon[line], lat[line]])))
    return contours


def _drop_lines_outside_projection(
    contours: list[tuple[float, np.ndarray]], projection
) -> list[tuple[float, np.ndarray]]:
    """Removes the lines that have no extent in a map projection, such as a
    line that lies along the edge of the map. GeoViews cannot project them."""
    import cartopy.crs as ccrs
    import shapely

    source = ccrs.PlateCarree()
    return [
        (level, line)
        for level, line in contours
        if not projection.project_geometry(shapely.LineString(line), source).is_empty
    ]


def _compute_contours(
    uxda: UxDataArray, levels: int | Sequence[float], method: str | None = None
) -> list[tuple[float, np.ndarray]]:
    """Computes contour lines of a data variable on its unstructured grid.

    If no method is given, "edges" is used for face-centered data and
    "interpolated" for node-centered data.

    Returns
    -------
    contours : list of (level, line)
        One entry per continuous line. ``line`` has shape ``(n_point, 2)`` and
        holds longitude and latitude in degrees.
    """
    if method not in ("edges", "interpolated", None):
        raise ValueError(
            f"Unsupported method. Expected one of ['edges', 'interpolated'], but received '{method}'"
        )
    _raise_hint_if_optional_deps_missing("matplotlib")

    values, dim = _spatial_values(uxda)
    levels = _contour_levels(values, levels)

    if method is None:
        method = "edges" if dim == "n_face" else "interpolated"

    if method == "edges":
        if dim != "n_face":
            raise DataCenteringError(
                "method='edges' is only supported for face-centered data. "
                "Use method='interpolated' for node-centered data."
            )
        return _edge_contours(values, uxda.uxgrid, levels)

    return _interpolated_contours(values, uxda.uxgrid, dim, levels)
