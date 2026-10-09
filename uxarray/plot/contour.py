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
from difflib import get_close_matches
from typing import TYPE_CHECKING, Sequence

import numpy as np

from uxarray.errors import DataCenteringError, DimensionError
from uxarray.utils.imports import _raise_hint_if_optional_deps_missing

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.grid import Grid

# Triangles wider than this in longitude wrap around the antimeridian
_MAX_LON_SPAN = 180.0

# Each level gets at most this many labels. Apart from its longest line, a line is
# labeled only if it is at least this long, as a fraction of the extent of all the lines
_MAX_LABELS_PER_LEVEL = 3
_MIN_LABELED_LENGTH = 0.1


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


def _triangulation(
    uxgrid: Grid, dim: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triangles whose corners are the locations of the data.

    For node-centered data the faces of the grid are split into triangles. For
    face-centered data the faces of the dual mesh are used instead, since its
    nodes are the face centers of the grid.

    Returns
    -------
    lon, lat : np.ndarray
        Coordinates of the triangle corners, in degrees. Triangles that cross
        the antimeridian use copies of their western corners, moved 360 degrees
        east, so longitudes can be greater than 180.
    triangles : np.ndarray
        Indices into ``lon`` and ``lat``, with shape ``(n_triangle, 3)``.
        Triangles that contain a pole are not included.
    index : np.ndarray
        Index of the data value at each corner.
    """
    source = uxgrid.get_dual() if dim == "n_face" else uxgrid

    lon = _wrap_longitude(source.node_lon.values)
    lat = source.node_lat.values
    triangles = _fan_triangles(
        source.face_node_connectivity.values, source.n_nodes_per_face.values
    )
    index = np.arange(len(lon))

    crossing = np.ptp(lon[triangles], axis=1) >= _MAX_LON_SPAN
    if crossing.any():
        corners = triangles[crossing]
        west = np.unique(corners[lon[corners] < 0])
        copy_of = np.full(len(lon), -1, dtype=triangles.dtype)
        copy_of[west] = len(lon) + np.arange(len(west))
        triangles[crossing] = np.where(lon[corners] < 0, copy_of[corners], corners)
        lon = np.concatenate([lon, lon[west] + 360.0])
        lat = np.concatenate([lat, lat[west]])
        index = np.concatenate([index, west])

    # A face that lists a node twice gives a triangle with a repeated corner.
    # Matplotlib's contouring does not return when it is given one.
    first, second, third = triangles.T
    usable = (first != second) & (second != third) & (first != third)
    usable &= np.ptp(lon[triangles], axis=1) < _MAX_LON_SPAN
    usable &= np.isfinite(lon[triangles]).all(axis=1)
    usable &= np.isfinite(lat[triangles]).all(axis=1)

    return lon, lat, triangles[usable], index


def _split_at_antimeridian(line: np.ndarray) -> list[np.ndarray]:
    """Splits a line whose longitudes run past -180 or 180 degrees into the
    parts on either side of the antimeridian, each with longitudes in
    [-180, 180]."""
    lon = line[:, 0]
    if lon.min() >= -180.0 and lon.max() <= 180.0:
        return [line]

    import shapely

    parts = []
    first_turn = int(np.floor((lon.min() + 180.0) / 360.0))
    last_turn = int(np.floor((lon.max() + 180.0) / 360.0))
    for turn in range(first_turn, last_turn + 1):
        moved = shapely.LineString(line - [360.0 * turn, 0.0])
        for part in shapely.get_parts(shapely.clip_by_rect(moved, -180, -91, 180, 91)):
            if isinstance(part, shapely.LineString) and len(part.coords) > 1:
                parts.append(np.asarray(part.coords))
    return parts


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
        _raise_hint_if_optional_deps_missing("matplotlib")
        from matplotlib.ticker import MaxNLocator

        levels = MaxNLocator(levels + 1, min_n_ticks=1).tick_values(vmin, vmax)
    else:
        if np.ndim(levels) != 1:
            raise ValueError(
                "levels must be an integer or a one-dimensional sequence of values, "
                f"but got {levels!r}"
            )
        levels = np.sort(np.asarray(levels, dtype=float))

    return levels[(levels > vmin) & (levels < vmax)]


def _interpolated_contours(
    values: np.ndarray, uxgrid: Grid, dim: str, levels: np.ndarray
) -> list[tuple[float, np.ndarray]]:
    """Contour lines from linear interpolation on a triangulation of the grid."""
    _raise_hint_if_optional_deps_missing("matplotlib")
    from matplotlib.figure import Figure
    from matplotlib.tri import Triangulation

    lon, lat, triangles, index = _triangulation(uxgrid, dim)
    values = values[index]

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
        (float(level), part)
        for level, lines in zip(contour_set.levels, contour_set.allsegs)
        for line in lines
        if len(line) > 1
        for part in _split_at_antimeridian(line)
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

    # Edges with a face on both sides, both with data
    has_two_faces = (edge_faces >= 0).all(axis=1)
    first, second = edge_faces[:, 0].clip(0), edge_faces[:, 1].clip(0)
    finite = np.isfinite(values)
    usable = has_two_faces & finite[first] & finite[second]

    contours = []
    for level in levels:
        above = values > level
        on_contour = usable & (above[first] != above[second])
        nodes = edge_nodes[on_contour]
        for line in _join_segments(nodes[:, 0], nodes[:, 1]):
            # Unwrapped, a line that crosses the antimeridian runs past 180 degrees
            points = np.column_stack([np.unwrap(lon[line], period=360.0), lat[line]])
            for part in _split_at_antimeridian(points):
                contours.append((float(level), part))
    return contours


def _drop_lines_outside_projection(
    contours: list[tuple[float, np.ndarray]], projection
) -> list[tuple[float, np.ndarray]]:
    """Removes the lines that have no extent in a map projection, such as a
    line that lies along the edge of the map. GeoViews cannot project them."""
    _raise_hint_if_optional_deps_missing("cartopy")
    import cartopy.crs as ccrs
    import shapely

    source = ccrs.PlateCarree()
    return [
        (level, line)
        for level, line in contours
        if not projection.project_geometry(shapely.LineString(line), source).is_empty
    ]


def _label_points(
    contours: list[tuple[float, np.ndarray]],
) -> tuple[list[float], list[float], list[str]]:
    """Positions and text of the labels of contour lines.

    Each level is labeled at the middle of its longest lines: the longest one,
    and up to ``_MAX_LABELS_PER_LEVEL - 1`` more that are not short compared
    with the extent of all the lines.

    Returns
    -------
    x, y : list of float
        Longitude and latitude of each label, in degrees.
    text : list of str
        The level of the line each label is on.
    """
    x, y, text = [], [], []
    if not contours:
        return x, y, text

    extent = np.ptp(np.concatenate([line for _, line in contours]), axis=0).max()
    middles = defaultdict(list)
    for level, line in contours:
        steps = np.hypot(*np.diff(line, axis=0).T)
        distance = np.concatenate([[0.0], np.cumsum(steps)])
        middle = [np.interp(distance[-1] / 2, distance, line[:, i]) for i in (0, 1)]
        middles[level].append((distance[-1], *middle))

    for level, found in middles.items():
        found.sort(reverse=True)
        for rank, (length, lon, lat) in enumerate(found[:_MAX_LABELS_PER_LEVEL]):
            if rank == 0 or length >= _MIN_LABELED_LENGTH * extent:
                x.append(float(lon))
                y.append(float(lat))
                text.append(f"{level:g}")

    return x, y, text


def _apply_options(element, options: dict):
    """Applies plot options to an element.

    As in hvPlot, which the other plot methods go through, the options are given
    with their Bokeh names and are translated for the backend that is active.
    """
    _raise_hint_if_optional_deps_missing("holoviews", "hvplot")
    import holoviews as hv
    from hvplot.backend_transforms import _transfer_opts_cur_backend

    allowed = set()
    groups = hv.Store.options(backend="bokeh")[type(element).__name__].groups
    for group in groups.values():
        allowed.update(group.allowed_keywords)

    unknown = sorted(set(options) - allowed)
    if unknown:
        similar = sorted(
            {match for name in unknown for match in get_close_matches(name, allowed)}
        )
        raise ValueError(
            f"Unsupported option(s) for plot.contour(): {unknown}. Options are given "
            "with their Bokeh names on both backends"
            + (f". Similar options: {similar}" if similar else "")
        )

    return _transfer_opts_cur_backend(element.opts(backend="bokeh", **options))


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
