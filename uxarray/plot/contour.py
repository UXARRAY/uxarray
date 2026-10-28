from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from difflib import get_close_matches
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from uxarray.constants import INT_DTYPE
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
    """Longitudes moved into [-180, 180), the range the lines are drawn in. Grids
    store them in either [0, 360) or [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


@njit(cache=True)
def _fan_corners(
    face_node_connectivity: np.ndarray,
    n_nodes_per_face: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """For each face, the position of a corner that can see every other corner, so that
    a fan of triangles from it covers the face exactly. 0 is tried first; -1 means that
    no corner can. Faces that touch or contain a pole, or have no area, keep 0: a pole is
    one point drawn at an arbitrary longitude, so their shape in longitude and latitude
    says nothing about dents.

    Used by ``_fan_triangles`` to find the few dented faces among all the faces of a
    grid, which is why it is compiled."""
    n_faces = len(n_nodes_per_face)
    corners = np.zeros(n_faces, dtype=np.int64)
    if n_faces == 0:  # e.g. the dual of a grid of two faces; .max() needs a face
        return corners
    # create x and y buffers only once, to avoid allocating many tiny numpy arrays
    max_n_nodes_per_face = n_nodes_per_face.max()
    x = np.empty(max_n_nodes_per_face)
    y = np.empty(max_n_nodes_per_face)
    for face in range(n_faces):
        n = n_nodes_per_face[face]
        if n < 4:
            continue  # all triangles are convex :)
        first = face_node_connectivity[face, 0]
        at_pole = False
        for k in range(n):
            node = face_node_connectivity[face, k]
            dx = lon[node] - lon[first]
            x[k] = dx - 360.0 * np.floor(dx / 360.0 + 0.5)
            y[k] = lat[node]
            at_pole = at_pole or abs(lat[node]) >= 89.999
        if at_pole or x[:n].max() - x[:n].min() >= 180.0:
            continue
        area = 0.0
        for k in range(n):
            area += x[k] * y[(k + 1) % n] - x[(k + 1) % n] * y[k]
        if not np.isfinite(area) or area == 0.0:
            continue
        corners[face] = -1
        for corner in range(n):
            sees_all = True
            for k in range(1, n - 1):
                a, b = (corner + k) % n, (corner + k + 1) % n
                turn = (x[a] - x[corner]) * (y[b] - y[corner]) - (x[b] - x[corner]) * (
                    y[a] - y[corner]
                )
                # a triangle that turns the other way than the face lies outside it
                if turn * area < 0.0 and abs(turn) > 1e-9 * abs(area):
                    sees_all = False
                    break
            if sees_all:
                corners[face] = corner
                break
    return corners


def _ear_clipping(x: np.ndarray, y: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangles of a simple polygon, as positions of its corners.

    Used by ``_fan_triangles`` for the rare dented faces in which no corner can see
    every other corner (a C-shaped face, for example), so that no fan of triangles
    covers them. It cuts off one "ear" at a time: three corners in a row whose triangle
    turns the same way as the polygon and holds no other corner. Plain Python is fast
    enough, as it only runs on those faces."""

    def turn(a, b, c):
        return (x[b] - x[a]) * (y[c] - y[a]) - (x[c] - x[a]) * (y[b] - y[a])

    orientation = np.sign(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    left = list(range(len(x)))
    triangles = []
    while len(left) > 3:
        for k in range(len(left)):
            a, b, c = left[k - 1], left[k], left[(k + 1) % len(left)]
            if turn(a, b, c) * orientation <= 0:
                continue
            others = [m for m in left if m not in (a, b, c)]
            if any(
                turn(a, b, m) * orientation >= 0
                and turn(b, c, m) * orientation >= 0
                and turn(c, a, m) * orientation >= 0
                for m in others
            ):
                continue
            triangles.append((a, b, c))
            del left[k]
            break
        else:
            break
    if len(left) == 3:
        triangles.append(tuple(left))
    return triangles


def _fan_triangles(
    face_node_connectivity: np.ndarray,
    n_nodes_per_face: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Splits each face into triangles that share one of its corners.

    Used by ``_triangulation``: Matplotlib contours data on triangles only."""
    # A fan from one corner covers a face exactly only if that corner can see every other
    # corner. Every convex face is like that, but dented faces occur, mostly among the
    # cells of the dual mesh of irregular meshes (FESOM-style ocean meshes, overlap meshes).
    # A fan from the wrong corner puts triangles outside the face, and the lines there are
    # interpolated from the wrong values. Such faces are fanned from a corner that sees the
    # whole face, or ear-clipped when no corner does. Faces whose first corner works, nearly
    # all of them, are split as before; finding the others is one compiled pass.
    corners = _fan_corners(face_node_connectivity, n_nodes_per_face, lon, lat)
    connectivity = face_node_connectivity
    moved = corners > 0
    if moved.any():
        n = np.maximum(n_nodes_per_face[moved], 1)[:, None]
        columns = (np.arange(connectivity.shape[1]) + corners[moved][:, None]) % n
        connectivity = connectivity.copy()
        connectivity[moved] = np.take_along_axis(connectivity[moved], columns, axis=1)

    fan = corners >= 0
    triangles = []
    for k in range(1, connectivity.shape[1] - 1):
        has_triangle = fan & (n_nodes_per_face > k + 1)
        triangles.append(connectivity[has_triangle][:, [0, k, k + 1]])
    for face in np.flatnonzero(~fan):
        nodes = face_node_connectivity[face, : n_nodes_per_face[face]]
        x = lon[nodes] - 360.0 * np.round((lon[nodes] - lon[nodes[0]]) / 360.0)
        clipped = np.array(_ear_clipping(x, lat[nodes]), dtype=np.int64).reshape(-1, 3)
        triangles.append(nodes[clipped])
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
        source.face_node_connectivity.values, source.n_nodes_per_face.values, lon, lat
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
    [-180, 180].

    A line that crosses the antimeridian is found in longitudes that run past 180
    degrees, so that it stays in one piece; drawn that way, it would leave the map.
    Both methods use it."""
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


def _contour_levels(
    values: np.ndarray, levels: int | Sequence[float], keep_min: bool = False
) -> np.ndarray:
    """Returns the contour levels that fall inside the range of the data.

    An integer requests roughly that many levels at "nice" values, chosen the
    way Matplotlib chooses them. Given levels are sorted, and a level given twice
    is used once. With ``keep_min``, a given level equal to the smallest value is
    kept: the method "edges" draws a line there, around the faces at that value.
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
        levels = np.unique(np.asarray(levels, dtype=float))
        if keep_min:
            return levels[(levels >= vmin) & (levels < vmax)]

    return levels[(levels > vmin) & (levels < vmax)]


def _interpolated_contours(
    values: np.ndarray, uxgrid: Grid, dim: str, levels: np.ndarray
) -> list[tuple[float, np.ndarray]]:
    """Contour lines from linear interpolation on a triangulation of the grid.

    The method ``"interpolated"``. The data values sit at the corners of the triangles
    from ``_triangulation``, and Matplotlib's ``tricontour`` draws each line straight
    across every triangle it passes, where the values change linearly. The lines are
    smooth but do not follow the edges of the grid."""
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


def _join_segments(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joins segments, given by the indices of their two end nodes, into
    continuous lines.

    Used by the method ``"edges"``, which finds each contour line as separate grid
    edges in no particular order. Joined, a line is drawn as one path and gets one
    label, and its length can be measured to place that label.

    Returns
    -------
    nodes : np.ndarray
        The node indices along every line, one line after the other.
    offsets : np.ndarray
        Where each line starts in ``nodes``, followed by the length of ``nodes``.
    """
    if len(start) == 0:
        return np.empty(0, dtype=INT_DTYPE), np.zeros(1, dtype=INT_DTYPE)
    return _join_segments_kernel(
        np.ascontiguousarray(start, dtype=INT_DTYPE),
        np.ascontiguousarray(end, dtype=INT_DTYPE),
    )


@njit(cache=True, nogil=True)
def _join_segments_kernel(start, end):
    """The compiled part of ``_join_segments``: lists the segments at each node,
    then walks from node to node through unused segments. Open lines are walked from
    one of their ends, so that each comes out in one piece."""
    n_segment = start.size

    # The nodes are numbered in the order they first appear, which is the order
    # the lines are started from them
    number = np.full(max(start.max(), end.max()) + 1, -1, dtype=INT_DTYPE)
    node_at = np.empty(2 * n_segment, dtype=INT_DTYPE)
    n_node = 0
    for i in range(n_segment):
        for node in (start[i], end[i]):
            if number[node] < 0:
                number[node] = n_node
                node_at[n_node] = node
                n_node += 1

    # The segments at each node, in the order of the segments, are
    # segments_at[first[k]:first[k + 1]] for the node numbered k
    first = np.zeros(n_node + 1, dtype=INT_DTYPE)
    for i in range(n_segment):
        first[number[start[i]] + 1] += 1
        first[number[end[i]] + 1] += 1
    for k in range(n_node):
        first[k + 1] += first[k]
    segments_at = np.empty(2 * n_segment, dtype=INT_DTYPE)
    filled = first[:-1].copy()
    for i in range(n_segment):
        for node in (start[i], end[i]):
            k = number[node]
            segments_at[filled[k]] = i
            filled[k] += 1

    # Each line adds one node more than its segments, so there are at most twice
    # as many nodes as segments, and at most as many lines as segments
    used = np.zeros(n_segment, dtype=np.bool_)
    unused_from = first[:-1].copy()
    nodes = np.empty(2 * n_segment, dtype=INT_DTYPE)
    offsets = np.zeros(n_segment + 1, dtype=INT_DTYPE)
    n_point, n_line = 0, 0

    # Open lines first, starting from their ends, then closed loops
    for only_line_ends in (True, False):
        for line_start in range(n_node):
            if only_line_ends and (first[line_start + 1] - first[line_start]) % 2 == 0:
                continue
            while _next_unused(line_start, first, segments_at, used, unused_from) >= 0:
                k = line_start
                nodes[n_point] = node_at[k]
                n_point += 1
                while True:
                    i = _next_unused(k, first, segments_at, used, unused_from)
                    if i < 0:
                        break
                    used[i] = True
                    node = end[i] if start[i] == node_at[k] else start[i]
                    k = number[node]
                    nodes[n_point] = node
                    n_point += 1
                n_line += 1
                offsets[n_line] = n_point

    return nodes[:n_point].copy(), offsets[: n_line + 1].copy()


@njit(cache=True, nogil=True, inline="always")
def _next_unused(k, first, segments_at, used, unused_from):
    """The first unused segment at the node numbered k, or -1 if all are used.
    The segments before ``unused_from[k]`` are known to be used."""
    while unused_from[k] < first[k + 1] and used[segments_at[unused_from[k]]]:
        unused_from[k] += 1
    if unused_from[k] == first[k + 1]:
        return -1
    return segments_at[unused_from[k]]


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
        line_nodes, offsets = _join_segments(nodes[:, 0], nodes[:, 1])
        if len(line_nodes) == 0:
            continue

        # The lines are processed together, as one array of points
        starts, ends = offsets[:-1], offsets[1:]
        points = np.column_stack(
            [_unwrap_lines(lon[line_nodes], starts), lat[line_nodes]]
        )

        # Unwrapped, a line that crosses the antimeridian runs past 180 degrees
        crosses = (np.minimum.reduceat(points[:, 0], starts) < -180.0) | (
            np.maximum.reduceat(points[:, 0], starts) > 180.0
        )
        for line, crossing in zip(np.split(points, ends[:-1]), crosses):
            parts = _split_at_antimeridian(line) if crossing else [line]
            contours.extend((float(level), part) for part in parts)
    return contours


def _unwrap_lines(lon: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Removes the jumps of 360 degrees from the longitudes of several lines
    stored one after the other, which begin at ``starts``. Each line is
    unwrapped on its own, as by ``np.unwrap(..., period=360)``.

    Used by the method ``"edges"``: a line along grid edges that crosses the
    antimeridian jumps from 180 to -180 degrees, and is drawn across the whole map
    unless it is unwrapped and then split with ``_split_at_antimeridian``."""
    step = np.diff(lon)
    # The whole turns that bring each step into [-180, 180]
    turns = np.where(np.abs(step) < 180.0, 0.0, np.round(step / 360.0))
    turns[starts[1:] - 1] = 0.0  # no step between the end of a line and the next
    shift = np.concatenate([[0.0], np.cumsum(turns)])
    shift -= np.repeat(shift[starts], np.diff(np.append(starts, len(lon))))
    return lon - 360.0 * shift


def _grid_extents(uxgrid: Grid) -> tuple[float, float, float, float]:
    """The area of the grid as (west, south, east, north), in degrees.

    Used as the extents of the lines, so that contours shown on their own cover the
    same area as ``.plot()`` of the same data, not just the area of the lines."""
    lon, lat = uxgrid.node_lon.values, uxgrid.node_lat.values
    west, east = float(np.nanmin(lon)), float(np.nanmax(lon))
    south, north = float(np.nanmin(lat)), float(np.nanmax(lat))
    if east - west > 350.0:  # all the way around the globe
        west, east = -180.0, 180.0
        # A face around a pole need not have a node there; its corners then go once
        # around the pole. Only the faces at the northernmost and southernmost nodes
        # can be such a face.
        connectivity = uxgrid.face_node_connectivity.values
        n_nodes = uxgrid.n_nodes_per_face.values
        for sign, extreme in ((1.0, north), (-1.0, south)):
            if abs(extreme) >= 90.0 - 1e-9:
                continue  # a node at the pole
            nodes = np.flatnonzero(sign * lat >= sign * extreme - 1e-9)
            for face in np.flatnonzero(np.isin(connectivity, nodes).any(axis=1)):
                corners = lon[connectivity[face, : n_nodes[face]]]
                turns = (
                    np.diff(np.append(corners, corners[0])) + 180.0
                ) % 360.0 - 180.0
                if abs(turns.sum()) > 180.0:
                    north, south = (90.0, south) if sign > 0 else (north, -90.0)
                    break
    # In the single precision of the polygons that .plot() shades, so that the lines never
    # widen the area of a shaded plot they are overlaid on.
    return tuple(float(np.float32(value)) for value in (west, south, east, north))


def _drop_lines_outside_projection(
    contours: list[tuple[float, np.ndarray]], projection
) -> list[tuple[float, np.ndarray]]:
    """Removes the lines that have no extent in a map projection, such as a
    line that lies along the edge of the map. GeoViews cannot project them."""
    _raise_hint_if_optional_deps_missing("cartopy")
    import cartopy.crs as ccrs
    import shapely

    if not contours:
        return contours
    source = ccrs.PlateCarree()

    # A line with a point well inside the map has extent on it. Projecting all the
    # points at once is much faster than projecting each line, so only the other
    # lines, such as those along the edge of the map, are projected as lines.
    points = np.concatenate([line for _, line in contours])
    x, y = projection.transform_points(source, points[:, 0], points[:, 1])[:, :2].T
    finite = np.isfinite(x) & np.isfinite(y)
    domain = projection.domain
    margin = 1e-6 * np.ptp(np.reshape(domain.bounds, (2, 2)), axis=0).max()
    inside = np.zeros(len(points), dtype=bool)
    inside[finite] = shapely.contains_xy(domain.buffer(-margin), x[finite], y[finite])
    starts = np.cumsum([0] + [len(line) for _, line in contours[:-1]])
    on_map = np.logical_or.reduceat(inside, starts)

    return [
        (level, line)
        for (level, line), certain in zip(contours, on_map)
        if certain
        or not projection.project_geometry(shapely.LineString(line), source).is_empty
    ]


def _paths_by_level(
    contours: list[tuple[float, np.ndarray]], level_name: str
) -> list[dict]:
    """Path data for HoloViews, with one path per level whose lines are
    separated by NaN.

    HoloViews and the plotting libraries handle each path separately, which is
    slow for the thousands of lines a noisy field gives, while a NaN is drawn as
    a gap at almost no cost.
    """
    lines_at = defaultdict(list)
    for level, line in contours:
        lines_at[level].append(line)

    gap = np.full((1, 2), np.nan)
    paths = []
    for level, lines in lines_at.items():
        points = np.concatenate([part for line in lines for part in (gap, line)][1:])
        paths.append({"x": points[:, 0], "y": points[:, 1], level_name: level})
    return paths


def _label_fraction(labels: bool | float) -> float:
    """The fraction of the levels to label, from the ``labels`` argument."""
    if isinstance(labels, bool):
        return 1.0 if labels else 0.0
    try:
        fraction = float(labels)
    except (TypeError, ValueError):
        fraction = np.nan
    # NaN fails this test as well
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "labels must be True, False or a fraction of the levels between 0 and 1, "
            f"but got {labels!r}"
        )
    return fraction


def _label_points(
    contours: list[tuple[float, np.ndarray]],
    fraction: float = 1.0,
) -> tuple[list[float], list[float], list[str]]:
    """Positions and text of the labels of contour lines.

    Each level is labeled at the middle of its longest lines: the longest one,
    and up to ``_MAX_LABELS_PER_LEVEL - 1`` more that are not short compared
    with the extent of all the lines. With a ``fraction`` below 1, only that
    share of the levels is labeled, spread evenly from the lowest one.

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

    line_levels = np.array([level for level, _ in contours])
    points = np.concatenate([line for _, line in contours])
    extent = np.ptp(points, axis=0).max()

    # a level is labeled where the running count of labels, fraction per level, steps up:
    # 0.5 labels the lowest level and every second one after it
    levels = np.unique(line_levels)
    count = np.floor(np.arange(-1, len(levels)) * fraction + 1e-9)
    labeled = set(levels[np.diff(count) > 0].tolist())

    # The length of every line, from the steps between the points of all of them
    counts = np.array([len(line) for _, line in contours])
    ends = np.cumsum(counts)
    steps = np.append(np.hypot(*np.diff(points, axis=0).T), 0.0)
    steps[ends - 1] = 0.0  # no step between the end of a line and the next
    lengths = np.add.reduceat(steps, ends - counts)

    # Only the middles of the lines that are labeled are found
    for level in dict.fromkeys(line_levels):
        if level not in labeled:
            continue
        at_level = np.flatnonzero(line_levels == level)
        longest = at_level[np.argsort(-lengths[at_level], kind="stable")]
        for rank, i in enumerate(longest[:_MAX_LABELS_PER_LEVEL]):
            if rank == 0 or lengths[i] >= _MIN_LABELED_LENGTH * extent:
                line = contours[i][1]
                distance = np.concatenate(
                    [[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
                )
                x.append(float(np.interp(distance[-1] / 2, distance, line[:, 0])))
                y.append(float(np.interp(distance[-1] / 2, distance, line[:, 1])))
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
    if method is None:
        method = "edges" if dim == "n_face" else "interpolated"

    # A level at the smallest value separates the faces at that value from the others,
    # but an interpolated line there would only touch the data at single points
    levels = _contour_levels(values, levels, keep_min=method == "edges")

    if method == "edges":
        if dim != "n_face":
            raise DataCenteringError(
                "method='edges' is only supported for face-centered data. "
                "Use method='interpolated' for node-centered data."
            )
        return _edge_contours(values, uxda.uxgrid, levels)

    return _interpolated_contours(values, uxda.uxgrid, dim, levels)
