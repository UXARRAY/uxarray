from warnings import warn

import numpy as np
import polars as pl

from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz


def _check_connectivity(grid):
    """Check if all nodes are referenced by at least one element.

    Node indices that are coincident duplicates of a node that *is*
    referenced are expected to be unreferenced -- connectivity is
    canonicalized to point at a single index per coincident group, while
    the duplicate coordinates themselves are left in place (see
    ``_find_duplicate_nodes``).
    """

    # Convert face_node_connectivity to a Polars Series and get unique values
    nodes_in_conn = pl.Series(grid.face_node_connectivity.values.flatten()).unique()

    # Filter out negative values
    nodes_in_conn = nodes_in_conn.filter(nodes_in_conn >= 0)

    n_duplicate_nodes = len(_find_duplicate_nodes(grid))

    # Check if the size of unique nodes in connectivity is equal to the number of
    # non-duplicate nodes
    if len(nodes_in_conn) == grid.n_node - n_duplicate_nodes:
        return True
    else:
        warn(
            f"Some nodes may not be referenced by any element. {len(nodes_in_conn)} and {grid.n_node - n_duplicate_nodes}",
            RuntimeWarning,
        )
        return False


def _check_duplicate_nodes(grid):
    """Check if there are duplicate nodes in the mesh."""

    # Convert grid to Polars DataFrame
    df = pl.DataFrame({"lon": grid.node_lon.values, "lat": grid.node_lat.values})

    # Find unique nodes based on 'lon' and 'lat'
    unique_df = df.unique(subset=["lon", "lat"], maintain_order=True)

    # Find duplicate nodes using an anti-join
    duplicate_df = df.join(unique_df, on=["lon", "lat"], how="anti")

    # Print duplicate nodes
    if not duplicate_df.is_empty():
        warn(
            f"Duplicate nodes found in the mesh. {duplicate_df.shape[0]} nodes are duplicates.",
            RuntimeWarning,
        )
    else:
        return True


def _check_duplicate_nodes_indices(grid):
    """Check if any face still references a duplicate node index, returns True if
    it does."""

    duplicate_node_map = _find_duplicate_nodes(grid)
    if not duplicate_node_map:
        return False

    duplicate_indices = np.fromiter(
        duplicate_node_map.keys(), dtype=INT_DTYPE, count=len(duplicate_node_map)
    )
    return bool(np.isin(grid.face_node_connectivity.values, duplicate_indices).any())


def _check_area(grid):
    """Check if each face area is greater than our constant ERROR_TOLERANCE."""
    areas = grid.face_areas
    # Check if area of any face is close to zero
    if np.any(np.isclose(areas, 0, atol=ERROR_TOLERANCE)):
        warn(
            "At least one face area is close to zero. Mesh may contain inverted elements",
            RuntimeWarning,
        )
        return False
    else:
        return True


def _coincident_node_canonical_indices(
    points_xyz=None, tolerance=ERROR_TOLERANCE, *, lonlat_rad=None, finite=None
):
    """For each point, find the lowest-indexed point within ``tolerance`` chordal
    distance on the unit sphere (a point with no coincident neighbor maps to itself).

    ``tolerance`` is a chord length on the unit sphere, not an angle in degrees.
    This differs from ``_read_structured_grid``, whose user-facing ``tol`` is in
    degrees and is converted to a chord via ``2*sin(tol/2)`` before the KDTree
    query; here the default ``ERROR_TOLERANCE`` is already that chord radius.

    At the geographic poles, only exactly equal XYZ rows are merged. Distinct
    floating representations remain separate because they can encode a face's
    locally meaningful pole longitude.

    Parameters
    ----------
    points_xyz : np.ndarray, optional
        ``(n, 3)`` unit vectors. Pass this or ``lonlat_rad``.
    tolerance : float
        Chord radius on the unit sphere.
    lonlat_rad : tuple of np.ndarray, optional
        ``(lon, lat)`` in radians. The search only needs x for every point and
        |z| for the pole test, so given lon/lat it builds full XYZ rows just for
        the candidates and poles, never the ``(n, 3)`` array.
    finite : np.ndarray, optional
        With ``lonlat_rad`` only: mask of points to search. The others map to
        themselves. ``None`` requires every point to be finite.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import KDTree

    if (points_xyz is None) == (lonlat_rad is None):
        raise TypeError("Pass exactly one of points_xyz or lonlat_rad.")

    if points_xyz is not None:
        if not np.isfinite(points_xyz).all():
            raise ValueError("Node XYZ coordinates must contain only finite values.")
        norms = np.hypot(np.hypot(points_xyz[:, 0], points_xyz[:, 1]), points_xyz[:, 2])
        if np.any(norms == 0.0):
            raise ValueError("Node XYZ coordinates must be non-zero vectors.")
        del norms
        n_points = len(points_xyz)
        abs_z = np.abs(points_xyz[:, 2])

        def xyz_at(indices):
            return points_xyz[indices]
    else:
        lon_rad, lat_rad = lonlat_rad
        if finite is None and not (
            np.isfinite(lon_rad).all() and np.isfinite(lat_rad).all()
        ):
            raise ValueError("Node XYZ coordinates must contain only finite values.")
        n_points = len(lon_rad)
        abs_z = np.abs(np.sin(lat_rad))

        def xyz_at(indices):
            return np.column_stack(
                _lonlat_rad_to_xyz(lon_rad[indices], lat_rad[indices])
            )

    # ``tolerance`` is a chord radius (see the ``query_pairs`` call below), so it
    # cannot be used directly as a deviation of |z| from 1. For a point at
    # colatitude t from the pole, 1 - |z| = 1 - cos(t) = 2*sin(t/2)**2 = chord**2/2.
    pole_mask = np.isclose(abs_z, 1.0, rtol=0.0, atol=tolerance**2 / 2)
    del abs_z
    # Poles and non-finite points stay out of the tolerance search.
    if finite is None:
        excluded = pole_mask
    else:
        pole_mask &= finite
        excluded = pole_mask | ~finite
    pole_indices = np.flatnonzero(pole_mask)
    del pole_mask
    if not excluded.any():
        excluded = None

    # x as in ``_lonlat_rad_to_xyz``, computed alone so y and z are never built
    # for every point.
    x = (
        points_xyz[:, 0]
        if points_xyz is not None
        else np.cos(lon_rad) * np.cos(lat_rad)
    )

    # Prescreen on x before paying for a KDTree. Two points within a chord of
    # ``tolerance`` differ by at most ``tolerance`` in x, so in x-sorted order every
    # consecutive gap between them is also at most ``tolerance``. A point whose
    # sorted neighbours are both further than that in x therefore cannot be
    # coincident with anything and is dropped. This is exact -- no candidate pair is
    # lost -- and on the common case of a grid with no coincident nodes it replaces
    # the tree build entirely with one sort.
    order = np.argsort(x, kind="stable")
    if excluded is not None:
        # Stable sort then filter keeps the same relative order as sorting only
        # the included points.
        order = order[~excluded[order]]
    del excluded
    sorted_x = x[order]
    del x
    gap_is_small = np.diff(sorted_x) <= tolerance
    del sorted_x
    is_candidate = np.zeros(len(order), dtype=bool)
    is_candidate[:-1] |= gap_is_small
    is_candidate[1:] |= gap_is_small
    del gap_is_small
    # kept ascending so that the first member of a connected component below is
    # still the lowest-numbered node in it
    candidates = np.sort(order[is_candidate])
    del order, is_candidate

    canonical = np.arange(n_points, dtype=INT_DTYPE)
    if len(pole_indices) > 1:
        _, first_indices, inverse = np.unique(
            xyz_at(pole_indices), axis=0, return_index=True, return_inverse=True
        )
        canonical[pole_indices] = pole_indices[first_indices[inverse.ravel()]]

    if len(candidates) < 2:
        return canonical

    tree = KDTree(xyz_at(candidates))
    pairs = tree.query_pairs(r=tolerance, output_type="ndarray")

    if len(pairs) == 0:
        return canonical

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    n_candidates = len(candidates)
    adj_matrix = coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_candidates, n_candidates)
    )
    _, labels = connected_components(csgraph=adj_matrix, directed=False)

    unique_labels, first_indices = np.unique(labels, return_index=True)
    sub_canonical = first_indices[np.searchsorted(unique_labels, labels)]
    canonical[candidates] = candidates[sub_canonical]
    return canonical


def _find_duplicate_node_map(node_lon, node_lat, tolerance=ERROR_TOLERANCE):
    """Map duplicate (within ``tolerance`` on the unit sphere) node indices to the
    lowest-indexed node sharing their location."""
    lon_rad = np.deg2rad(np.asarray(node_lon, dtype=np.float64))
    lat_rad = np.deg2rad(np.asarray(node_lat, dtype=np.float64))
    canonical = _coincident_node_canonical_indices(
        lonlat_rad=(lon_rad, lat_rad), tolerance=tolerance
    )

    n_node = len(lon_rad)
    duplicate_indices = np.flatnonzero(canonical != np.arange(n_node, dtype=INT_DTYPE))
    return {
        INT_DTYPE(index): INT_DTYPE(canonical[index]) for index in duplicate_indices
    }


def _find_duplicate_nodes(grid):
    """Map duplicate node indices to the canonical (lowest-indexed) node sharing
    their coordinates.

    Cached on the grid: the search is a KDTree query over every node, while the
    node coordinates it reads are fixed once the grid is constructed. Callers
    such as ``Grid.get_dual`` run this on every invocation, so recomputing it
    would make an O(n_node) search part of the cost of each dual construction.
    """
    if grid._duplicate_node_map is None:
        grid._duplicate_node_map = _find_duplicate_node_map(
            grid.node_lon.values, grid.node_lat.values
        )
    return grid._duplicate_node_map


def _live_node_indices(grid):
    """Node indices still referenced after connectivity is canonicalized.

    Duplicate node coordinates are left in the node arrays by design (see
    ``_find_duplicate_nodes``), but a raw coordinate-space search (e.g. a
    node KDTree/BallTree) can otherwise select a dead duplicate index that no
    face references. Callers building such trees should restrict to this set.
    """
    duplicate_map = _find_duplicate_nodes(grid)
    if not duplicate_map:
        return np.arange(grid.n_node, dtype=INT_DTYPE)
    dead = np.fromiter(duplicate_map.keys(), dtype=INT_DTYPE, count=len(duplicate_map))
    return np.setdiff1d(
        np.arange(grid.n_node, dtype=INT_DTYPE), dead, assume_unique=True
    )


def _check_normalization(grid):
    if grid._normalized:
        return True

    for name in ("node", "edge", "face"):
        if f"{name}_x" not in grid._ds:
            continue

        x = grid._ds[f"{name}_x"]
        y = grid._ds[f"{name}_y"]
        z = grid._ds[f"{name}_z"]

        # compute the max deviation from 1.0
        max_dev = abs((x**2 + y**2 + z**2 - 1.0).max().compute())

        if max_dev > ERROR_TOLERANCE:
            grid._normalized = False
            return False

    grid._normalized = True
    return True
