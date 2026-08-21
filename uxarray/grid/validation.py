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
    """Check if there are duplicate node indices, returns True if there are."""

    # Create a duplication dictionary
    duplicate_node_dict = _find_duplicate_nodes(grid)

    for face_nodes in grid.face_node_connectivity.values:
        for node in face_nodes:
            if node in duplicate_node_dict.keys():
                return True

    return False


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


def _coincident_node_canonical_indices(points_xyz, tolerance=ERROR_TOLERANCE):
    """For each point, find the lowest-indexed point within ``tolerance`` chordal
    distance on the unit sphere (a point with no coincident neighbor maps to itself).

    Points at the geographic poles are never merged with one another: longitude is
    singular there, and grid files (e.g. SCRIP cube-sphere) commonly give each face
    touching a pole its own arbitrary-but-meaningful longitude for that corner, which
    downstream lat/lon bounds and zonal-weight code relies on staying distinct per
    face even though the xyz location is identical.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import KDTree

    n_points = len(points_xyz)
    canonical = np.arange(n_points, dtype=INT_DTYPE)

    pole_mask = np.isclose(np.abs(points_xyz[:, 2]), 1.0, atol=tolerance)
    mergeable_indices = np.flatnonzero(~pole_mask)

    if len(mergeable_indices) < 2:
        return canonical

    tree = KDTree(points_xyz[mergeable_indices])
    pairs = tree.query_pairs(r=tolerance, output_type="ndarray")

    if len(pairs) == 0:
        return canonical

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    n_mergeable = len(mergeable_indices)
    adj_matrix = coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_mergeable, n_mergeable)
    )
    _, labels = connected_components(csgraph=adj_matrix, directed=False)

    unique_labels, first_indices = np.unique(labels, return_index=True)
    sub_canonical = first_indices[np.searchsorted(unique_labels, labels)]
    canonical[mergeable_indices] = mergeable_indices[sub_canonical]
    return canonical


def _find_duplicate_node_map(node_lon, node_lat, tolerance=ERROR_TOLERANCE):
    """Map duplicate (within ``tolerance`` on the unit sphere) node indices to the
    lowest-indexed node sharing their location."""
    points_xyz = np.column_stack(
        _lonlat_rad_to_xyz(np.deg2rad(node_lon), np.deg2rad(node_lat))
    )
    canonical = _coincident_node_canonical_indices(points_xyz, tolerance)

    n_node = len(node_lon)
    duplicate_indices = np.flatnonzero(canonical != np.arange(n_node, dtype=INT_DTYPE))
    return {
        INT_DTYPE(index): INT_DTYPE(canonical[index]) for index in duplicate_indices
    }


def _find_duplicate_nodes(grid):
    """Map duplicate node indices to the canonical (lowest-indexed) node sharing
    their coordinates."""
    return _find_duplicate_node_map(grid.node_lon.values, grid.node_lat.values)


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
