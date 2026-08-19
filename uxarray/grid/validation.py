from warnings import warn

import numpy as np
import polars as pl

from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE


def _check_connectivity(grid):
    """Check if all nodes are referenced by at least one element."""

    # Convert face_node_connectivity to a Polars Series and get unique values
    nodes_in_conn = pl.Series(grid.face_node_connectivity.values.flatten()).unique()

    # Filter out negative values
    nodes_in_conn = nodes_in_conn.filter(nodes_in_conn >= 0)

    # Check if the size of unique nodes in connectivity is equal to the number of nodes
    if len(nodes_in_conn) == grid.n_node:
        return True
    else:
        warn(
            f"Some nodes may not be referenced by any element. {len(nodes_in_conn)} and {grid.n_node}",
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


def _find_duplicate_nodes(grid):
    """Map duplicate node indices to the first index with the same coordinates."""
    node_coordinates = np.column_stack((grid.node_lon.values, grid.node_lat.values))
    _, first_indices, inverse_indices = np.unique(
        node_coordinates, axis=0, return_index=True, return_inverse=True
    )

    duplicate_indices = np.flatnonzero(
        np.arange(grid.n_node, dtype=INT_DTYPE) != first_indices[inverse_indices]
    )
    return {
        INT_DTYPE(index): INT_DTYPE(first_indices[inverse_indices[index]])
        for index in duplicate_indices
    }


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
