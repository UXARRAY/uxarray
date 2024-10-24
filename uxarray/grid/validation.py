import numpy as np
from warnings import warn


from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE


# validation helper functions
def _check_connectivity(grid):
    """Check if all nodes are referenced by at least one element.

    If not, the mesh may have hanging nodes and may not a valid UGRID
    mesh
    """

    # Check if all nodes are referenced by at least one element
    # get unique nodes in connectivity
    nodes_in_conn = np.unique(grid.face_node_connectivity.values.flatten())
    #  remove negative indices/fill values from the list
    nodes_in_conn = nodes_in_conn[nodes_in_conn >= 0]

    # check if the size of unique nodes in connectivity is equal to the number of nodes
    if nodes_in_conn.size == grid.n_node:
        return True
    else:
        warn(
            "Some nodes may not be referenced by any element. {0} and {1}".format(
                nodes_in_conn.size, grid.n_node
            ),
            RuntimeWarning,
        )
        return False


def _check_duplicate_nodes(grid):
    """Check if there are duplicate nodes in the mesh."""

    coords = np.vstack([grid.node_lon.values, grid.node_lat.values])
    unique_nodes, indices = np.unique(coords, axis=0, return_index=True)
    duplicate_indices = np.setdiff1d(np.arange(len(coords)), indices)

    if duplicate_indices.size > 0:
        warn(
            "Duplicate nodes found in the mesh. {0} nodes are duplicates.".format(
                duplicate_indices.size
            ),
            RuntimeWarning,
        )
        return False
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
    # list of tuple indices
    lonlat_t = [
        (lon, lat) for lon, lat in zip(grid.node_lon.values, grid.node_lat.values)
    ]

    # # Dictionary to track first occurrence and subsequent indices
    occurrences = {}

    # Iterate through the list and track occurrences
    for index, tpl in enumerate(lonlat_t):
        if tpl in occurrences:
            occurrences[tpl].append((INT_DTYPE(index)))
        else:
            occurrences[tpl] = [INT_DTYPE(index)]

    duplicate_dict = {}

    for tpl, indices in occurrences.items():
        if len(indices) > 1:
            source_idx = indices[0]
            for duplicate_idx in indices[1:]:
                duplicate_dict[duplicate_idx] = source_idx

    return duplicate_dict


def _check_normalization(grid):
    """Checks whether all the cartesiain coordinates are normalized."""

    if grid._normalized is True:
        # grid is already normalized, no need to run extra checks
        return grid._normalized

    if "node_x" in grid._ds:
        if not (
            np.isclose(
                (grid.node_x**2 + grid.node_y**2 + grid.node_z**2),
                1.0,
                atol=ERROR_TOLERANCE,
            )
        ).all():
            grid._normalized = False
            return False
    if "edge_x" in grid._ds:
        if not (
            np.isclose(
                (grid.node_x**2 + grid.node_y**2 + grid.node_z**2),
                1.0,
                atol=ERROR_TOLERANCE,
            )
        ).all():
            grid._normalized = False
            return False
    if "face_x" in grid._ds:
        if not (
            np.isclose(
                (grid.node_x**2 + grid.node_y**2 + grid.node_z**2),
                1.0,
                atol=ERROR_TOLERANCE,
            )
        ).all():
            grid._normalized = False
            return False

    # set the grid as normalized
    grid._normalized = True

    return True
