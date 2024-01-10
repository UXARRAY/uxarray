import numpy as np
from warnings import warn

from uxarray.constants import ERROR_TOLERANCE


# validation helper functions
def _check_connectivity(self):
    """Check if all nodes are referenced by at least one element.

    If not, the mesh may have hanging nodes and may not a valid UGRID
    mesh
    """

    # Check if all nodes are referenced by at least one element
    # get unique nodes in connectivity
    nodes_in_conn = np.unique(self.face_node_connectivity.values.flatten())
    #  remove negative indices/fill values from the list
    nodes_in_conn = nodes_in_conn[nodes_in_conn >= 0]

    # check if the size of unique nodes in connectivity is equal to the number of nodes
    if nodes_in_conn.size == self.n_node:
        print("-All nodes are referenced by at least one element.")
        return True
    else:
        warn(
            "Some nodes may not be referenced by any element. {0} and {1}".format(
                nodes_in_conn.size, self.n_node
            ),
            RuntimeWarning,
        )
        return False


def _check_duplicate_nodes(self):
    """Check if there are duplicate nodes in the mesh."""

    coords1 = np.column_stack((np.vstack(self.node_lon), np.vstack(self.node_lat)))
    unique_nodes, indices = np.unique(coords1, axis=0, return_index=True)
    duplicate_indices = np.setdiff1d(np.arange(len(coords1)), indices)

    if duplicate_indices.size > 0:
        warn(
            "Duplicate nodes found in the mesh. {0} nodes are duplicates.".format(
                duplicate_indices.size
            ),
            RuntimeWarning,
        )
        return False
    else:
        print("-No duplicate nodes found in the mesh.")
        return True


def _check_area(self):
    """Check if each face area is greater than our constant ERROR_TOLERANCE."""
    areas = self.face_areas
    # Check if area of any face is close to zero
    if np.any(np.isclose(areas, 0, atol=ERROR_TOLERANCE)):
        warn(
            "At least one face area is close to zero. Mesh may contain inverted elements",
            RuntimeWarning,
        )
        return False
    else:
        print("-No face area is close to zero.")
        return True
