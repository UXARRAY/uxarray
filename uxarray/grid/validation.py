import numpy as np
from uxarray.io._ugrid import _is_ugrid
from uxarray.constants import ERROR_TOLERANCE


# validation helper functions
def _check_connectivity(self):
    """Check if all nodes are referenced by at least one element.

    If not, the mesh may have hanging nodes and may not a valid UGRID
    mesh
    """

    # Check if all nodes are referenced by at least one element
    # get unique nodes in connectivity
    nodes_in_conn = np.unique(self.Mesh2_face_nodes.values.flatten())
    #  remove negative indices/fill values from the list
    nodes_in_conn = nodes_in_conn[nodes_in_conn >= 0]

    # check if the size of unique nodes in connectivity is equal to the number of nodes
    if (nodes_in_conn.size == self.nMesh2_node):
        print("-All nodes are referenced by at least one element.")
        return True
    else:
        print("-WARNING: Some nodes may not referenced by any element.",
              nodes_in_conn.size, self.nMesh2_node)
        return False


def _check_duplicate_nodes(self):
    """Check if there are duplicate nodes in the mesh."""

    coords1 = np.column_stack(
        (np.vstack(self.Mesh2_node_x), np.vstack(self.Mesh2_node_y)))
    unique_nodes, indices = np.unique(coords1, axis=0, return_index=True)
    duplicate_indices = np.setdiff1d(np.arange(len(coords1)), indices)

    if duplicate_indices.size > 0:
        print("-WARNING: Duplicate nodes found in the mesh. # ",
              duplicate_indices.size, " nodes are duplicates.")
        return False
    else:
        print("-No duplicate nodes found in the mesh.")
        return True
