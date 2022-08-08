import numpy as np
class Edge:
    """The Uxarray Edge object class for undirected edge.
    In current implementation, each node is the node index
    """

    def __init__(self, input_edge):
        """Initializing the Edge object from input edge [node 0, node 1]
        Parameters
        ----------
        input_edge : xarray.Dataset, ndarray, list, tuple, required
            - The indexes of two nodes [node0_index, node1_index], the order doesn't matter
        ----------------
        """
        # for every input_edge, sort the node index in ascending order.
        edge_sorted = np.sort(input_edge)
        self.node0 = edge_sorted[0]
        self.node1 = edge_sorted[1]

    def __eq__(self, other):
        # Undirected edge
        return (self.node0 == other.node0 and self.node1 == other.node1) or \
               (self.node1 == other.node0 and self.node0 == other.node1)

    def __hash__(self):
        # Collisions are possible for hash
        return hash(self.node0 + self.node1)

    # Return nodes in list
    def get_nodes(self):
        return [self.node0, self.node1]