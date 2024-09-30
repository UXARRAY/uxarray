import numpy as np
import xarray as xr


def get_face_partitions(n_nodes_per_face):
    """TODO:"""
    # sort number of nodes per face in ascending order
    n_nodes_per_face_sorted_indices = np.argsort(n_nodes_per_face)

    # unique element sizes and their respective counts
    element_sizes, size_counts = np.unique(n_nodes_per_face, return_counts=True)
    element_sizes_sorted_ind = np.argsort(element_sizes)

    # sort elements by their size
    element_sizes = element_sizes[element_sizes_sorted_ind]
    size_counts = size_counts[element_sizes_sorted_ind]

    # find the index at the point where the geometry changes from one shape to another
    change_ind = np.cumsum(size_counts)
    change_ind = np.concatenate((np.array([0]), change_ind))

    return change_ind, n_nodes_per_face_sorted_indices, element_sizes, size_counts


def initialize_face_partition_variables(uxgrid):
    """TODO:"""

    (
        face_geometry_inflections,
        n_nodes_per_face_sorted_indices,
        face_geometries,
        face_geometries_counts,
    ) = get_face_partitions(uxgrid.n_nodes_per_face.values)

    uxgrid._face_geometry_inflections = face_geometry_inflections
    uxgrid._n_nodes_per_face_sorted_indices = n_nodes_per_face_sorted_indices
    uxgrid._face_geometries = face_geometries
    uxgrid._face_geometries_counts = face_geometries_counts


def _build_partitioned_face_connectivity(uxgrid, connectivity_name):
    """TODO:"""
    if not hasattr(uxgrid, "_face_geometry_inflections"):
        # TODO:
        initialize_face_partition_variables(uxgrid)

    if connectivity_name == "face_node_connectivity":
        conn = uxgrid.face_node_connectivity.values
    elif connectivity_name == "face_edge_connectivity":
        conn = uxgrid.face_edge_connectivity.values
    else:
        raise ValueError("TODO: ")

    face_geometry_inflections = uxgrid._face_geometry_inflections
    n_nodes_per_face_sorted_indices = uxgrid._n_nodes_per_face_sorted_indices
    face_geometries = uxgrid._face_geometries
    face_geometries_counts = uxgrid._face_geometries

    if connectivity_name == "face_node_connectivity":
        partition_obj = PartitionedFaceNodeConnectivity
    elif connectivity_name == "face_edge_connectivity":
        partition_obj = PartitionedFaceEdgeConnectivity
    else:
        raise ValueError()

    partitioned_connectivity = partition_obj(
        face_geometries,
        connectivity_name=connectivity_name,
        indices_name="original_face_indices",
    )

    for e, e_count, start, end in zip(
        face_geometries,
        face_geometries_counts,
        face_geometry_inflections[:-1],
        face_geometry_inflections[1:],
    ):
        original_face_indices = n_nodes_per_face_sorted_indices[start:end]
        conn_par = conn[original_face_indices, 0:e]

        # Store partitioned face node connectivity
        partitioned_connectivity._ds[f"{connectivity_name}_{e}"] = xr.DataArray(
            data=conn_par, dims=[f"n_face_{str(e)}", str(e)]
        )

        if f"original_face_indices_{e}" not in partitioned_connectivity._ds:
            # Store original face indices (avoid duplicates)
            partitioned_connectivity._ds[f"original_face_indices_{e}"] = xr.DataArray(
                data=original_face_indices, dims=[f"n_face_{str(e)}"]
            )

    return partitioned_connectivity


class BasePartitionedConnectivity:
    def __init__(self, geometries, connectivity_name="", indices_name=""):
        # TODO
        self._geometries = set(geometries)
        self._connectivity_name = connectivity_name
        self._indices_name = indices_name
        self._ds = xr.Dataset()

    def __getitem__(self, geometry):
        return (
            self._ds[f"{self._connectivity_name}_{geometry}"].data,
            self._ds[f"{self._indices_name}_{geometry}"].data,
        )

    def __iter__(self):
        for geom in self.geometries:
            yield self[str(geom)]

    @property
    def geometries(self):
        # TODO:
        return self._geometries

    @property
    def partitions(self):
        return [self[geom][0].data for geom in self.geometries]

    @property
    def original_face_indices(self):
        return [self[geom][1].data for geom in self.geometries]


class PartitionedFaceNodeConnectivity(BasePartitionedConnectivity):
    def __repr__(self):
        repr_str = "Partitioned Face Node Connectivity\n"
        repr_str += "----------------------------------\n"
        for geom in self.geometries:
            repr_str += f" - {geom}: {self[geom][0].shape}\n"

        return repr_str


class PartitionedFaceEdgeConnectivity(BasePartitionedConnectivity):
    def __repr__(self):
        repr_str = "Partitioned Face Edge Connectivity\n"
        repr_str += "----------------------------------\n"
        for geom in self.geometries:
            repr_str += f" - {geom}: {self[geom][0].shape}\n"

        return repr_str
