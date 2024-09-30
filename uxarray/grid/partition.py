import numpy as np
import dask.array as da


def get_face_partitions(n_nodes_per_face):
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
    # number of nodes and edges for a face are equal

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


def _build_partitioned_face_node_connectivity(uxgrid):
    if not hasattr(uxgrid, "_face_geometry_inflections"):
        # TODO:
        initialize_face_partition_variables(uxgrid)

    face_node_connectivity = uxgrid.face_node_connectivity.values
    face_geometry_inflections = uxgrid._face_geometry_inflections
    n_nodes_per_face_sorted_indices = uxgrid._n_nodes_per_face_sorted_indices
    face_geometries = uxgrid._face_geometries
    face_geometries_counts = uxgrid._face_geometries

    # TODO:
    partitioned_face_node_connectivity = PartitionedFaceNodeConnectivity(
        face_geometries,
        connectivity_name="face_node_connectivity",
        indices_name="original_face_indices",
    )

    for e, e_count, start, end in zip(
        face_geometries,
        face_geometries_counts,
        face_geometry_inflections[:-1],
        face_geometry_inflections[1:],
    ):
        original_face_indices = n_nodes_per_face_sorted_indices[start:end]
        face_nodes_par = face_node_connectivity[original_face_indices, 0:e]

        # TODO
        setattr(
            partitioned_face_node_connectivity,
            f"face_node_connectivity_{e}",
            face_nodes_par,
        )
        setattr(
            partitioned_face_node_connectivity,
            f"original_face_indices_{e}",
            original_face_indices,
        )

    return partitioned_face_node_connectivity


class BasePartitionedConnectivity:
    def __init__(self, geometries, connectivity_name="", indices_name=""):
        # TODO
        self._geometries = set(geometries)
        self._connectivity_name = connectivity_name
        self._indices_name = indices_name

    def __getitem__(self, geometry):
        return (
            getattr(self, f"{self._connectivity_name}_{geometry}"),
            getattr(self, f"{self._indices_name}_{geometry}"),
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
        return [self[geom] for geom in self.geometries]

    @property
    def original_face_indices(self):
        return getattr(f"original_face_indices_{geom}" for geom in self.geometries)

    def chunk(self, chunks=-1):
        for geom in self.geometries:
            partition = da.from_array(
                getattr(self, f"{self._connectivity_name}_{geom}"), chunks=chunks
            )
            face_indices = da.from_array(
                getattr(self, f"{self._indices_name}_{geom}"), chunks=chunks
            )

            setattr(self, f"{self._connectivity_name}_{geom}", partition)
            setattr(self, f"{self._indices_name}_{geom}", face_indices)

    def persist(self):
        for geom in self.geometries:
            partition = getattr(self, f"{self._connectivity_name}_{geom}").persist()
            face_indices = getattr(self, f"{self._indices_name}_{geom}").persist()

            setattr(self, f"{self._connectivity_name}_{geom}", partition)
            setattr(self, f"{self._indices_name}_{geom}", face_indices)


class PartitionedFaceNodeConnectivity(BasePartitionedConnectivity):
    def __repr__(self):
        repr_str = "Partitioned Face Node Connectivity\n"
        repr_str += "----------------------------------\n"
        for geom in self.geometries:
            repr_str += f" - {geom}: {self[geom][0].shape}\n"

        return repr_str
