import numpy as np
import xarray as xr


def get_face_partitions(n_nodes_per_face):
    """Calculate the partitions of faces based on the number of nodes per face.

    This function organizes faces by their size, creating partitions that can be used
    for operations requiring uniform face geometry. It returns information about
    the boundaries of each partition and the sorted order of nodes.

    Parameters
    ----------
    n_nodes_per_face : array-like
        An array specifying the number of nodes per face for each face in the grid.

    Returns
    -------
    tuple
        - change_ind (np.ndarray): Indices indicating where the geometry changes
          from one shape to another in the sorted `n_nodes_per_face` array.
        - n_nodes_per_face_sorted_indices (np.ndarray): Indices that sort the `n_nodes_per_face`
          array in ascending order.
        - element_sizes (np.ndarray): The unique sizes of the faces in the grid.
        - size_counts (np.ndarray): The count of faces corresponding to each unique size.
    """
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
    """Initialize partitioning variables for the face connectivity based on the
    number of nodes per face.

    This function sets internal variables on the provided `uxgrid`
    object, which are used to define partitions for faces of different
    geometries. These partitions are necessary for efficient operations
    on unstructured grid data.
    """

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
    """Partitions the face connectivity (either face_node_connectivity or
    face_edge_connectivity) into a partitioned format.

    This function creates partitions of a dense face connectivity variable, such as `face_node_connectivity`
    or `face_edge_connectivity`, into a partitioned form based on the face geometries. The partitioned form
    allows for vectorized operations on individual face partitions.

    Parameters
    ----------
    uxgrid : ux.Grid
        Grid object

    connectivity_name : str
        Name of the connectivity variable to partition. Must be either `"face_node_connectivity"` or
        `"face_edge_connectivity"`.

    Returns
    -------
    PartitionedFaceNodeConnectivity or PartitionedFaceEdgeConnectivity
        An object containing the partitioned face connectivity and original face indices for each
        geometry in the grid.
    """
    if not hasattr(uxgrid, "_face_geometry_inflections"):
        # initialize partition variables
        initialize_face_partition_variables(uxgrid)

    if connectivity_name == "face_node_connectivity":
        conn = uxgrid.face_node_connectivity.values
    elif connectivity_name == "face_edge_connectivity":
        conn = uxgrid.face_edge_connectivity.values
    else:
        raise ValueError(f"Unknown connectivity name: {connectivity_name}")

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
    """A class for storing partitioned connectivity variables and original
    indices for geometries to enable efficient vectorized operations on each
    block of partitions.

    This class eliminates the need to store fill values by representing data in a partitioned format,
    making it easier to perform operations directly on the partitioned data. Each geometry is associated
    with a unique connectivity and index pair, stored in an internal xarray dataset.

    Parameters
    ----------
    geometries : list or set
        A collection of geometry identifiers. Each identifier corresponds to a distinct partition
        of face node connectivity and original face indices.
    connectivity_name : str, optional
        Name of the connectivity variable used in the internal dataset. Default is an empty string.
    indices_name : str, optional
        Name of the indices variable used in the internal dataset. Default is an empty string.

    Attributes
    ----------
    _geometries : set
        Set of geometry identifiers, representing each partition for which connectivity and indices are stored.
    _connectivity_name : str
        The name used for the partitioned connectivity variable.
    _indices_name : str
        The name used for the original face indices variable.
    _ds : xr.Dataset
        Internal xarray dataset used to store the connectivity and indices variables for each geometry.
    """

    def __init__(self, geometries, connectivity_name="", indices_name=""):
        """Class for storing partitioned connectivity variables without needing
        to store fill values."""
        self._geometries = set(geometries)
        self._connectivity_name = connectivity_name
        self._indices_name = indices_name
        self._ds = xr.Dataset()

    def __getitem__(self, geometry):
        """Retrieve the partitioned connectivity and original indices for a
        given geometry.

        Parameters
        ----------
        geometry : str or int
            The geometry identifier.

        Returns
        -------
        tuple
            A tuple containing:
            - `cur_partition` (any): The partitioned connectivity data for the specified geometry.
            - `cur_original__indices` (any): The original indices corresponding to the specified geometry.

        Example
        -------
        >>> cur_face_node_partition, cur_original_face_indices = uxgrid.partitioned_face_node_connectivity["3"]
        """
        return (
            self._ds[f"{self._connectivity_name}_{geometry}"].data,
            self._ds[f"{self._indices_name}_{geometry}"].data,
        )

    def __iter__(self):
        """Custom iterator that yields the partitioned connectivity and
        original indices for each geometry in the collection.

        Yields
        ------
        tuple
            A tuple containing:
            - `cur_partition` (any): Partitioned connectivity for the current geometry.
            - `cur_original_indices` (any): Original ndices corresponding to the current geometry.

        Example
        -------
        >>> for cur_face_node_partition, cur_original_face_indices in uxgrid.partitioned_face_node_connectivity:
        ...     # Access the partition and original indices for a given geometry (i.e. 3, 4, etc.)
        """
        for geom in self.geometries:
            yield self[str(geom)]

    @property
    def geometries(self):
        """The unique geometries present in the partitioned connectivity.

        Returns
        -------
        set
            A set of unique geometry identifiers representing each partition for which connectivity
            and original indices are stored.
        """
        return self._geometries

    @property
    def partitions(self):
        """Retrieve the partitioned connectivity for each geometry.

        Returns
        -------
        list
            A list where each entry is the partitioned connectivity data for a specific geometry.
        """
        return [self[geom][0].data for geom in self.geometries]

    @property
    def original_face_indices(self):
        """Retrieve the original indices that map each partitioned connectivity
        entry back to the index in the source connectivity.

        Returns
        -------
        list
            A list where each element is the array of original face indices for a specific geometry,
            allowing for reconstruction or indexing of the original connectivity
        """
        return [self[geom][1].data for geom in self.geometries]


class PartitionedFaceNodeConnectivity(BasePartitionedConnectivity):
    """Represents the ``face_node_connectivity`` in a partitioned manner."""

    def __repr__(self):
        repr_str = "Partitioned Face Node Connectivity\n"
        repr_str += "----------------------------------\n"
        for geom in self.geometries:
            repr_str += f" - {geom}: {self[geom][0].shape}\n"
        return repr_str


class PartitionedFaceEdgeConnectivity(BasePartitionedConnectivity):
    """Represents the ``face_node_connectivity`` in a partitioned manner."""

    def __repr__(self):
        repr_str = "Partitioned Face Edge Connectivity\n"
        repr_str += "----------------------------------\n"
        for geom in self.geometries:
            repr_str += f" - {geom}: {self[geom][0].shape}\n"
        return repr_str
