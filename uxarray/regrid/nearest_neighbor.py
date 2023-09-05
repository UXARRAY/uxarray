from uxarray.grid import Grid
import numpy as np


# node to node              node_xy -> node_xy              what tree we need to query
# node to face              node_xy -> face_xy
# face to node              face_xy -> node_xy
# face to face              face_xy -> face_xy
def nearest_neighbor(source_grid: Grid,
                     destination_grid: Grid,
                     source_data: np.ndarray,
                     destination_data_mapping: str = "nodes",
                     coord_type: str = "lonlat") -> np.ndarray:
    """Nearest Neighbor Remapping between two grids, mapping data that resides
    on either the corner nodes or face centers on the source grid to the corner
    nodes or face centers of the destination grid..

    Parameters
    ---------
    source_grid : Grid
        Source grid that data is mapped to
    destination_grid : Grid
        Destination grid to regrid data to
    source_data : np.ndarray
        Data variable to regrid
    destination_data_mapping : str, default="nodes"
        Location of where to map data, either "nodes" or "face centers"

    Returns
    -------
    destination_data : np.ndarray
        Data mapped to destination grid
    """

    # TODO: implementation in latlon, consider cartesiain once KDtree is implemented

    # ensure array is an np.ndarray
    source_data = np.asarray(source_data)

    n_elements = source_data.shape[-1]

    if n_elements == source_grid.nMesh2_node:
        source_data_mapping = "nodes"
    elif n_elements == source_grid.nMesh2_face:
        source_data_mapping = "face centers"
    else:
        raise ValueError(
            f"Invalid source_data shape. The final dimension should be either match the number of corner "
            f"nodes ({source_grid.nMesh2_node}) or face centers ({source_grid.nMesh2_face}) in the "
            f"source grid, but received: {source_data.shape}")

    if coord_type == "lonlat":
        # get destination coordinate pairs
        if destination_data_mapping == "nodes":
            lon, lat = destination_grid.Mesh2_node_x.values, destination_grid.Mesh2_node_y.values

        elif destination_data_mapping == "face centers":
            lon, lat = destination_grid.Mesh2_face_x.values, destination_grid.Mesh2_face_y.values
        else:
            raise ValueError(
                f"Invalid destination_data_mapping. Expected 'nodes' or 'face centers', "
                f"but received: {destination_data_mapping}")

        # specify whether to query on the corner nodes or face centers based on source grid
        _source_tree = source_grid.get_ball_tree(tree_type=source_data_mapping)

        # prepare coordinates for query
        lonlat = np.vstack([lon, lat]).T

        _, nearest_neighbor_indices = _source_tree.query(lonlat, k=1)

        # data values from source data to destination data using nearest neighbor indices
        destination_data = source_data[nearest_neighbor_indices]

        # case for 1D slice of data
        if destination_data.ndim > 1:
            destination_data = destination_data.squeeze()

        return destination_data

    elif coord_type == "cartesian":
        # TODO: once a cartesian balltree/kdtree is implemented, implement this
        raise ValueError(
            f"Nearest Neighbor Regridding using Cartesian coordinates is not yet supported"
        )

    else:
        raise ValueError(
            f"Invalid coord_type. Expected either 'lonlat' or 'artesian', but received {coord_type}"
        )
