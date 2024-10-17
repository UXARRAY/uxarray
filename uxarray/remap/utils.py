import numpy as np


def _remap_grid_parse(
    source_data, source_grid, destination_grid, coord_type, remap_to, k, query
):
    """Gets the destination coordinates from the destination grid for
    remapping, as well as retrieving the nearest neighbor indices and
    distances.

    Parameters:
    -----------
    source_data : np.ndarray
        Data variable to remap.
    source_grid : Grid
        Source grid that data is mapped from.
    destination_grid : Grid
        Destination grid to remap data to.
    coord_type: str
        Coordinate type to use for nearest neighbor query, either "spherical" or "Cartesian".
    remap_to : str
        Location of where to map data, either "nodes", "edge centers", or "face centers".
    k : int
        Number of nearest neighbors to consider in the weighted calculation.
    query : bool
        Whether to construct and query the tree based on the source grid.

    Returns:
    --------
    dest_coords : np.ndarray
        Returns the proper destination coordinates based on `remap_to`
    distances : np.ndarray
        Returns the distances of the query of `k` nearest neighbors.
    nearest_neighbor_indices : np.ndarray
        Returns the nearest neighbor indices of number `k`.
    """

    n_elements = source_data.shape[-1]

    if n_elements == source_grid.n_node:
        source_data_mapping = "nodes"
    elif n_elements == source_grid.n_face:
        source_data_mapping = "face centers"
    elif n_elements == source_grid.n_edge:
        source_data_mapping = "edge centers"
    else:
        raise ValueError(
            f"Invalid source_data shape. The final dimension should match the number of corner "
            f"nodes ({source_grid.n_node}), edge nodes ({source_grid.n_edge}), or face centers ({source_grid.n_face}) "
            f"in the source grid, but received: {source_data.shape}"
        )

    if coord_type == "spherical":
        if remap_to == "nodes":
            lon, lat = (
                destination_grid.node_lon.values,
                destination_grid.node_lat.values,
            )
        elif remap_to == "face centers":
            lon, lat = (
                destination_grid.face_lon.values,
                destination_grid.face_lat.values,
            )
        elif remap_to == "edge centers":
            lon, lat = (
                destination_grid.edge_lon.values,
                destination_grid.edge_lat.values,
            )
        else:
            raise ValueError(
                f"Invalid remap_to. Expected 'nodes', 'edge centers', or 'face centers', "
                f"but received: {remap_to}"
            )

        _source_tree = source_grid.get_ball_tree(
            coordinates=source_data_mapping, reconstruct=True
        )

        dest_coords = np.vstack([lon, lat]).T

        if query:
            distances, nearest_neighbor_indices = _source_tree.query(dest_coords, k=k)

    elif coord_type == "cartesian":
        if remap_to == "nodes":
            x, y, z = (
                destination_grid.node_x.values,
                destination_grid.node_y.values,
                destination_grid.node_z.values,
            )
        elif remap_to == "face centers":
            x, y, z = (
                destination_grid.face_x.values,
                destination_grid.face_y.values,
                destination_grid.face_z.values,
            )
        elif remap_to == "edge centers":
            x, y, z = (
                destination_grid.edge_x.values,
                destination_grid.edge_y.values,
                destination_grid.edge_z.values,
            )
        else:
            raise ValueError(
                f"Invalid remap_to. Expected 'nodes', 'edge centers', or 'face centers', "
                f"but received: {remap_to}"
            )

        _source_tree = source_grid.get_ball_tree(
            coordinates=source_data_mapping,
            coordinate_system="cartesian",
            distance_metric="minkowski",
            reconstruct=True,
        )

        dest_coords = np.vstack([x, y, z]).T

        if query:
            distances, nearest_neighbor_indices = _source_tree.query(dest_coords, k=k)

    else:
        raise ValueError(
            f"Invalid coord_type. Expected either 'spherical' or 'cartesian', but received {coord_type}"
        )

    if nearest_neighbor_indices.ndim > 1:
        nearest_neighbor_indices = nearest_neighbor_indices.squeeze()

    if query:
        return dest_coords, distances, nearest_neighbor_indices
    else:
        return dest_coords
