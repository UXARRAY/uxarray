import numpy as np


def inverse_distance_weighted_remap(source_grid,
                                    destination_grid,
                                    source_data,
                                    remap_to="nodes",
                                    coord_type="spherical",
                                    power=2,
                                    k_neighbors=8):
    """Inverse Distance Weighted Remapping between two grids.

    Parameters:
    -----------
    source_grid : Grid
        Source grid that data is mapped from.
    destination_grid : Grid
        Destination grid to remap data to.
    source_data : np.ndarray
        Data variable to remap.
    remap_to : str, default="nodes"
        Location of where to map data, either "nodes" or "face centers".
    coord_type: str, default="spherical"
        Coordinate type to use for nearest neighbor query, either "spherical" or "Cartesian".
    power : float, default=2
        Power parameter for inverse distance weighting.
    k_neighbors : int, default=8
        Number of nearest neighbors to consider in the weighted calculation.

    Returns:
    --------
    destination_data : np.ndarray
        Data mapped to the destination grid.
    """

    source_data = np.asarray(source_data)
    n_elements = source_data.shape[-1]

    if n_elements == source_grid.n_node:
        source_data_mapping = "nodes"
    elif n_elements == source_grid.n_face:
        source_data_mapping = "face centers"
    else:
        raise ValueError(
            f"Invalid source_data shape. The final dimension should match the number of corner "
            f"nodes ({source_grid.n_node}) or face centers ({source_grid.n_face}) in the "
            f"source grid, but received: {source_data.shape}")

    if coord_type == "spherical":
        if remap_to == "nodes":
            lon, lat = destination_grid.node_lon.values, destination_grid.node_lat.values
        elif remap_to == "face centers":
            lon, lat = destination_grid.face_lon.values, destination_grid.face_lat.values
        else:
            raise ValueError(
                f"Invalid remap_to. Expected 'nodes' or 'face centers', "
                f"but received: {remap_to}")

        _source_tree = source_grid.get_ball_tree(tree_type=source_data_mapping)

        dest_coords = np.vstack([lon, lat]).T

        distances, nearest_neighbor_indices = _source_tree.query(dest_coords,
                                                                 k=k_neighbors)

    elif coord_type == "cartesian":
        if remap_to == "nodes":
            x, y, z = (destination_grid.node_x.values,
                       destination_grid.node_y.values,
                       destination_grid.node_z.values)
        elif remap_to == "face centers":
            x, y, z = (destination_grid.face_x.values,
                       destination_grid.face_y.values,
                       destination_grid.face_z.values)
        else:
            raise ValueError(
                f"Invalid remap_to. Expected 'nodes' or 'face centers', "
                f"but received: {remap_to}")

        _source_tree = source_grid.get_kd_tree(tree_type=source_data_mapping)

        dest_coords = np.vstack([x, y, z]).T

        distances, nearest_neighbor_indices = _source_tree.query(dest_coords,
                                                                 k=k_neighbors)

    else:
        raise ValueError(
            f"Invalid coord_type. Expected either 'spherical' or 'cartesian', but received {coord_type}"
        )

    if nearest_neighbor_indices.ndim > 1:
        nearest_neighbor_indices = nearest_neighbor_indices.squeeze()

    weights = 1 / (distances**power + 1e-6)
    weights /= np.sum(weights)

    destination_data = np.sum(source_data[..., nearest_neighbor_indices] *
                              weights,
                              axis=-1)

    return destination_data
