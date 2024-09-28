import numpy as np


# TODO: Better name for function
def source_tree_query(
    source_data, source_grid, destination_grid, coord_type, remap_to, k, query
):
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
