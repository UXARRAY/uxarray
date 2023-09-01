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
    """Nearest Neighbor Remapping between two grids.

    Parameters
    ---------
    source_grid : Grid
        Source grid that data is mapped to
    destination_grid : Grid
        Destination grid to regrid data to
    source_data : np.ndarray
        Data variable to regrid
    destination_data_mapping : str, default="nodes"
        Location of where to map data

    Returns
    -------
    destination_data : np.ndarray
        Data mapped to destination grid
    """

    # TODO: implementation in latlon, consider cartesiain once KDtree is implemented

    if source_data.size[-1] == source_grid.nMesh2_node:
        source_data_mapping = "nodes"
    elif source_data.size[-1] == source_grid.nMesh2_face:
        source_data_mapping = "face centers"
    else:
        raise ValueError

    _source_tree = source_grid.get_ball_tree(tree_type=source_data_mapping)

    # regrid a variable that originates on each node
    if source_data_mapping == "nodes":

        if destination_data_mapping == "nodes":
            x, y = destination_grid.Mesh2_node_x.values, destination_grid.Mesh2_node_y.values,

        elif destination_data_mapping == "face centers":
            x, y = destination_grid.Mesh2_face_x.values, destination_grid.Mesh2_face_y.values,
            pass

        # TODO: rest of algorithm
        xy = np.vstack(x, y).T

        d, ind = _source_tree.query(xy)

    # regrid a variable that originates on each face center
    if source_data_mapping == "face centers":

        if destination_data_mapping == "nodes":
            pass
        elif destination_data_mapping == "face centers":
            pass

    # TODO: generalize for higher dimensional data
    destination_data = source_data[ind]
