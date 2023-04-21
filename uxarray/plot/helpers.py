import uxarray

from uxarray.constants import INT_FILL_VALUE
from uxarray.helpers import close_face_nodes


def closed_face_verts(grid: uxarray.Grid):

    x = grid.Mesh2_node_x.values
    y = grid.Mesh2_node_y.values
    face_dimension = grid.Mesh2_face_dimension.values

    closed_face_nodes = close_face_nodes(grid.Mesh2_face_nodes.values,
                                         grid.nMesh2_face,
                                         grid.nMaxMesh2_face_nodes)
    face_verts_x = []
    face_verts_y = []
    for face_idx, face_dim in enumerate(face_dimension):
        face_verts_x.append(x[closed_face_nodes[face_idx, 0:face_dim]])
        face_verts_y.append(y[closed_face_nodes[face_idx, 0:face_dim]])

    return face_verts_x, face_verts_y


def compute_antimeridian_crossing(grid: uxarray.Grid, threshold=180.0):
    """Locates any face that crosses the antimeridian."""

    face_verts_x, face_verts_y = closed_face_verts(grid)

    # for vert_x, vert_y in zip(face_verts_x, face_verts_y):

    return

    # longitude (x) values of each face
    # face_nodes = self.Mesh2_face_nodes.astype(np.int32).values
    # face_lon = self.Mesh2_node_x.values[face_nodes]

    # pad last value to properly represent each side
    face_lon_pad = np.pad(face_lon, (0, 1), 'wrap')[:-1]

    # magnitude of each side
    side_diff = np.abs(np.diff(face_lon_pad))

    # true for any node that is part of an edge that crosses antimeridian
    crossed_mask_2d = side_diff > threshold

    # true for any face that contains a node that crosses antimeridian
    crossed_mask_1d = np.any(crossed_mask_2d, axis=1)

    # indices of faces that cross antimeridian
    crossed_indices = tuple(np.argwhere(crossed_mask_1d).squeeze())

    # include only faces that cross antimeridian
    crossed_mask_2d = crossed_mask_2d[crossed_indices, :].squeeze()

    return crossed_indices, crossed_mask_2d
