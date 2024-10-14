from uxarray.constants import INT_FILL_VALUE

CONVENTIONS_ATTR = "UGRID-v1.0"

# minimum grid topology, additions are made depending on what is present in a grid
BASE_GRID_TOPOLOGY_ATTRS = {
    "cf_role": "mesh_topology",
    "topology_dimension": 2,
    # dimensions
    "face_dimension": "n_face",
    "node_dimension": "n_node",
    # coordinates
    "node_coordinates": "node_lon node_lat",
    # connectivity
    "face_node_connectivity": "face_node_connectivity",
}

DIM_NAMES = [
    "n_node",
    "n_edge",
    "n_face",
    "n_max_face_nodes",
    "n_max_face_edges",
    "n_max_face_faces",
    "n_max_edge_nodes",
    "n_max_edge_edges",
    "n_max_edge_faces",
    "n_max_node_faces",
    "n_max_node_edges",
    "two",
]

NODE_DIM = "n_node"
EDGE_DIM = "n_edge"
FACE_DIM = "n_face"
N_MAX_FACE_NODES_DIM = "n_max_face_nodes"

NODE_COORDINATES = ["node_lon", "node_lat"]

# coordinates (spherical)
NODE_LON_ATTRS = {
    "standard_name": "longitude",
    "long name": "Longitude of the corner nodes of each face",
    "units": "degrees_east",
}

NODE_LAT_ATTRS = {
    "standard_name": "latitude",
    "long name": "Latitude of the corner nodes of each face",
    "units": "degrees_north",
}

EDGE_COORDINATES = ["edge_lon", "edge_lat"]

EDGE_LON_ATTRS = {
    "standard_name": "longitude",
    "long name": "Longitude of the center of each edge",
    "units": "degrees_east",
}

EDGE_LAT_ATTRS = {
    "standard_name": "latitude",
    "long name": "Latitude of the center of each edge",
    "units": "degrees_north",
}

FACE_COORDINATES = ["face_lon", "face_lat"]

FACE_LON_ATTRS = {
    "standard_name": "longitude",
    "long name": "Longitude of the center of each face",
    "units": "degrees_east",
}

FACE_LAT_ATTRS = {
    "standard_name": "latitude",
    "long name": "Latitude of the center of each face",
    "units": "degrees_north",
}

CARTESIAN_NODE_COORDINATES = ["node_x", "node_y", "node_z"]

NODE_X_ATTRS = {
    "standard_name": "x",
    "long name": "Cartesian x location of the corner nodes of each face",
    "units": "meters",
}

NODE_Y_ATTRS = {
    "standard_name": "y",
    "long name": "Cartesian y location of the corner nodes of each face",
    "units": "meters",
}

NODE_Z_ATTRS = {
    "standard_name": "z",
    "long name": "Cartesian z location of the corner nodes of each face",
    "units": "meters",
}

CARTESIAN_EDGE_COORDINATES = ["edge_x", "edge_y", "edge_z"]

EDGE_X_ATTRS = {
    "standard_name": "x",
    "long name": "Cartesian x location of the center of each edge",
    "units": "meters",
}

EDGE_Y_ATTRS = {
    "standard_name": "y",
    "long name": "Cartesian y location of the center of each edge",
    "units": "meters",
}

EDGE_Z_ATTRS = {
    "standard_name": "z",
    "long name": "Cartesian z location of the center of each edge",
    "units": "meters",
}

CARTESIAN_FACE_COORDINATES = ["face_x", "face_y", "face_z"]

FACE_X_ATTRS = {
    "standard_name": "x",
    "long name": "Cartesian x location of the center of each face",
    "units": "meters",
}

FACE_Y_ATTRS = {
    "standard_name": "y",
    "long name": "Cartesian y location of the center of each face",
    "units": "meters",
}

FACE_Z_ATTRS = {
    "standard_name": "z",
    "long name": "Cartesian z location of the center of each face",
    "units": "meters",
}

# connectivity (face_)
FACE_NODE_CONNECTIVITY_ATTRS = {
    "cf_role": "face_node_connectivity",
    "long name": "Maps every face to its corner nodes.",
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE,
}
FACE_NODE_CONNECTIVITY_DIMS = ["n_face", "n_max_face_nodes"]

FACE_EDGE_CONNECTIVITY_ATTRS = {
    "cf_role": "face_edge_connectivity",
    "long name": "Maps every face to its edges.",
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE,
}
FACE_EDGE_CONNECTIVITY_DIMS = [
    "n_face",
    "n_max_face_edges",
]  # n_max_face_edges equiv to n_max_face_nodes

FACE_FACE_CONNECTIVITY_ATTRS = {
    "cf_role": "face_face_connectivity",
    "long name": "Faces that neighbor each face.",
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE,
}
FACE_FACE_CONNECTIVITY_DIMS = [
    "n_face",
    "n_max_face_faces",
]

# connectivity (edge_)
EDGE_NODE_CONNECTIVITY_ATTRS = {
    "cf_role": "edge_node_connectivity",
    "long name": "Maps every edge to the two nodes that it connects.",
    "start_index": 0,
}
EDGE_NODE_CONNECTIVITY_DIMS = ["n_edge", "two"]


# edge_edge_connectivity not yet supported
# EDGE_EDGE_CONNECTIVITY_ATTRS = {
#     "cf_role": "edge_edge_connectivity",
#     "long name": "Edges that neighbor each edge",
#     "start_index": 0,
#     "_FillValue": INT_FILL_VALUE,
#     "dtype": INT_DTYPE,
# }
#
# EDGE_EDGE_CONNECTIVITY_DIMS = ["n_edge", "n_max_edge_edges"]

EDGE_FACE_CONNECTIVITY_ATTRS = {
    "cf_role": "edge_face_connectivity",
    "long name": "Faces that neighbor each edge",
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE,
}
EDGE_FACE_CONNECTIVITY_DIMS = ["n_edge", "two"]


NODE_EDGE_CONNECTIVITY_ATTRS = {
    "cf_role": "node_edge_connectivity",
    "long name": "Edges that neighbor each node",
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE,
}

NODE_EDGE_CONNECTIVITY_DIMS = ["n_node", "n_max_node_edges"]

NODE_FACE_CONNECTIVITY_ATTRS = {
    "cf_role": "node_face_connectivity",
    "long name": "Faces that neighbor each node",
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE,
}

NODE_FACE_CONNECTIVITY_DIMS = ["n_node", "n_max_node_faces"]


N_NODES_PER_FACE_ATTRS = {
    "cf_role": "n_nodes_per_face",
    "long name": "Number of nodes per face",
}

N_NODES_PER_FACE_DIMS = ["n_face"]


CONNECTIVITY_NAMES = [
    "face_node_connectivity",
    "face_edge_connectivity",
    "face_face_connectivity",
    "edge_node_connectivity",
    "edge_face_connectivity",
    "node_edge_connectivity",
    "node_face_connectivity",
]

# as of UGRID v1.0
UGRID_COMPLIANT_CONNECTIVITY_NAMES = [
    "edge_node_connectivity",
    "face_node_connectivity",
    "face_edge_connectivity",
    "edge_face_connectivity",
    "face_face_connectivity",
]
CONNECTIVITY = {
    "face_node_connectivity": {
        "dims": FACE_NODE_CONNECTIVITY_DIMS,
        "attrs": FACE_NODE_CONNECTIVITY_ATTRS,
    },
    "face_edge_connectivity": {
        "dims": FACE_EDGE_CONNECTIVITY_DIMS,
        "attrs": FACE_EDGE_CONNECTIVITY_ATTRS,
    },
    "face_face_connectivity": {
        "dims": FACE_FACE_CONNECTIVITY_DIMS,
        "attrs": FACE_FACE_CONNECTIVITY_ATTRS,
    },
    "edge_node_connectivity": {
        "dims": EDGE_NODE_CONNECTIVITY_DIMS,
        "attrs": EDGE_NODE_CONNECTIVITY_ATTRS,
    },
    # "edge_edge_connectivity": {
    #     "dims": EDGE_EDGE_CONNECTIVITY_DIMS,
    #     "attrs": EDGE_EDGE_CONNECTIVITY_ATTRS,
    # },
    "edge_face_connectivity": {
        "dims": EDGE_FACE_CONNECTIVITY_DIMS,
        "attrs": EDGE_FACE_CONNECTIVITY_ATTRS,
    },
    "node_edge_connectivity": {
        "dims": NODE_EDGE_CONNECTIVITY_DIMS,
        "attrs": NODE_EDGE_CONNECTIVITY_ATTRS,
    },
    "node_face_connectivity": {
        "dims": NODE_FACE_CONNECTIVITY_DIMS,
        "attrs": NODE_FACE_CONNECTIVITY_ATTRS,
    },
}

SPHERICAL_COORD_NAMES = [
    "node_lon",
    "node_lat",
    "edge_lon",
    "edge_lat",
    "face_lon",
    "face_lat",
]

SPHERICAL_COORDS = {
    "node_lon": {"dims": [NODE_DIM], "attrs": NODE_LON_ATTRS},
    "node_lat": {"dims": [NODE_DIM], "attrs": NODE_LAT_ATTRS},
    "edge_lon": {"dims": [EDGE_DIM], "attrs": EDGE_LON_ATTRS},
    "edge_lat": {"dims": [EDGE_DIM], "attrs": EDGE_LAT_ATTRS},
    "face_lon": {"dims": [FACE_DIM], "attrs": FACE_LON_ATTRS},
    "face_lat": {"dims": [FACE_DIM], "attrs": FACE_LAT_ATTRS},
}

CARTESIAN_COORD_NAMES = [
    "node_x",
    "node_y",
    "node_z",
    "edge_x",
    "edge_y",
    "edge_z",
    "face_x",
    "face_y",
    "face_z",
]

CARTESIAN_COORDS = {
    "node_x": {"dims": [NODE_DIM], "attrs": NODE_X_ATTRS},
    "node_y": {"dims": [NODE_DIM], "attrs": NODE_Y_ATTRS},
    "node_z": {"dims": [NODE_DIM], "attrs": NODE_Z_ATTRS},
    "edge_x": {"dims": [EDGE_DIM], "attrs": EDGE_X_ATTRS},
    "edge_y": {"dims": [EDGE_DIM], "attrs": EDGE_Y_ATTRS},
    "edge_z": {"dims": [EDGE_DIM], "attrs": EDGE_Z_ATTRS},
    "face_x": {"dims": [FACE_DIM], "attrs": FACE_X_ATTRS},
    "face_y": {"dims": [FACE_DIM], "attrs": FACE_Y_ATTRS},
    "face_z": {"dims": [FACE_DIM], "attrs": FACE_Z_ATTRS},
}
