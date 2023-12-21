from uxarray.constants import INT_FILL_VALUE

CONVENTIONS_ATTR = "UGRID-v1.0"

GRID_TOPOLOGY_ATTRS = {
    "cf_role": "mesh_topology",
    "topology_dimension": 2,
    # dimensions
    "face_dimension": "n_face",
    "node_dimension": "n_node",
    "edge_dimension": "n_edge",
    # coordinates
    "node_coordinates": "node_lon node_lat",
    "edge_coordinates": "edge_lon edge_lat",
    "face_coordinates": "face_lon face_lat",
    # connectivity
    "face_node_connectivity": "face_node_connectivity",
}

NODE_DIMS = ['n_node']
EDGE_DIMS = ['n_edge']
FACE_DIMS = ['n_face']

# coordinates
NODE_LON_ATTRS = {
    "standard_name": "longitude",
    "long name": "Longitude of Face Corner Nodes",
    "units": "degrees_east"
}

NODE_LAT_ATTRS = {
    "standard_name": "latitude",
    "long name": "Latitude of Face Corner Nodes",
    "units": "degrees_east"
}

EDGE_LON_ATTRS = {
    "standard_name": "longitude",
    "long name": "Longitude of Edge Centers",
    "units": "degrees_east"
}

EDGE_LAT_ATTRS = {
    "standard_name": "latitude",
    "long name": "Latitude of Edge Centers",
    "units": "degrees_east"
}

FACE_LON_ATTRS = {
    "standard_name": "longitude",
    "long name": "Longitude of Face Centers",
    "units": "degrees_east"
}

FACE_LAT_ATTRS = {
    "standard_name": "latitude",
    "long name": "Latitude of Face Centers",
    "units": "degrees_east"
}

# connectivity (face_)
FACE_NODE_CONNECTIVITY_ATTRS = {
    "cf_role": "face_node_connectivity",
    "long name": "Maps every face to its corner nodes.",
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE
}
FACE_NODE_CONNECTIVITY_DIMS = ["n_face", "n_max_face_nodes"]

FACE_EDGE_CONNECTIVITY_ATTRS = {
    "cf_role": "face_edge_connectivity",
    "long name": "Maps every face to its edges.",
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE
}
FACE_EDGE_CONNECTIVITY_DIMS = ["n_face", "n_max_face_nodes"]

FACE_FACE_CONNECTIVITY_ATTRS = {
    "cf_role": "face_face_connectivity",
    "long name": "Faces that neighbor each face.",
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE
}
FACE_FACE_CONNECTIVITY_DIMS = ["n_face", "n_max_face_nodes"]

# connectivity (edge_)
EDGE_NODE_CONNECTIVITY_ATTRS = {
    "cf_role": "edge_node_connectivity",
    "long name": "Maps every edge to the two nodes that it connects.",
    "start_index": 0,
}
EDGE_NODE_CONNECTIVITY_DIMS = ["n_edge", "n_max_face_nodes"]

# todo
# EDGE_EDGE_CONNECTIVITY_ATTRS = {
#     "cf_role": None,
#     "long name": None,
#     "start_index": None,
#     "_FillValue": None
# }
#
# EDGE_EDGE_CONNECTIVITY_DIMS = ['n_edge', 'n_max_face_nodes_two']

EDGE_FACE_CONNECTIVITY_ATTRS = {
    "cf_role": "edge_face_connectivity",
    "long name": "Faces that neighbor each edge.",
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE
}
EDGE_FACE_CONNECTIVITY_DIMS = ["n_edge", "two"]

# connectivity (node_)
NODE_NODE_CONNECTIVITY_ATTRS = {
    "cf_role": "node_node_connectivity",
    "long name": None,
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE
}

NODE_NODE_CONNECTIVITY_DIMS = ["n_node", "n_max_face_nodes"]

NODE_EDGE_CONNECTIVITY_ATTRS = {
    "cf_role": None,
    "long name": None,
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE
}

NODE_EDGE_CONNECTIVITY_DIMS = ["n_node", "n_max_face_nodes"]

NODE_FACE_CONNECTIVITY_ATTRS = {
    "cf_role": None,
    "long name": None,
    "start_index": 0,
    "_FillValue": INT_FILL_VALUE
}

NODE_FACE_CONNECTIVITY_DIMS = ["n_node", "n_max_face_nodes"]

CONNECTIVITY_NAMES = [
    "face_node_connectivity", "face_edge_connectivity",
    "face_face_connectivity", "edge_node_connectivity",
    "edge_edge_connectivity", "edge_face_connectivity",
    "node_node_connectivity", "node_edge_connectivity", "node_face_connectivity"
]

CONNECTIVITY = {
    "face_node_connectivity": {
        "dims": FACE_NODE_CONNECTIVITY_DIMS,
        "attrs": FACE_NODE_CONNECTIVITY_ATTRS
    },
    "face_edge_connectivity": {
        "dims": FACE_EDGE_CONNECTIVITY_DIMS,
        "attrs": FACE_EDGE_CONNECTIVITY_ATTRS
    },
    "face_face_connectivity": {
        "dims": FACE_FACE_CONNECTIVITY_DIMS,
        "attrs": FACE_FACE_CONNECTIVITY_ATTRS
    },
    "edge_node_connectivity": {
        "dims": EDGE_NODE_CONNECTIVITY_DIMS,
        "attrs": EDGE_NODE_CONNECTIVITY_ATTRS
    },
    # "edge_edge_connectivity": {"dims":  EDGE_EDGE_CONNECTIVITY_DIMS,
    #                            "attrs": EDGE_EDGE_CONNECTIVITY_ATTRS},
    "edge_face_connectivity": {
        "dims": EDGE_FACE_CONNECTIVITY_DIMS,
        "attrs": EDGE_FACE_CONNECTIVITY_ATTRS
    },
    "node_node_connectivity": {
        "dims": NODE_NODE_CONNECTIVITY_DIMS,
        "attrs": NODE_NODE_CONNECTIVITY_ATTRS
    },
    "node_edge_connectivity": {
        "dims": NODE_EDGE_CONNECTIVITY_DIMS,
        "attrs": NODE_EDGE_CONNECTIVITY_ATTRS
    },
    "node_face_connectivity": {
        "dims": NODE_FACE_CONNECTIVITY_DIMS,
        "attrs": NODE_FACE_CONNECTIVITY_ATTRS
    }
}

COORD_NAMES = [
    "node_lon", "node_lat", "edge_lon", "edge_lat", "face_lon", "face_lat"
]

COORDS = {
    "node_lon": {
        "dims": NODE_DIMS,
        "attrs": NODE_LON_ATTRS
    },
    "node_lat": {
        "dims": NODE_DIMS,
        "attrs": NODE_LAT_ATTRS
    },
    "edge_lon": {
        "dims": EDGE_DIMS,
        "attrs": EDGE_LON_ATTRS
    },
    "edge_lat": {
        "dims": EDGE_DIMS,
        "attrs": EDGE_LAT_ATTRS
    },
    "face_lon": {
        "dims": FACE_DIMS,
        "attrs": FACE_LON_ATTRS
    },
    "face_lat": {
        "dims": FACE_DIMS,
        "attrs": FACE_LAT_ATTRS
    },
}
