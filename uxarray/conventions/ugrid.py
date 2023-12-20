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

EDGE_EDGE_CONNECTIVITY_ATTRS = {
    "cf_role": None,
    "long name": None,
    "start_index": None,
    "_FillValue": None
}

EDGE_FACE_CONNECTIVITY_ATTRS = {
    "cf_role": "edge_face_connectivity",
    "long name": "Faces that neighbor each edge.",
    "start_index": None,
    "_FillValue": None
}
EDGE_FACE_CONNECTIVITY_DIMS = ["n_edge", "two"]

# connectivity (node_)
NODE_NODE_CONNECTIVITY_ATTRS = {
    "cf_role": None,
    "long name": None,
    "start_index": None,
    "_FillValue": None
}

NODE_EDGE_CONNECTIVITY_ATTRS = {
    "cf_role": None,
    "long name": None,
    "start_index": None,
    "_FillValue": None
}

NODE_FACE_CONNECTIVITY_ATTRS = {
    "cf_role": None,
    "long name": None,
    "start_index": None,
    "_FillValue": None
}
