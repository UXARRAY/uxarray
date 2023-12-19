from uxarray.constants import INT_FILL_VALUE

GRID_TOPOLOGY_ATTRS = {
    "cf_role": "mesh_topology",
    "topology_dimension": 2,
    "node_coordinates": "node_lon node_lat",
    "face_node_connectivity": "face_node_connectivity",
    "face_dimension": "n_face"
}

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

FACE_EDGE_CONNECTIVITY_ATTRS = {
    "cf_role": None,
    "long name": None,
    "start_index": None,
    "_FillValue": None
}

FACE_FACE_CONNECTIVITY_ATTRS = {
    "cf_role": None,
    "long name": None,
    "start_index": None,
    "_FillValue": None
}

# connectivity (edge_)
EDGE_NODE_CONNECTIVITY_ATTRS = {
    "cf_role": None,
    "long name": None,
    "start_index": None,
    "_FillValue": None
}

EDGE_EDGE_CONNECTIVITY_ATTRS = {
    "cf_role": None,
    "long name": None,
    "start_index": None,
    "_FillValue": None
}

EDGE_FACE_CONNECTIVITY_ATTRS = {
    "cf_role": None,
    "long name": None,
    "start_index": None,
    "_FillValue": None
}

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
