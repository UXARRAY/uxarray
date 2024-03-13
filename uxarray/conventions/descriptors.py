DESCRIPTOR_NAMES = ["face_areas", "edge_face_distances", "edge_node_distances"]


FACE_AREAS_DIMS = ["n_face"]

FACE_AREAS_ATTRS = {"cf_role": "face_areas"}

EDGE_FACE_DISTANCES_DIMS = ["n_edge"]
EDGE_FACE_DISTANCES_ATTRS = {
    "cf_role": "edge_face_distances",
    "long_name": "Distances between the face centers that " "saddle each edge",
}

EDGE_NODE_DISTANCES_DIMS = ["n_edge"]
EDGE_NODE_DISTANCES_ATTRS = {
    "cf_role": "edge_node_distances",
    "long_name": "Distances between the nodes that make up " "each edge.",
}
