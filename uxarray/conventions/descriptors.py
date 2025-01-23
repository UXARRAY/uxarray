DESCRIPTOR_NAMES = [
    "face_areas",
    "n_nodes_per_face",
    "edge_face_distances",
    "edge_node_distances",
    "boundary_edge_indices",
    "boundary_face_indices",
    "boundary_node_indices",
]


FACE_AREAS_DIMS = ["n_face"]

FACE_AREAS_ATTRS = {"cf_role": "face_areas", "long_name": "Area of each face."}


# TODO: add n_nodes_per_face


EDGE_FACE_DISTANCES_DIMS = ["n_edge"]
EDGE_FACE_DISTANCES_ATTRS = {
    "cf_role": "edge_face_distances",
    "long_name": "Distances between the face centers that saddle each edge",
}

EDGE_NODE_DISTANCES_DIMS = ["n_edge"]
EDGE_NODE_DISTANCES_ATTRS = {
    "cf_role": "edge_node_distances",
    "long_name": "Distances between the nodes that make up each edge.",
}

HOLE_EDGE_INDICES_DIMS = ["n_edge"]
HOLE_EDGE_INDICES_ATTRS = {
    "cf_role": "hole_edge_indices",
    "long_name": "Indices of edges that border a region of the grid not covered by any geometry",
}
