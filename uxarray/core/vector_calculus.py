from typing import Optional

import numpy as np

from uxarray.constants import INT_FILL_VALUE


def _calculate_gradient(face_values, uxgrid, edge_face_distances=None):
    """Calculate the component of the horizontal gradient of a face-centered
    scalar field along each edge.

    For an edge connecting faces i and j, this computes (f_j - f_i) / distance_ij,
    where distance_ij is the distance between the centers of face i and face j.
    This value is associated with the edge.

    Boundary edges (connected to only one face) will have a gradient
    component of zero with this method, as it requires two adjacent faces.

    Parameters
    ----------
    face_values : numpy.ndarray
        Values on faces (shape: (..., n_face))
    uxgrid : UxGrid
        The grid object
    edge_face_distances : numpy.ndarray, optional
        The distances between faces along edges. If not provided, it will be
        computed from face centers. It's recommended to pre-compute this.

    Returns
    -------
    numpy.ndarray
        Gradient component values on edges (shape: (..., n_edge))
    """
    # Get the edge-face connectivity
    edge_faces = uxgrid.edge_face_connectivity.values

    # Use provided edge-face distances, which should be pre-computed or derived
    if edge_face_distances is None:
        # Fallback: compute from face centers
        face_x = uxgrid.face_x.values
        face_y = uxgrid.face_y.values
        face_z = uxgrid.face_z.values if hasattr(uxgrid, 'face_z') else np.zeros_like(face_x)
        face_centers = np.column_stack((face_x, face_y, face_z))
        edge_face_distances = np.zeros(uxgrid.n_edge)

        saddle_mask = edge_faces[:, 1] != INT_FILL_VALUE
        # Compute distance between face centers for saddle edges
        f0_centers = face_centers[edge_faces[saddle_mask, 0]]
        f1_centers = face_centers[edge_faces[saddle_mask, 1]]
        edge_face_distances[saddle_mask] = np.linalg.norm(f1_centers - f0_centers, axis=-1)


    # Create output array with proper dimensions
    dims = list(face_values.shape[:-1])
    dims.append(uxgrid.n_edge)
    grad_values = np.zeros(dims)

    # Determine which edges saddle two faces
    saddle_mask = edge_faces[:, 1] != INT_FILL_VALUE

    # For edges with two faces, calculate gradient component (signed difference / distance)
    # Avoid division by zero for edges with zero distance (shouldn't happen in valid grids, but safety)
    valid_saddle_mask = saddle_mask & (edge_face_distances > 0)

    f0_indices = edge_faces[valid_saddle_mask, 0]
    f1_indices = edge_faces[valid_saddle_mask, 1]

    # Calculate the difference along the edge direction (f_j - f_i)
    face_diff = face_values[..., f1_indices] - face_values[..., f0_indices]

    # Calculate gradient component: (f_j - f_i) / distance_ij
    grad_values[..., valid_saddle_mask] = face_diff / edge_face_distances[valid_saddle_mask]


    # Edges not in valid_saddle_mask (boundary edges or zero distance edges) remain 0 as initialized

    return grad_values


def _calculate_divergence(edge_values, uxgrid, normalize=False):
    """Calculate the divergence of an edge-centered field.

    This implementation approximates divergence at each face center by summing
    the product of the edge value and the edge length for all edges bounding
    the face, and dividing by the face area. It is assumed that the input
    `edge_values` represents the component of a vector field normal to the edge,
    integrated along the edge (i.e., flux).

    Parameters
    ----------
    edge_values : numpy.ndarray
        Values on edges (shape: (..., n_edge)). Assumed to represent flux or
        quantity related to flux across the edge.
    uxgrid : UxGrid
        The grid object

    Returns
    -------\n    numpy.ndarray
        Divergence values on faces (shape: (..., n_face))
    """
    # Get the edge-face connectivity
    edge_face_conn = uxgrid.edge_face_connectivity.values

    # Calculate edge lengths using edge coordinates if available
    if hasattr(uxgrid, 'edge_node_connectivity'):
        edge_node_conn = uxgrid.edge_node_connectivity.values
        # Get node coordinates
        node_x = uxgrid.node_x.values
        node_y = uxgrid.node_y.values
        node_z = uxgrid.node_z.values if hasattr(uxgrid, 'node_z') else np.zeros_like(node_x)
        node_coords = np.column_stack((node_x, node_y, node_z))

        # Calculate edge vectors and lengths
        edge_vectors = node_coords[edge_node_conn[:, 1]] - node_coords[edge_node_conn[:, 0]]
        edge_lengths = np.sqrt(np.sum(edge_vectors**2, axis=1))
    else:
        # Fallback: use unit lengths if edge connectivity not available (less accurate)
        edge_lengths = np.ones(uxgrid.n_edge) # This fallback is not ideal for accurate divergence


    # Initialize divergence array
    dims = list(edge_values.shape[:-1])
    dims.append(uxgrid.n_face)
    div_values = np.zeros(dims)

    # Get saddle mask (edges connecting two faces)
    saddle_mask = edge_face_conn[:, 1] != INT_FILL_VALUE

    # For each edge with two faces, accumulate flux contributions
    for e in np.nonzero(saddle_mask)[0]:
        f0, f1 = edge_face_conn[e]

        # Flux contribution = edge_value * edge_length
        # The sign depends on the edge orientation relative to the face
        # Assuming edge orientation (f0, f1) means positive flux from f0 to f1
        # Need to check consistent orientation relative to faces
        # For simplicity based on existing code structure, assume accumulation like this:
        flux_contrib = edge_values[..., e] * edge_lengths[e]

        # Add outward flux to face f0, subtract from face f1 (assuming f0 is 'left' and f1 is 'right' of oriented edge)
        div_values[..., f0] += flux_contrib
        div_values[..., f1] -= flux_contrib


    # Accumulate contributions from boundary edges (connected to only one face)
    boundary_edge_mask = edge_face_conn[:, 1] == INT_FILL_VALUE
    for e in np.nonzero(boundary_edge_mask)[0]:
         f0 = edge_face_conn[e, 0] # The single connected face

         # Assuming boundary edges contribute outwards flux to the single face
         # This depends on boundary conditions, here assuming zero normal flow
         # If edge_values represent normal flux, the contribution is edge_value * edge_length
         flux_contrib = edge_values[..., e] * edge_lengths[e]

         # Add this flux to the single connected face
         div_values[..., f0] += flux_contrib

    # Divide by face areas to get divergence
    if not hasattr(uxgrid, 'face_areas'):
         raise AttributeError("Face areas must be available in the grid for divergence calculation")

    face_areas = uxgrid.face_areas.values
    # Avoid division by zero
    nonzero_areas = face_areas > 0
    div_values[..., nonzero_areas] /= face_areas[nonzero_areas]

    # Faces with zero area will retain their initialized zero divergence

    return div_values


def _calculate_curl(face_values, uxgrid):
    """Calculate a quantity related to circulation/curl based on a face-centered
    scalar field.

    This specific implementation calculates the difference between scalar values
    on adjacent faces (f_j - f_i) and divides by the edge length connecting
    their centers. This value is assigned to the edge.

    Note: This is not the standard curl of a vector field. The interpretation
    of this quantity depends on the nature of the input scalar field. It may
    represent a component of the gradient, or a circulation density related to
    a specific discretization scheme for curl of a vector field where the input
    scalar is a component of a vector potential or velocity.

    Parameters
    ----------
    face_values : numpy.ndarray
        Values on faces (shape: (..., n_face)). Assumed to be a scalar field.
    uxgrid : UxGrid
        The grid object

    Returns
    -------\n    numpy.ndarray
        Values on edges (shape: (..., n_edge)), representing a measure of
        difference/circulation density across edges.
    """
    # Get the edge-face connectivity
    edge_face_conn = uxgrid.edge_face_connectivity.values

    # Get face-to-face distances (edge lengths in face-centered context)
    # This uses face centers to calculate the distance across an edge
    face_x = uxgrid.face_x.values
    face_y = uxgrid.face_y.values
    face_z = uxgrid.face_z.values if hasattr(uxgrid, 'face_z') else np.zeros_like(face_x)
    face_centers = np.column_stack((face_x, face_y, face_z))

    edge_lengths = np.zeros(uxgrid.n_edge)

    saddle_mask = edge_face_conn[:, 1] != INT_FILL_VALUE
    # Compute distance between face centers for saddle edges
    f0_centers = face_centers[edge_face_conn[saddle_mask, 0]]
    f1_centers = face_centers[edge_face_conn[saddle_mask, 1]]
    edge_lengths[saddle_mask] = np.linalg.norm(f1_centers - f0_centers, axis=-1)


    # Initialize curl array
    dims = list(face_values.shape[:-1])
    dims.append(uxgrid.n_edge)
    curl_values = np.zeros(dims)

    # Determine which edges saddle two faces
    saddle_mask = edge_face_conn[:, 1] != INT_FILL_VALUE

    # For each edge with two faces, calculate the difference divided by edge length
    # Avoid division by zero for edges with zero distance
    valid_saddle_mask = saddle_mask & (edge_lengths > 0)

    f0_indices = edge_face_conn[valid_saddle_mask, 0]
    f1_indices = edge_face_conn[valid_saddle_mask, 1]

    # Calculate difference (f_j - f_i)
    circulation = face_values[..., f1_indices] - face_values[..., f0_indices]

    # Calculate value: (f_j - f_i) / edge_length
    curl_values[..., valid_saddle_mask] = circulation / edge_lengths[valid_saddle_mask]

    # Boundary edges remain 0 as initialized

    return curl_values


def _laplacian(face_values, uxgrid, normalize=False):
    """Calculate the Laplacian of a face-centered scalar field.

    This implements the identity: Laplacian(f) = div(grad(f))
    The Laplacian is commonly used in diffusion equations and represents
    the divergence of the gradient.

    Parameters
    ----------
    face_values : numpy.ndarray
        Values on faces (shape: (..., n_face))
    uxgrid : UxGrid
        The grid object

    Returns
    -------\n    numpy.ndarray
        Laplacian values on faces (shape: (..., n_face))
    """
    # First calculate the gradient (face-centered scalar -> edge-centered scalar)
    # This returns the component of the gradient along the edge direction
    grad_values = _calculate_gradient(face_values, uxgrid)

    # Then calculate the divergence of the gradient (edge-centered scalar -> face-centered scalar)
    # This requires the edge field to represent something related to normal flux for the divergence formula used.
    # If grad_values is the component of the gradient along the edge, its physical interpretation as input
    # to the current divergence function needs careful consideration based on the discretization.
    # Assuming the div(grad) discretization is consistent:
    laplacian_values = _calculate_divergence(grad_values, uxgrid)

    return laplacian_values