import numpy as np
import xarray as xr

from uxarray.constants import INT_DTYPE
from uxarray.conventions import ugrid


def _read_structured_grid(lon, lat, tol=1e-10):
    """
    Constructs an unstructured grid dataset from structured longitude and latitude coordinates.

    This function takes 1D arrays of longitude and latitude values, computes the grid edges,
    identifies unique grid nodes within a specified tolerance, and establishes connectivity
    between nodes to form grid faces. The resulting unstructured grid is returned as an
    `xarray.Dataset`, along with a dictionary of source dimensions.

    Parameters
    ----------
    lon : array_like
        1D array of longitude coordinates in degrees.
    lat : array_like
        1D array of latitude coordinates in degrees.
    tol : float, optional
        Tolerance for considering nodes as identical (default is `1e-10`).

    Returns
    -------
    out_ds : xr.Dataset
        An `xarray.Dataset` representing the structured grid encoded in the UGRID conventions


    Notes
    -----
    - The function first sorts the longitude values and computes the edges for both longitude
      and latitude to create a meshgrid of node coordinates.
    - A KDTree is used to identify and merge nodes that are within the specified tolerance.
    """

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import KDTree

    out_ds = xr.Dataset()

    sorted_indices = np.argsort(lon)
    lon = lon[sorted_indices]

    # Compute the edges for longitude
    lon_diff = np.diff(lon)
    lon_edges = np.concatenate(
        (
            [lon[0] - 0.5 * lon_diff[0]],
            lon[:-1] + 0.5 * lon_diff,
            [lon[-1] + 0.5 * lon_diff[-1]],
        )
    )

    # Sort the longitude edges
    sorted_edge_indices = np.argsort(lon_edges)
    lon_edges = lon_edges[sorted_edge_indices]

    # Compute the edges for latitude
    lat_diff = np.diff(lat)
    lat_edges = np.concatenate(
        (
            [lat[0] - 0.5 * lat_diff[0]],
            lat[:-1] + 0.5 * lat_diff,
            [lat[-1] + 0.5 * lat_diff[-1]],
        )
    )

    # Create a meshgrid of the edges
    node_lon_2d, node_lat_2d = np.meshgrid(lon_edges, lat_edges)

    # Flatten the 2D arrays to 1D arrays
    node_lon = node_lon_2d.ravel()
    node_lat = node_lat_2d.ravel()

    # Stack longitude and latitude for processing
    nodes = np.column_stack((node_lon, node_lat))

    # Build KDTree
    tree = KDTree(nodes)

    # Find all pairs of nodes within the tolerance
    pairs = tree.query_pairs(r=tol)

    n_nodes = len(nodes)
    if pairs:
        # Build adjacency matrix
        pairs_array = np.array(list(pairs))
        rows = pairs_array[:, 0]
        cols = pairs_array[:, 1]

        # Create symmetric adjacency matrix
        data = np.ones(len(rows) * 2)
        row = np.concatenate([rows, cols])
        col = np.concatenate([cols, rows])

        adj_matrix = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        # Use connected_components to find connected components
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False)
    else:
        # No pairs within tolerance, each node is its own component
        labels = np.arange(n_nodes)

    # Extract unique nodes based on the labels
    unique_labels, index = np.unique(labels, return_index=True)
    unique_nodes = nodes[index]
    unique_node_lon = unique_nodes[:, 0]
    unique_node_lat = unique_nodes[:, 1]

    # Update node_indices_unique using the labels
    node_indices_unique = labels.reshape((len(lat_edges), len(lon_edges)))

    # Define connectivity for unstructured grid
    num_cells_lon = len(lon)
    num_cells_lat = len(lat)

    # Create arrays of i and j indices
    ii = np.arange(num_cells_lon)
    ji = np.arange(num_cells_lat)

    # Create a meshgrid of cell indices
    i, j = np.meshgrid(ii, ji)

    # Flatten meshgrid
    i = i.ravel()
    j = j.ravel()

    # Calculate node indices for each face using the unique mapping
    n1 = node_indices_unique[j, i]
    n2 = node_indices_unique[j, i + 1]
    n3 = node_indices_unique[j + 1, i + 1]
    n4 = node_indices_unique[j + 1, i]

    # Stack the node indices to form face_node_connectivity
    face_node_conn = np.vstack((n1, n2, n3, n4), dtype=INT_DTYPE).T

    out_ds["node_lon"] = xr.DataArray(
        data=unique_node_lon, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LON_ATTRS
    )
    out_ds["node_lat"] = xr.DataArray(
        data=unique_node_lat, dims=ugrid.NODE_DIM, attrs=ugrid.NODE_LAT_ATTRS
    )

    out_ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_conn,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    return out_ds
