import numpy as np
import xarray as xr

from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid
from uxarray.grid.coordinates import _lonlat_rad_to_xyz

# ``ERROR_TOLERANCE`` is defined as a Cartesian distance on the unit sphere; convert
# it to the angular tolerance `tol` is documented in so the default tracks the same
# precision assumption used everywhere else in the codebase.
_DEFAULT_STRUCTURED_TOL_DEG = np.rad2deg(2.0 * np.arcsin(ERROR_TOLERANCE / 2.0))


def _read_structured_grid(lon, lat, tol=_DEFAULT_STRUCTURED_TOL_DEG):
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
        Tolerance in degrees for considering nodes as identical. Defaults to the angle
        whose chord length on the unit sphere equals ``uxarray.constants.ERROR_TOLERANCE``,
        matching the precision assumption used elsewhere in the codebase.

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

    # Coincidence detection below relies on float64 precision (~1e-16); real-world
    # datasets often store lon/lat as float32 (~1e-7), which silently propagates
    # through this pipeline and causes pole/antimeridian merges to fail or merge
    # only partially, regardless of ``tol``.
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

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

    # Match nodes on the sphere rather than in the lon/lat plane, so that the poles
    # (many lon values, one point) and the antimeridian seam (lon differing by 360)
    # are recognized as coincident.
    lon_rad = np.deg2rad(node_lon)
    lat_rad = np.deg2rad(node_lat)
    node_xyz = np.column_stack(_lonlat_rad_to_xyz(lon_rad, lat_rad))

    # Build KDTree
    tree = KDTree(node_xyz)

    # ``tol`` is an angle in degrees; on the unit sphere the matching radius is the
    # chord subtended by that angle, so the threshold keeps its documented meaning.
    chord_tol = 2.0 * np.sin(np.deg2rad(tol) / 2.0)

    # Find all pairs of nodes within the tolerance
    pairs = tree.query_pairs(r=chord_tol)

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

    # No new faces are created here -- this only shrinks the width of existing rows
    # in face_node_conn for faces that became degenerate after the pole merge above.
    #
    # A face touching the pole is built from 2 distinct edge-longitudes at the pole
    # latitude, e.g. corners (n1, n2, n3, n4) = (A, P, P, B), where P is the single
    # merged pole node that both pole-row corners now point to (n2 == n3). That is a
    # triangle A-P-B stored as a 4-column quad with one corner repeated, so:
    #   1. `keep` marks, per face, which corners differ from their cyclic predecessor
    #      (n2 == n3 above means the P at position 2 is dropped from that row).
    #   2. The kept corners are pushed to the front of each row (`order`), giving
    #      (A, P, B, B) instead of (A, P, P, B) -- still 4 columns, but the last
    #      column is now the padding slot for a 3-node face.
    #   3. `n_max_face_nodes` is the largest node count any face still needs (3 here,
    #      unless some other face in the grid still has 4 distinct corners, in which
    #      case nothing is trimmed and this is a no-op). Columns beyond each face's
    #      own count are set to `INT_FILL_VALUE`, giving (A, P, B, FILL).
    keep = face_node_conn != np.roll(face_node_conn, 1, axis=1)
    if not keep.all():
        n_nodes_per_face = keep.sum(axis=1)
        order = np.argsort(~keep, axis=1, kind="stable")
        compacted = np.take_along_axis(face_node_conn, order, axis=1)
        n_max_face_nodes = n_nodes_per_face.max()
        compacted = compacted[:, :n_max_face_nodes]
        compacted[np.arange(n_max_face_nodes) >= n_nodes_per_face[:, None]] = (
            INT_FILL_VALUE
        )
        face_node_conn = compacted

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
