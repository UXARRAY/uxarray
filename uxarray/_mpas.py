import xarray as xr
import numpy as np

# remove/edit once unified fill value approach is implemented
int_dtype = np.uint32
fill_val = np.iinfo(int_dtype).max


def _primal_to_ugrid(in_ds, out_ds):
    """If input dataset is an unstructured MPAS file, function will encode the
    MPAS Primal-Mesh in the UGRID conventions.

    Parameters
    ----------
    in_ds : xarray.Dataset
        Input MPAS dataset
    out_ds : xarray.Dataset
        Output dataset where the MPAS Primal-Mesh is encoded in the UGRID
        conventions
    """

    # corners of primal-mesh cells (in degrees)
    lonVertex = np.rad2deg(in_ds['lonVertex'].values)
    latVertex = np.rad2deg(in_ds['latVertex'].values)

    out_ds['Mesh2_node_x'] = xr.DataArray(
        lonVertex,
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "longitude",
            "long_name": "longitude of mesh nodes",
            "units": "degrees_east",
        })

    out_ds['Mesh2_node_y'] = xr.DataArray(
        latVertex,
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "latitude",
            "long_name": "latitude of mesh nodes",
            "units": "degrees_north",
        })

    # centers of primal-mesh cells (in degrees)
    lonCell = np.rad2deg(in_ds['lonCell'].values)
    latCell = np.rad2deg(in_ds['latCell'].values)

    out_ds['Mesh2_face_x'] = xr.DataArray(
        lonCell,
        dims=["nMesh2_face"],
        attrs={
            "standard_name": "longitude",
            "long_name": "longitude of center nodes",
            "units": "degrees_east",
        })

    out_ds['Mesh2_face_y'] = xr.DataArray(
        latCell,
        dims=["nMesh2_face"],
        attrs={
            "standard_name": "latitude",
            "long_name": "latitude of center nodes",
            "units": "degrees_north",
        })

    # vertex indices that surround each primal-mesh cell
    verticesOnCell = np.array(in_ds['verticesOnCell'].values, dtype=int_dtype)

    nEdgesOnCell = np.array(in_ds['nEdgesOnCell'].values, dtype=int_dtype)

    # replace padded values with fill values
    verticesOnCell = _replace_padding(verticesOnCell, nEdgesOnCell)

    # replace missing/zero values with fill values
    verticesOnCell = _replace_zeros(verticesOnCell)

    # convert to zero-indexed
    verticesOnCell = _to_zero_index(verticesOnCell)

    out_ds["Mesh2_face_nodes"] = xr.DataArray(
        data=verticesOnCell,
        dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
        attrs={
            "cf_role": "face_node_connectivity",
            "_FillValue": fill_val,
            "start_index": int_dtype(0)
        })

    # vertex indices that saddle a given edge
    verticesOnEdge = np.array(in_ds['verticesOnEdge'].values, dtype=int_dtype)

    # replace missing/zero values with fill value
    verticesOnEdge = _replace_zeros(verticesOnEdge)

    # convert to zero-indexed
    verticesOnEdge = _to_zero_index(verticesOnEdge)

    out_ds["Mesh2_edge_nodes"] = xr.DataArray(
        data=verticesOnEdge,
        dims=["nMesh2_edge", "Two"],
        attrs={
            "cf_role": "edge_node_connectivity",
            "start_index": int_dtype(0)
        })


def _dual_to_ugrid(in_ds, out_ds):
    """If input dataset is an unstructured MPAS file, function will encode the
    MPAS Dual-Mesh in the UGRID conventions.

    Parameters
    ----------
    in_ds : xarray.Dataset
        Input MPAS dataset
    out_ds : xarray.Dataset
        Output dataset where the MPAS Dual-Mesh is encoded in the UGRID
        conventions
    """

    # corners of dual-mesh cells (in degrees)
    lonCell = np.rad2deg(in_ds['lonCell'].values)
    latCell = np.rad2deg(in_ds['latCell'].values)

    out_ds['Mesh2_node_x'] = xr.DataArray(
        lonCell,
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "longitude",
            "long_name": "longitude of mesh nodes",
            "units": "degrees_east",
        })

    out_ds['Mesh2_node_y'] = xr.DataArray(
        latCell,
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "latitude",
            "long_name": "latitude of mesh nodes",
            "units": "degrees_north",
        })

    # centers of dual-mesh cells (in degrees)
    lonVertex = np.rad2deg(in_ds['lonVertex'].values)
    latVertex = np.rad2deg(in_ds['latVertex'].values)

    out_ds['Mesh2_face_x'] = xr.DataArray(
        lonVertex,
        dims=["nMesh2_face"],
        attrs={
            "standard_name": "longitude",
            "long_name": "longitude of center nodes",
            "units": "degrees_east",
        })

    out_ds['Mesh2_face_y'] = xr.DataArray(
        latVertex,
        dims=["nMesh2_face"],
        attrs={
            "standard_name": "latitude",
            "long_name": "latitude of center nodes",
            "units": "degrees_north",
        })

    # vertex indices that surround each dual-mesh cell
    cellsOnVertex = np.array(in_ds['cellsOnVertex'].values, dtype=int_dtype)

    # replace missing/zero values with fill values
    _replace_zeros(cellsOnVertex)

    # convert to zero-indexed
    _to_zero_index(cellsOnVertex)

    out_ds["Mesh2_face_nodes"] = xr.DataArray(
        data=cellsOnVertex,
        dims=["nMesh2_face", "Three"],
        attrs={
            "cf_role": "face_node_connectivity",
            "start_index": int_dtype(0)
        })

    # vertex indices that saddle a given edge
    cellsOnEdge = np.array(in_ds['cellsOnEdge'].values, dtype=int_dtype)

    # replace missing/zero values with fill values
    _replace_zeros(cellsOnEdge)

    # convert to zero-indexed
    _to_zero_index(cellsOnEdge)

    out_ds["Mesh2_edge_nodes"] = xr.DataArray(
        data=cellsOnEdge,
        dims=["nMesh2_edge", "Two"],
        attrs={
            "cf_role": "edge_node_connectivity",
            "start_index": int_dtype(0)
        })


def _replace_padding(verticesOnCell, nEdgesOnCell):
    """Replaces the padded values in verticesOnCell defined by nEdgesOnCell
    with a fill-value.

    Parameters
    ----------
    verticesOnCell : np.ndarray
        Vertex indices that surround a given cell

    nEdgesOnCell : np.ndarray
        Number of edges on a given cell

    Outputs
    ----------
    verticesOnCell : np.ndarray
        Vertex indices that surround a given cell with padded values replaced
        by fill values, done in-place
    """

    # max vertices/edges per cell
    maxEdges = verticesOnCell.shape[1]

    # iterate over the maximum number of vertices on a cell
    for vert_idx in range(maxEdges):
        # mask for non-padded values
        mask = vert_idx < nEdgesOnCell
        # replace remaining padding or zeros with fill_value
        verticesOnCell[np.logical_not(mask), vert_idx] = fill_val

    return verticesOnCell


def _replace_zeros(grid_var):
    """Replaces all instances of a zero (invalid/missing MPAS value) with a
    fill value.

    Parameters
    ----------
    grid_var : np.ndarray
        Grid variable that may contain zeros that need to be replaced

     Outputs
    ----------
    grid_var : np.ndarray
        Grid variable with zero replaced by fill values, done in-place
    """

    # one dimensional view of grid variable
    grid_var_flat = grid_var.ravel()

    # replace all zeros with a fill value
    grid_var_flat[grid_var_flat == 0] = fill_val

    return grid_var


def _to_zero_index(grid_var):
    """Given an input using that is one-indexed, subtracts one from all non-
    fill value entries to convert to zero-indexed.

    Parameters
    ----------
    grid_var : np.ndarray
        Grid variable that is one-indexed

     Outputs
    ----------
    grid_var : np.ndarray
        Grid variable that is converted to zero-indexed, done in-place
    """
    # one dimensional view of grid variable
    grid_var_flat = grid_var.ravel()

    # convert non-fill values to zero-indexed
    grid_var_flat[grid_var_flat != fill_val] -= 1

    return grid_var


def _read_mpas(ext_ds, use_dual=False):
    """Function to read in a MPAS Grid dataset and encode either the Primal or
    Dual mesh in the UGRID conventions.

    Parameters
    ----------
    ext_ds : xarray.Dataset, required
        MPAS datafile of interest
    use_dual : bool
        Flag to select whether to encode the Dual-Mesh

    Returns
    -------
    ds : xarray.Dataset
        ugrid aware :class:`xarray.Dataset`
    """

    # empty dataset that will contain our encoded MPAS mesh
    ds = xr.Dataset()

    try:
        # convert dual-mesh to UGRID
        if use_dual:
            _dual_to_ugrid(ext_ds, ds)
        # convert primal-mesh to UGRID
        else:
            _primal_to_ugrid(ext_ds, ds)

        ds["Mesh2"] = xr.DataArray(
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of unstructured mesh",
                "topology_dimension": 2,
                "node_coordinates": "Mesh2_node_x Mesh2_node_y",
                "node_dimension": "nMesh2_node",
                "face_node_connectivity": "Mesh2_face_nodes",
                "face_dimension": "nMesh2_face"
            })
    except:
        print("Variables not in recognized MPAS form. Please refer to",
              "https://mpas-dev.github.io/files/documents/MPAS-MeshSpec.pdf",
              "for more information on MPAS Grid file formatting")

    return ds
