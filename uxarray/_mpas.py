import xarray as xr
import numpy as np
from pathlib import PurePath
from datetime import datetime

int_dtype = np.uint32
fill_val = np.iinfo(int_dtype).max


def _to_ugrid(in_ds, out_ds):
    """If input dataset (``in_ds``) file is an unstructured MPAS file, function
    will reassign MPAS variables to UGRID conventions in output file
    (``out_ds``).

    Parameters
    ----------
    in_ds : xarray.Dataset
        Original MPAS dataset of interest being used

    out_ds : xarray.Dataset
        Output dataset encoded in the UGRID conventions
    """

    # corners of primal mesh cells (in degrees)
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

    # centers of primal mesh cells (in degrees)
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

    # vertex indices that surround each primal mesh cell
    verticesOnCell = in_ds['verticesOnCell'].values

    # max number of edges & vertices that make up each cell
    nEdgesOnCell = in_ds['nEdgesOnCell'].values

    # correct to be zero-indexed and to use fill values
    verticesOnCell_fill = _add_fill_values(verticesOnCell, nEdgesOnCell)

    out_ds["Mesh2_face_nodes"] = xr.DataArray(
        data=verticesOnCell_fill,
        dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
        attrs={
            "cf_role": "face_node_connectivity",
            "_FillValue": fill_val,
            "start_index": int_dtype(0)
        })

    # add Mesh2_edge_nodes if available
    if "verticesOnEdge" in in_ds:
        # vertex indices that saddle a given edge
        verticesOnEdge = in_ds['verticesOnEdge'].values

        # convert to zero-indexed
        verticesOnEdge -= 1

        out_ds["Mesh2_edge_nodes"] = xr.DataArray(
            data=verticesOnEdge,
            dims=["nMesh2_edge", "Two"],
            attrs={
                "cf_role": "edge_node_connectivity",
                "start_index": int_dtype(0)
            })


def _add_fill_values(verticesOnCell, nEdgesOnCell):
    """Corrects the input (``verticesOnCell``) to be zero-indexed and to use
    fill-values instead of padding / zero-fills.

    Parameters
    ----------
    verticesOnCell : np.ndarray
        Vertex indices that surround a given cell

    nEdgesOnCell : np.ndarray
        Number of edges on a given cell
    """

    # convert to unsigned integers
    verticesOnCell = verticesOnCell.astype(int_dtype)

    # max vertices/edges per cell
    maxEdges = verticesOnCell.shape[1]

    # iterate over the maximum number of vertices on a cell
    for vert_idx in range(maxEdges):
        # mask for non-padded values
        mask = vert_idx < nEdgesOnCell
        # convert non-padded values to zero-index
        verticesOnCell[mask, vert_idx] -= 1
        # replace remaining padding or zeros with fill_value
        verticesOnCell[np.logical_not(mask), vert_idx] = fill_val

    return verticesOnCell


def _read_mpas(ext_ds):
    """Function to read in a MPAS dataset and encode the Primal mesh in the
    UGRID conventions.

    Parameters
    ----------
    ext_ds : xarray.Dataset, required
        MPAS datafile of interest

    Returns
    -------
    ds : xarray.Dataset
        ugrid aware :class:`xarray.Dataset`
    """
    ds = xr.Dataset()

    try:
        _to_ugrid(ext_ds, ds)
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
