import xarray as xr
import numpy as np
import warnings

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


def _primal_to_ugrid(in_ds, out_ds):
    """Encodes the MPAS Primal-Mesh in the UGRID conventions.

    Parameters
    ----------
    in_ds : xarray.Dataset
        Input MPAS dataset
    out_ds : xarray.Dataset
        Output dataset where the MPAS Primal-Mesh is encoded in the UGRID
        conventions
    """

    source_dims_dict = {}

    # set mesh topologys
    out_ds["Mesh2"] = xr.DataArray(
        attrs={
            "cf_role": "mesh_topology",
            "long_name": "Topology data of unstructured mesh",
            "topology_dimension": 2,
            "node_coordinates": "Mesh2_node_x Mesh2_node_y",
            "node_dimension": "nMesh2_node",
            "face_node_connectivity": "Mesh2_face_nodes",
            "face_dimension": "nMesh2_face"
        })

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
    verticesOnCell = np.array(in_ds['verticesOnCell'].values, dtype=INT_DTYPE)

    nEdgesOnCell = np.array(in_ds['nEdgesOnCell'].values, dtype=INT_DTYPE)

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
            "_FillValue": INT_FILL_VALUE,
            "start_index": INT_DTYPE(0)
        })

    # vertex indices that saddle a given edge
    verticesOnEdge = np.array(in_ds['verticesOnEdge'].values, dtype=INT_DTYPE)

    # replace missing/zero values with fill value
    verticesOnEdge = _replace_zeros(verticesOnEdge)

    # convert to zero-indexed
    verticesOnEdge = _to_zero_index(verticesOnEdge)

    out_ds["Mesh2_edge_nodes"] = xr.DataArray(
        data=verticesOnEdge,
        dims=["nMesh2_edge", "Two"],
        attrs={
            "cf_role": "edge_node_connectivity",
            "start_index": INT_DTYPE(0)
        })

    # set global attributes
    _set_global_attrs(in_ds, out_ds)

    # populate source dims
    source_dims_dict['nVertices'] = 'nMesh2_node'
    source_dims_dict[in_ds['verticesOnCell'].dims[0]] = 'nMesh2_face'
    source_dims_dict[in_ds['verticesOnCell'].dims[1]] = 'nMaxMesh2_face_nodes'
    source_dims_dict[in_ds['verticesOnEdge'].dims[0]] = "nMesh2_edge"

    return source_dims_dict


def _dual_to_ugrid(in_ds, out_ds):
    """Encodes the MPAS Dual-Mesh in the UGRID conventions.

    Parameters
    ----------
    in_ds : xarray.Dataset
        Input MPAS dataset
    out_ds : xarray.Dataset
        Output dataset where the MPAS Dual-Mesh is encoded in the UGRID
        conventions
    """

    source_dims_dict = {}

    # set mesh topology
    out_ds["Mesh2"] = xr.DataArray(
        attrs={
            "cf_role": "mesh_topology",
            "long_name": "Topology data of unstructured mesh",
            "topology_dimension": 2,
            "node_coordinates": "Mesh2_node_x Mesh2_node_y",
            "node_dimension": "nMesh2_node",
            "face_node_connectivity": "Mesh2_face_nodes",
            "face_dimension": "nMesh2_face"
        })

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
    cellsOnVertex = np.array(in_ds['cellsOnVertex'].values, dtype=INT_DTYPE)

    # replace missing/zero values with fill values
    _replace_zeros(cellsOnVertex)

    # convert to zero-indexed
    _to_zero_index(cellsOnVertex)

    out_ds["Mesh2_face_nodes"] = xr.DataArray(
        data=cellsOnVertex,
        dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
        attrs={
            "cf_role": "face_node_connectivity",
            "_FillValue": INT_FILL_VALUE,
            "start_index": INT_DTYPE(0)
        })

    # vertex indices that saddle a given edge
    cellsOnEdge = np.array(in_ds['cellsOnEdge'].values, dtype=INT_DTYPE)

    # replace missing/zero values with fill values
    _replace_zeros(cellsOnEdge)

    # convert to zero-indexed
    _to_zero_index(cellsOnEdge)

    out_ds["Mesh2_edge_nodes"] = xr.DataArray(
        data=cellsOnEdge,
        dims=["nMesh2_edge", "Two"],
        attrs={
            "cf_role": "edge_node_connectivity",
            "start_index": INT_DTYPE(0)
        })

    # set global attributes
    _set_global_attrs(in_ds, out_ds)

    # populate source dims
    source_dims_dict[in_ds['latCell'].dims[0]] = "nMesh2_node"
    source_dims_dict[in_ds['cellsOnVertex'].dims[0]] = "nMesh2_face"
    source_dims_dict[in_ds['cellsOnVertex'].dims[1]] = "nMaxMesh2_face_nodes"
    source_dims_dict[in_ds['cellsOnEdge'].dims[0]] = "nMesh2_edge"

    return source_dims_dict


def _set_global_attrs(in_ds, out_ds):
    """Helper to set MPAS global attributes.

    Parameters
    ----------
    in_ds : xarray.Dataset
        Input MPAS dataset
    out_ds : xarray.Dataset
        Output dataset where the MPAS Primal-Mesh is encoded in the UGRID
        conventions with global attributes included
    """

    # defines if the mesh describes points that lie on the surface of a sphere or not
    if 'sphere_radius' in in_ds.attrs:
        out_ds.attrs['sphere_radius'] = in_ds.sphere_radius
    else:
        warnings.warn("Missing Required Attribute: 'sphere_radius'")

    # typically a random string used for tracking mesh provenance
    if 'mesh_id' in in_ds.attrs:
        out_ds.attrs['mesh_id'] = in_ds.mesh_id
    # else:
    #     warnings.warn("Missing Required Attribute: 'mesh_id'")

    # defines the version of the MPAS Mesh specification the mesh conforms to
    if 'mesh_spec' in in_ds.attrs:
        out_ds.attrs['mesh_spec'] = in_ds.mesh_spec
    # else:
    #     warnings.warn("Missing Required Attribute: 'mesh_spec'")

    # defines if the mesh describes points that lie on the surface of a sphere or not
    if "on_a_sphere" in in_ds.attrs:
        out_ds.attrs['on_a_sphere'] = in_ds.on_a_sphere
        # required attributes if mesh does not lie on a sphere
        if in_ds.on_a_sphere == "NO":
            # defines if the mesh has any periodic boundaries
            if "is_periodic" in in_ds.attrs:
                out_ds.attrs['is_periodic'] = in_ds.is_periodic
                if in_ds.is_periodic == "YES":
                    # period of the mesh in the x direction
                    if "x_period" in in_ds.attrs:
                        out_ds.attrs['x_period'] = in_ds.x_period
                    else:
                        warnings.warn("Missing Required Attribute: 'x_period'")
                    # period of the mesh in the y direction
                    if "y_period" in in_ds.attrs:
                        out_ds.attrs['y_period'] = in_ds.y_period
                    else:
                        warnings.warn("Missing Required Attribute: 'y_period'")
    else:
        warnings.warn("Missing Required Attribute: 'on_a_sphere'")


def _replace_padding(verticesOnCell, nEdgesOnCell):
    """Replaces the padded values in verticesOnCell defined by nEdgesOnCell
    with a fill-value.

    Parameters
    ----------
    verticesOnCell : numpy.ndarray
        Vertex indices that surround a given cell

    nEdgesOnCell : numpy.ndarray
        Number of edges on a given cell

    Returns
    -------
    verticesOnCell : numpy.ndarray
        Vertex indices that surround a given cell with padded values replaced
        by fill values, done in-place
    """

    # max vertices/edges per cell
    maxEdges = verticesOnCell.shape[1]

    # mask for non-padded values
    mask = np.arange(maxEdges) < nEdgesOnCell[:, None]

    # replace remaining padding or zeros with INT_FILL_VALUE
    verticesOnCell[np.logical_not(mask)] = INT_FILL_VALUE

    return verticesOnCell


def _replace_zeros(grid_var):
    """Replaces all instances of a zero (invalid/missing MPAS value) with a
    fill value.

    Parameters
    ----------
    grid_var : numpy.ndarray
        Grid variable that may contain zeros that need to be replaced

    Returns
    -------
    grid_var : numpy.ndarray
        Grid variable with zero replaced by fill values, done in-place
    """

    # replace all zeros with INT_FILL_VALUE
    grid_var[grid_var == 0] = INT_FILL_VALUE

    return grid_var


def _to_zero_index(grid_var):
    """Given an input using that is one-indexed, subtracts one from all non-
    fill value entries to convert to zero-indexed.

    Parameters
    ----------
    grid_var : numpy.ndarray
        Grid variable that is one-indexed

    Returns
    -------
    grid_var : numpy.ndarray
        Grid variable that is converted to zero-indexed, done in-place
    """

    # convert non-fill values to zero-indexed
    grid_var[grid_var != INT_FILL_VALUE] -= 1

    return grid_var


def _read_mpas(ext_ds, use_dual=False):
    """Function to read in a MPAS Grid dataset and encode either the Primal or
    Dual Mesh in the UGRID conventions.

    Adheres to the MPAS Mesh Specifications outlined in the following document:
    https://mpas-dev.github.io/files/documents/MPAS-MeshSpec.pdf

    Parameters
    ----------
    ext_ds : xarray.Dataset, required
        MPAS datafile of interest
    use_dual : bool, optional
        Flag to select whether to encode the Dual-Mesh. Defaults to False

    Returns
    -------
    ds : xarray.Dataset
        UGRID dataset derived from inputted MPAS dataset
    """

    # empty dataset that will contain our encoded MPAS mesh
    ds = xr.Dataset()

    # convert dual-mesh to UGRID
    if use_dual:
        source_dim_map = _dual_to_ugrid(ext_ds, ds)
    # convert primal-mesh to UGRID
    else:
        source_dim_map = _primal_to_ugrid(ext_ds, ds)

    return ds, source_dim_map
