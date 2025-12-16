from copy import deepcopy

import numpy as np

import uxarray.core.dataset

# To preserve old names for remapping
KDTREE_DIM_MAP: dict[str, str] = {
    "corner nodes": "node",
    "edge centers": "edge",
    "face centers": "face",
    "nodes": "node",
    "faces": "face",
    "edges": "edge",
    "node": "node",
    "edge": "edge",
    "face": "face",
    "n_face": "face",
    "n_edge": "edge",
    "n_node": "node",
}

# To preserve old names for remapping
LABEL_TO_COORD: dict[str, str] = {
    "corner nodes": "n_node",
    "edge centers": "n_edge",
    "face centers": "n_face",
    "nodes": "n_node",
    "faces": "n_face",
    "edges": "n_edge",
    "node": "n_node",
    "edge": "n_edge",
    "face": "n_face",
    "n_face": "n_face",
    "n_edge": "n_edge",
    "n_node": "n_node",
}

SPATIAL_DIMS = {"n_node", "n_edge", "n_face"}


def _assert_dimension(dim):
    """
    Validate that `dim` is a recognized label or spatial dimension.

    Parameters
    ----------
    dim : str
        The dimension or label to validate against `LABEL_TO_COORD`.

    Raises
    ------
    ValueError
        If `dim` is not a key in `LABEL_TO_COORD`.
    """
    if dim not in LABEL_TO_COORD:
        raise ValueError(f"Invalid spatial dimension: {dim!r}")


def _construct_remapped_ds(source, remapped_vars, destination_grid, destination_dim):
    """
    Construct a new UxDataset from remapped data variables and updated coordinates.

    Parameters
    ----------
    source : UxDataArray or UxDataset
        Original UXarray object used to extract non-spatial coordinates.
    remapped_vars : dict[str, xr.DataArray]
        Mapping of variable names to their remapped DataArrays.
    destination_grid : Grid
        The UXarray grid instance representing the new topology.
    destination_dim : str
        The spatial dimension name (e.g., 'n_face') for the destination grid.

    Returns
    -------
    UxDataset
        A new dataset containing only the remapped variables and retained coordinates.
    """
    destination_coords = deepcopy(source.coords)
    if destination_dim in destination_coords:
        del destination_coords[destination_dim]

    ds_remapped = uxarray.core.dataset.UxDataset(
        data_vars=remapped_vars,
        uxgrid=destination_grid,
        coords=destination_coords,
    )

    return ds_remapped


def _to_dataset(source):
    """
    Normalize input to a UxDataset for unified remapping logic.

    Parameters
    ----------
    source : UxDataArray or UxDataset
        The input object to convert.

    Returns
    -------
    ds : UxDataset
        The dataset representation of `source`.
    is_da : bool
        True if the original `source` was a UxDataArray.
    name : str or None
        The variable name of the original DataArray, or None for datasets.
    """
    if isinstance(source, uxarray.core.dataarray.UxDataArray):
        is_da = True
        name = source.name if source.name is not None else "nearest_neighbor_remap"
        ds = source.to_dataset(name=name) if is_da else source
    else:
        is_da = False
        ds = source
        name = None

    return ds, is_da, name


def _get_remap_dims(ds):
    """
    Identify which spatial dimensions are present in a dataset.

    Scans all data variables and returns the set of dimensions intersecting `SPATIAL_DIMS`.

    Parameters
    ----------
    ds : UxDataset
        The dataset to inspect.

    Returns
    -------
    dims_to_remap : set of str
        Spatial dimensions found in `ds` (subset of {'n_node','n_edge','n_face'}).

    Raises
    ------
    ValueError
        If no spatial dimensions are detected in the dataset.
    """
    dims_to_remap: set[str] = set()
    for da in ds.data_vars.values():
        dims_to_remap |= set(da.dims) & SPATIAL_DIMS

    if not dims_to_remap:
        raise ValueError(
            "No spatial dimensions (n_node, n_edge, or n_face) found in source to remap."
        )

    return dims_to_remap


def _prepare_points(grid, element_dim):
    """
    Gather 3D Cartesian coordinates for grid elements to query against.

    Parameters
    ----------
    grid : Grid
        The UXarray grid containing coordinate arrays.
    element_dim : str
        A label or key indicating which set of coordinates to use
        (mapped via `KDTREE_DIM_MAP`).

    Returns
    -------
    points : np.ndarray, shape (n_points, 3)
        Stack of x, y, z values for the specified grid elements.

    Raises
    ------
    ValueError
        If `element_dim` is not in `KDTREE_DIM_MAP`.
    """
    grid.normalize_cartesian_coordinates()
    element_dim = KDTREE_DIM_MAP[element_dim]
    return np.vstack(
        [
            getattr(grid, f"{element_dim}_x").values,
            getattr(grid, f"{element_dim}_y").values,
            getattr(grid, f"{element_dim}_z").values,
        ]
    ).T
