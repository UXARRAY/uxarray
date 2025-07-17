from __future__ import annotations

import os
import sys
from html import escape
from typing import IO, Any, Optional, Union
from warnings import warn

import numpy as np
import xarray as xr
from xarray.core import dtypes
from xarray.core.options import OPTIONS
from xarray.core.utils import UncachedAccessor

import uxarray
from uxarray.core.dataarray import UxDataArray
from uxarray.core.utils import _map_dims_to_ugrid
from uxarray.formatting_html import dataset_repr
from uxarray.grid import Grid
from uxarray.grid.dual import construct_dual
from uxarray.grid.validation import _check_duplicate_nodes_indices
from uxarray.io._healpix import get_zoom_from_cells
from uxarray.plot.accessor import UxDatasetPlotAccessor
from uxarray.remap.accessor import RemapAccessor


class UxDataset(xr.Dataset):
    """Grid informed ``xarray.Dataset`` with an attached ``Grid`` accessor and
    grid-specific functionality.

    Parameters
    ----------
    uxgrid : uxarray.Grid, optional
        The ``Grid`` object that makes this array aware of the unstructured
        grid topology it belongs to.

        If ``None``, it needs to be an instance of ``uxarray.Grid``.

    Other Parameters
    ----------------
    *args:
        Arguments for the ``xarray.Dataset`` class
    **kwargs:
        Keyword arguments for the ``xarray.Dataset`` class

    Notes
    -----
    See `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`__
    for further information about Datasets.
    """

    # expected instance attributes, required for subclassing with xarray (as of v0.13.0)
    __slots__ = (
        "_uxgrid",
        "_source_datasets",
    )

    def __init__(
        self,
        *args,
        uxgrid: Grid = None,
        source_datasets: Optional[str] = None,
        **kwargs,
    ):
        self._uxgrid = None
        self._source_datasets = source_datasets
        # setattr(self, 'source_datasets', source_datasets)

        if uxgrid is not None and not isinstance(uxgrid, Grid):
            raise RuntimeError(
                "uxarray.UxDataset.__init__: uxgrid can be either None or "
                "an instance of the `uxarray.Grid` class"
            )
        else:
            self._uxgrid = uxgrid

        super().__init__(*args, **kwargs)

    # declare plotting accessor
    plot = UncachedAccessor(UxDatasetPlotAccessor)
    remap = UncachedAccessor(RemapAccessor)

    def _repr_html_(self) -> str:
        if OPTIONS["display_style"] == "text":
            return f"<pre>{escape(repr(self))}</pre>"
        return dataset_repr(self)

    def __getitem__(self, key):
        """Override to make sure the result is an instance of
        ``uxarray.UxDataArray`` or ``uxarray.UxDataset``."""

        value = super().__getitem__(key)

        if isinstance(value, xr.DataArray):
            value = UxDataArray(value, uxgrid=self.uxgrid)
        elif isinstance(value, xr.Dataset):
            value = UxDataset(
                value, uxgrid=self.uxgrid, source_datasets=self.source_datasets
            )

        return value

    # def __setitem__(self, key, value):
    #     """Override to make sure the `value` is an instance of
    #     ``uxarray.UxDataArray``."""
    #     if isinstance(value, xr.DataArray):
    #         value = UxDataArray(value, uxgrid=self.uxgrid)
    #
    #     if isinstance(value, UxDataArray):
    #         value = value.to_dataarray()
    #
    #     super().__setitem__(key, value)

    @property
    def source_datasets(self):
        """Property to keep track of the source data sets used to instantiate
        this ``uxarray.UxDataset``.

        Can be used as metadata for diagnosis purposes.

        Examples
        --------
        uxds = ux.open_dataset(grid_path, data_path)
        uxds.source_datasets
        """
        return self._source_datasets

    # a setter function
    @source_datasets.setter
    def source_datasets(self, source_datasets_input):
        self._source_datasets = source_datasets_input

    @property
    def uxgrid(self):
        """Linked ``Grid`` representing to the unstructured grid the data
        resides on."""
        return self._uxgrid

    # a setter function
    @uxgrid.setter
    def uxgrid(self, ugrid_obj):
        self._uxgrid = ugrid_obj

    def _calculate_binary_op(self, *args, **kwargs):
        """Override to make the result a complete instance of
        ``uxarray.UxDataset``."""
        ds = super()._calculate_binary_op(*args, **kwargs)

        if isinstance(ds, UxDataset):
            ds.uxgrid = self.uxgrid
            ds.source_datasets = self.source_datasets
        else:
            ds = UxDataset(ds, uxgrid=self.uxgrid, source_datasets=self.source_datasets)

        return ds

    def _construct_dataarray(self, name) -> UxDataArray:
        """Override to make the result an instance of
        ``uxarray.UxDataArray``."""
        xarr = super()._construct_dataarray(name)
        return UxDataArray(xarr, uxgrid=self.uxgrid)

    @classmethod
    def _construct_direct(cls, *args, **kwargs):
        """Override to make the result an ``uxarray.UxDataset`` class."""

        return cls(xr.Dataset._construct_direct(*args, **kwargs))

    def _copy(self, **kwargs):
        """Override to make the result a complete instance of
        ``uxarray.UxDataset``."""
        copied = super()._copy(**kwargs)

        deep = kwargs.get("deep", None)

        if deep:
            # Reinitialize the uxgrid assessor
            copied.uxgrid = self.uxgrid.copy()  # deep copy
        else:
            # Point to the existing uxgrid object
            copied.uxgrid = self.uxgrid

        return copied

    def _replace(self, *args, **kwargs):
        """Override to make the result a complete instance of
        ``uxarray.UxDataset``."""
        ds = super()._replace(*args, **kwargs)

        if isinstance(ds, UxDataset):
            ds.uxgrid = self.uxgrid
            ds.source_datasets = self.source_datasets
        else:
            ds = UxDataset(ds, uxgrid=self.uxgrid, source_datasets=self.source_datasets)

        return ds

    @classmethod
    def from_dataframe(cls, dataframe):
        """Override to make the result a ``uxarray.UxDataset`` class."""

        return cls(
            {col: ("index", dataframe[col].values) for col in dataframe.columns},
            coords={"index": dataframe.index},
        )

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Override to make the result a ``uxarray.UxDataset`` class."""

        return cls(
            {key: ("index", val) for key, val in data.items()},
            coords={"index": range(len(next(iter(data.values()))))},
            **kwargs,
        )

    @classmethod
    def from_structured(cls, ds: xr.Dataset):
        """Converts a structured ``xarray.Dataset`` into an unstructured ``uxarray.UxDataset``

        Parameters
        ----------
        ds : xr.Dataset
            The structured `xarray.Dataset` to convert. Must contain longitude and latitude variables consistent
            with the CF-conventions

        tol : float, optional
            Tolerance for considering nodes as identical when constructing the grid from longitude and latitude.
            Default is `1e-10`.

        Returns
        -------
        UxDataset
            An instance of `uxarray.UxDataset`
        """
        from uxarray import Grid

        uxgrid = Grid.from_dataset(ds)

        ds = _map_dims_to_ugrid(ds, uxgrid._source_dims_dict, uxgrid)

        # Drop spatial coordinates
        coords_to_drop = [
            coord for coord, da_coord in ds.coords.items() if "n_face" in da_coord.dims
        ]
        ds = ds.drop_vars(coords_to_drop)

        return cls(ds, uxgrid=uxgrid)

    @classmethod
    def from_xarray(cls, ds: xr.Dataset, uxgrid: Grid = None, ugrid_dims: dict = None):
        """
        Converts a ``xarray.Dataset`` into a ``uxarray.UxDataset``, paired with either a user-defined or
        parsed ``Grid``

        Parameters
        ----------
        ds: xr.Dataset
            An Xarray dataset containing data residing on an unstructured grid
        uxgrid: Grid, optional
            ``Grid`` object representing an unstructured grid. If a grid is not provided, the source ds will be
            parsed to see if a ``Grid`` can be constructed.
        ugrid_dims: dict, optional
            A dictionary mapping dataset dimensions to UGRID dimensions.

        Returns
        -------
        cls
            A ``ux.UxDataset`` with data from the ``xr.Dataset` paired with a ``ux.Grid``
        """
        if uxgrid is not None:
            if ugrid_dims is None and uxgrid._source_dims_dict is not None:
                ugrid_dims = uxgrid._source_dims_dict
            # Grid is provided,
        else:
            # parse
            uxgrid = Grid.from_dataset(ds)
            ugrid_dims = uxgrid._source_dims_dict

        # map each dimension to its UGRID equivalent
        ds = _map_dims_to_ugrid(ds, ugrid_dims, uxgrid)

        return cls(ds, uxgrid=uxgrid)

    @classmethod
    def from_healpix(
        cls,
        ds: Union[str, os.PathLike, xr.Dataset],
        pixels_only: bool = True,
        face_dim: str = "cell",
        **kwargs,
    ):
        """
        Loads a dataset represented in the HEALPix format into a ``ux.UxDataSet``, paired
        with a ``Grid`` containing information about the HEALPix definition.

        Parameters
        ----------
        ds: str, os.PathLike, xr.Dataset
            Reference to a HEALPix Dataset
        pixels_only : bool, optional
            Whether to only compute pixels (`face_lon`, `face_lat`) or to also construct boundaries (`face_node_connectivity`, `node_lon`, `node_lat`)
        face_dim: str, optional
            Data dimension corresponding to the HEALPix face mapping. Typically, is set to "cell", but may differ.

        Returns
        -------
        cls
            A ``ux.UxDataset`` instance
        """

        if not isinstance(ds, xr.Dataset):
            ds = xr.open_dataset(ds, **kwargs)

        if face_dim not in ds.dims:
            raise ValueError(
                f"The provided face dimension '{face_dim}' is present in the provided healpix dataset."
                f"Please set 'face_dim' to the dimension corresponding to the healpix face dimension."
            )

        # Attach a HEALPix Grid
        uxgrid = Grid.from_healpix(
            zoom=get_zoom_from_cells(ds.sizes[face_dim]),
            pixels_only=pixels_only,
            **kwargs,
        )

        return cls.from_xarray(ds, uxgrid, {face_dim: "n_face"})

    def info(self, buf: IO = None, show_attrs=False) -> None:
        """Concise summary of Dataset variables and attributes including grid
        topology information stored in the ``uxgrid`` property.

        Parameters
        ----------
        buf : file-like, default: sys.stdout
            writable buffer
        show_attrs : bool
            Flag to select whether to show attributes

        See Also
        --------
        pandas.DataFrame.assign
        ncdump : netCDF's ncdump
        """
        if buf is None:  # pragma: no cover
            buf = sys.stdout

        lines = []
        lines.append("uxarray.Dataset {")

        lines.append("grid topology dimensions:")
        for name, size in self.uxgrid._ds.sizes.items():
            lines.append(f"\t{name} = {size}")

        lines.append("\ngrid topology variables:")
        for name, da in self.uxgrid._ds.variables.items():
            dims = ", ".join(map(str, da.dims))
            lines.append(f"\t{da.dtype} {name}({dims})")
            if show_attrs:
                for k, v in da.attrs.items():
                    lines.append(f"\t\t{name}:{k} = {v}")

        lines.append("\ndata dimensions:")
        for name, size in self.sizes.items():
            lines.append(f"\t{name} = {size}")

        lines.append("\ndata variables:")
        for name, da in self.variables.items():
            dims = ", ".join(map(str, da.dims))
            lines.append(f"\t{da.dtype} {name}({dims})")
            if show_attrs:
                for k, v in da.attrs.items():
                    lines.append(f"\t\t{name}:{k} = {v}")

        if show_attrs:
            lines.append("\nglobal attributes:")
            for k, v in self.attrs.items():
                lines.append(f"\t:{k} = {v}")

        lines.append("}")
        buf.write("\n".join(lines))

    def integrate(self, quadrature_rule="triangular", order=4):
        """Integrates over all the faces of the givfen mesh.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Calculated integral : float

        Examples
        --------
        Open a Uxarray dataset

        >>> import uxarray as ux
        >>> uxds = ux.open_dataset("grid.ug", "centroid_pressure_data_ug")

        # Compute the integral
        >>> integral = uxds.integrate()
        """

        # TODO: Deprecation Warning
        warn(
            "This method currently only works when there is a single DataArray in this Dataset. For integration of a "
            "single data variable, use the UxDataArray.integrate() method instead. This function will be deprecated and "
            "replaced with one that can perform a Dataset-wide integration in a future release.",
            DeprecationWarning,
        )

        integral = 0.0

        # call function to get area of all the faces as a np array
        face_areas, face_jacobian = self.uxgrid.compute_face_areas(
            quadrature_rule, order
        )

        # TODO: Should we fix this requirement? Shouldn't it be applicable to
        # TODO: all variables of dataset or a dataarray instead?
        var_key = list(self.keys())
        if len(var_key) > 1:
            # warning: print message
            print(
                "WARNING: The dataset has more than one variable, using the first variable for integration"
            )

        var_key = var_key[0]
        face_vals = self[var_key].to_numpy()
        integral = np.dot(face_areas, face_vals)

        return integral

    def to_array(self) -> UxDataArray:
        """Override to make the result an instance of
        ``uxarray.UxDataArray``."""

        xarr = super().to_array()
        return UxDataArray(xarr, uxgrid=self.uxgrid)

    def to_xarray(self, grid_format: str = "UGRID") -> xr.Dataset:
        """
        Converts a ``ux.UXDataset`` to a ``xr.Dataset`` in the specified grid format.

        Parameters
        ----------
        grid_format : str, default="UGRID"
            The format in which to convert the grid. Supported values are:
            - "UGRID": UGRID conventions format (default)
            - "HEALPix": HEALPix grid format
            - "ESMF": ESMF Unstructured Grid Format
            - "SCRIP": SCRIP grid format
            - "Exodus": Exodus file format

        Returns
        -------
        xr.Dataset
            The ``ux.UXDataset`` represented as a ``xr.Dataset`` in the requested format
        """
        from datetime import datetime

        import numpy as np

        from uxarray.conventions import ugrid
        from uxarray.io._scrip import grid_center_lat_lon

        grid_format = grid_format.upper()

        if grid_format == "HEALPIX":
            ds = self.rename_dims({"n_face": "cell"})
            return xr.Dataset(ds)

        elif grid_format == "ESMF":
            # Create output dataset
            out_ds = xr.Dataset()

            # Node Coordinates (nodeCoords)
            if "node_lon" in self and "node_lat" in self:
                out_ds["nodeCoords"] = xr.concat(
                    [self["node_lon"], self["node_lat"]],
                    dim=xr.DataArray([0, 1], dims="coordDim"),
                )
                out_ds["nodeCoords"] = out_ds["nodeCoords"].rename(
                    {"n_node": "nodeCount"}
                )
                out_ds["nodeCoords"] = out_ds["nodeCoords"].transpose(
                    "nodeCount", "coordDim"
                )
                out_ds["nodeCoords"] = out_ds["nodeCoords"].assign_attrs(
                    units="degrees"
                )
                # Clean up unwanted attributes
                for attr in ["standard_name", "long_name"]:
                    if attr in out_ds["nodeCoords"].attrs:
                        del out_ds["nodeCoords"].attrs[attr]
            else:
                raise ValueError(
                    "Input dataset must contain 'node_lon' and 'node_lat'."
                )

            # Face Node Connectivity (elementConn)
            if "face_node_connectivity" in self:
                out_ds["elementConn"] = xr.DataArray(
                    self["face_node_connectivity"] + 1,
                    dims=("elementCount", "maxNodePElement"),
                    attrs={
                        "long_name": "Node Indices that define the element connectivity",
                        "_FillValue": -1,
                    },
                )
                out_ds["elementConn"].encoding = {"dtype": np.int32}
            else:
                raise ValueError("Input dataset must contain 'face_node_connectivity'.")

            # Optional face coordinates (centerCoords)
            if "face_lon" in self and "face_lat" in self:
                out_ds["centerCoords"] = xr.concat(
                    [self["face_lon"], self["face_lat"]],
                    dim=xr.DataArray([0, 1], dims="coordDim"),
                )
                out_ds["centerCoords"] = out_ds["centerCoords"].rename(
                    {"n_face": "elementCount"}
                )
                out_ds["centerCoords"] = out_ds["centerCoords"].transpose(
                    "elementCount", "coordDim"
                )
                out_ds["centerCoords"] = out_ds["centerCoords"].assign_attrs(
                    units="degrees"
                )
                # Clean up unwanted attributes
                for attr in ["standard_name", "long_name"]:
                    if attr in out_ds["centerCoords"].attrs:
                        del out_ds["centerCoords"].attrs[attr]

            # Add global attributes
            out_ds.attrs = {
                "title": "ESMF Unstructured Grid from uxarray",
                "source": "Converted from UGRID conventions by uxarray",
                "date_created": datetime.now().isoformat(),
            }

            return out_ds

        elif grid_format == "SCRIP":
            # Create empty dataset for SCRIP format
            out_ds = xr.Dataset()

            face_node_conn = self["face_node_connectivity"].values.astype(int)
            node_lon = self["node_lon"]
            node_lat = self["node_lat"]

            # Get face areas from the grid
            face_areas = self.uxgrid.compute_face_areas()[0]

            # Reshape arrays for SCRIP format
            f_nodes = face_node_conn.ravel()
            lat_nodes = node_lat[f_nodes].values
            lon_nodes = node_lon[f_nodes].values

            reshp_lat = np.reshape(
                lat_nodes, [face_node_conn.shape[0], face_node_conn.shape[1]]
            )
            reshp_lon = np.reshape(
                lon_nodes, [face_node_conn.shape[0], face_node_conn.shape[1]]
            )

            # Add variables to SCRIP dataset
            out_ds["grid_corner_lat"] = xr.DataArray(
                data=reshp_lat, dims=["grid_size", "grid_corners"]
            )
            out_ds["grid_corner_lon"] = xr.DataArray(
                data=reshp_lon, dims=["grid_size", "grid_corners"]
            )
            out_ds["grid_rank"] = xr.DataArray(data=[1], dims=["grid_rank"])
            out_ds["grid_dims"] = xr.DataArray(
                data=[len(lon_nodes)], dims=["grid_rank"]
            )
            out_ds["grid_imask"] = xr.DataArray(
                data=np.ones(len(reshp_lon), dtype=int), dims=["grid_size"]
            )
            out_ds["grid_area"] = xr.DataArray(data=face_areas, dims=["grid_size"])

            # Calculate grid centers
            center_lat, center_lon = grid_center_lat_lon(out_ds)
            out_ds["grid_center_lon"] = xr.DataArray(
                data=center_lon, dims=["grid_size"]
            )
            out_ds["grid_center_lat"] = xr.DataArray(
                data=center_lat, dims=["grid_size"]
            )

            return out_ds

        elif grid_format == "UGRID":
            # Create output dataset with all grid variables except grid_topology
            if "grid_topology" in self.uxgrid._ds:
                out_ds = self.uxgrid._ds.drop_vars(["grid_topology"])
            else:
                out_ds = self.uxgrid._ds

            grid_topology = ugrid.BASE_GRID_TOPOLOGY_ATTRS

            if "n_edge" in self.uxgrid._ds.dims:
                grid_topology["edge_dimension"] = "n_edge"

            if "face_lon" in self.uxgrid._ds:
                grid_topology["face_coordinates"] = "face_lon face_lat"
            if "edge_lon" in self.uxgrid._ds:
                grid_topology["edge_coordinates"] = "edge_lon edge_lat"

            # TODO: Encode spherical (i.e. node_x) coordinates eventually (need to extend ugrid conventions)

            for conn_name in ugrid.CONNECTIVITY_NAMES:
                if conn_name in self.uxgrid._ds:
                    grid_topology[conn_name] = conn_name

            grid_topology_da = xr.DataArray(data=-1, attrs=grid_topology)

            out_ds["grid_topology"] = grid_topology_da

            return out_ds

        elif grid_format == "EXODUS":
            from uxarray.constants import INT_DTYPE
            from uxarray.grid.coordinates import _lonlat_rad_to_xyz
            from uxarray.io._exodus import _get_element_type

            # Note this is 1-based unlike native Mesh2 construct
            exo_ds = xr.Dataset()

            now = datetime.now()
            date = now.strftime("%Y:%m:%d")
            time = now.strftime("%H:%M:%S")
            fp_word = INT_DTYPE(8)
            exo_version = np.float32(5.0)
            api_version = np.float32(5.0)

            exo_ds.attrs = {
                "api_version": api_version,
                "version": exo_version,
                "floating_point_word_size": fp_word,
                "file_size": 0,
            }

            exo_ds["time_whole"] = xr.DataArray(data=[], dims=["time_step"])

            # qa_records
            ux_exodus_version = 1.0
            qa_records = [["uxarray"], [ux_exodus_version], [date], [time]]
            exo_ds["qa_records"] = xr.DataArray(
                data=xr.DataArray(np.array(qa_records, dtype="str")),
                dims=["four", "num_qa_rec"],
            )

            if "node_x" not in self.uxgrid._ds:
                x, y, z = _lonlat_rad_to_xyz(
                    self.uxgrid._ds["node_lon"].values,
                    self.uxgrid._ds["node_lat"].values,
                )
                c_data = xr.DataArray([x, y, z])
            else:
                c_data = xr.DataArray(
                    [
                        self.uxgrid._ds["node_x"].data.tolist(),
                        self.uxgrid._ds["node_y"].data.tolist(),
                        self.uxgrid._ds["node_z"].data.tolist(),
                    ]
                )
            exo_ds["coord"] = xr.DataArray(data=c_data, dims=["num_dim", "num_nodes"])

            # process face nodes, this array holds num faces at corresponding location
            # eg num_el_all_blks = [0, 0, 6, 12] signifies 6 TRI and 12 SHELL elements
            num_el_all_blks = np.zeros(self.uxgrid._ds.sizes["n_max_face_nodes"], "i8")
            # this list stores connectivity without filling
            conn_nofill = []

            # store the number of faces in an array
            for row in self.uxgrid._ds["face_node_connectivity"].astype(INT_DTYPE).data:
                # find out -1 in each row, this indicates lower than max face nodes
                arr = np.where(row == -1)
                # arr[0].size returns the location of first -1 in the conn list
                # if > 0, arr[0][0] is the num_nodes forming the face
                if arr[0].size > 0:
                    # increment the number of faces at the corresponding location
                    num_el_all_blks[arr[0][0] - 1] += 1
                    # append without -1s eg. [1, 2, 3, -1] to [1, 2, 3]
                    # convert to list (for sorting later)
                    row = row[: (arr[0][0])].tolist()
                    list_node = list(map(int, row))
                    conn_nofill.append(list_node)
                elif arr[0].size == 0:
                    # increment the number of faces for this "n_max_face_nodes" face
                    num_el_all_blks[self.uxgrid._ds.sizes["n_max_face_nodes"] - 1] += 1
                    # get integer list nodes
                    list_node = list(map(int, row.tolist()))
                    conn_nofill.append(list_node)
                else:
                    raise RuntimeError(
                        "num nodes in conn array is greater than n_max_face_nodes. Abort!"
                    )
            # get number of blks found
            num_blks = np.count_nonzero(num_el_all_blks)

            # sort connectivity by size, lower dim faces first
            conn_nofill.sort(key=len)

            # get index of blocks found
            nonzero_el_index_blks = np.nonzero(num_el_all_blks)

            # break face_node_connectivity into blks
            start = 0
            for blk in range(num_blks):
                blkID = blk + 1
                str_el_in_blk = "num_el_in_blk" + str(blkID)
                str_nod_per_el = "num_nod_per_el" + str(blkID)
                str_att_in_blk = "num_att_in_blk" + str(blkID)
                str_global_id = "global_id" + str(blkID)
                str_edge_type = "edge_type" + str(blkID)
                str_attrib = "attrib" + str(blkID)
                str_connect = "connect" + str(blkID)

                # get element type
                num_nodes = len(conn_nofill[start])
                element_type = _get_element_type(num_nodes)

                # get number of faces for this block
                num_faces = num_el_all_blks[nonzero_el_index_blks[0][blk]]
                # assign Data variables
                # convert list to np.array, sorted list guarantees we have the correct info
                conn_blk = conn_nofill[start : start + num_faces]
                conn_np = np.array([np.array(xi, dtype=INT_DTYPE) for xi in conn_blk])
                exo_ds[str_connect] = xr.DataArray(
                    data=xr.DataArray((conn_np[:] + 1)),
                    dims=[str_el_in_blk, str_nod_per_el],
                    attrs={"elem_type": element_type},
                )

                # edge type
                exo_ds[str_edge_type] = xr.DataArray(
                    data=xr.DataArray(
                        np.zeros((num_faces, num_nodes), dtype=INT_DTYPE)
                    ),
                    dims=[str_el_in_blk, str_nod_per_el],
                )

                # global id
                gid = np.arange(start + 1, start + num_faces + 1, 1)
                exo_ds[str_global_id] = xr.DataArray(data=(gid), dims=[str_el_in_blk])

                # attrib
                # TODO: fix num attr
                num_attr = 1
                exo_ds[str_attrib] = xr.DataArray(
                    data=xr.DataArray(np.zeros((num_faces, num_attr), float)),
                    dims=[str_el_in_blk, str_att_in_blk],
                )

                start = num_faces

            # eb_prop1
            prop1_vals = np.arange(1, num_blks + 1, 1)
            exo_ds["eb_prop1"] = xr.DataArray(
                data=prop1_vals, dims=["num_el_blk"], attrs={"name": "ID"}
            )
            # eb_status
            exo_ds["eb_status"] = xr.DataArray(
                data=xr.DataArray(np.ones([num_blks], dtype=INT_DTYPE)),
                dims=["num_el_blk"],
            )

            # eb_names
            eb_names = np.empty(num_blks, dtype="str")
            exo_ds["eb_names"] = xr.DataArray(
                data=xr.DataArray(eb_names), dims=["num_el_blk"]
            )
            cnames = ["x", "y", "z"]

            exo_ds["coor_names"] = xr.DataArray(
                data=xr.DataArray(np.array(cnames, dtype="str")), dims=["num_dim"]
            )

            return exo_ds
        else:
            raise ValueError(
                f"Unsupported grid_format: {grid_format}. "
                f"Supported formats are: UGRID, HEALPix, ESMF, SCRIP, EXODUS"
            )

    def get_dual(self):
        """Compute the dual mesh for a dataset, returns a new dataset object.

        Returns
        --------
        dual : uxds
            Dual Mesh `uxds` constructed
        """

        if _check_duplicate_nodes_indices(self.uxgrid):
            raise RuntimeError("Duplicate nodes found, cannot construct dual")

        if self.uxgrid.partial_sphere_coverage:
            warn(
                "This mesh is partial, which could cause inconsistent results and data will be lost",
                Warning,
            )

        # Get dual mesh node face connectivity
        dual_node_face_conn = construct_dual(grid=self.uxgrid)

        # Construct dual mesh
        dual = self.uxgrid.from_topology(
            self.uxgrid.face_lon.values,
            self.uxgrid.face_lat.values,
            dual_node_face_conn,
        )

        # Initialize new dataset
        dataset = uxarray.UxDataset(uxgrid=dual)

        # Dictionary to swap dimensions
        dim_map = {"n_face": "n_node", "n_node": "n_face"}

        # For each data array in the dataset, reconstruct the data array with the dual mesh
        for var in self.data_vars:
            # Get correct dimensions for the dual
            dims = [dim_map.get(dim, dim) for dim in self[var].dims]

            # Get the values from the data array
            data = np.array(self[var].values)

            # Construct the new data array
            uxda = uxarray.UxDataArray(uxgrid=dual, data=data, dims=dims, name=var)

            # Add data array to dataset
            dataset[var] = uxda

        return dataset

    def where(self, cond: Any, other: Any = dtypes.NA, drop: bool = False):
        return UxDataset(super().where(cond, other, drop), uxgrid=self.uxgrid)

    where.__doc__ = xr.Dataset.where.__doc__

    def sel(
        self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs
    ):
        return UxDataset(
            super().sel(indexers, method, tolerance, drop, **indexers_kwargs),
            uxgrid=self.uxgrid,
        )

    sel.__doc__ = xr.Dataset.sel.__doc__

    def fillna(self, value: Any):
        return UxDataset(super().fillna(value), uxgrid=self.uxgrid)

    fillna.__doc__ = xr.Dataset.fillna.__doc__
