"""uxarray grid module."""
import os
import xarray as xr
import numpy as np

# reader and writer imports
from ._exodus import _read_exodus, _write_exodus
from ._ugrid import _read_ugrid, _write_ugrid
from ._shapefile import _read_shpfile
from ._scrip import _read_scrip, _write_scrip
from .helpers import get_all_face_area_from_coords, parse_grid_type


class Grid:

    def __init__(self, ds, **kwargs):
        # unpack kwargs
        # sets default values for all kwargs to None
        kwargs_list = [
            'gridspec', 'vertices', 'islatlon', 'concave', 'mesh_filetype',
            'source_grid', 'source_datasets'
        ]
        for key in kwargs_list:
            setattr(self, key, kwargs.get(key, None))

        self.__init_ds_var_names__()

        # initialize face_area variable
        self._face_areas = None

        # check if initializing from verts:
        if isinstance(ds, (list, tuple, np.ndarray)):
            self.vertices = ds
            self.__from_vert__()
            self.source_grid = "From vertices"
            self.source_datasets = None
        # check if initializing from string
        # TODO: re-add gridspec initialization when implemented
        elif isinstance(ds, xr.Dataset):
            self.ds = ds
            self.mesh_filetype = parse_grid_type(ds, **kwargs)
            self.__from_ds__(dataset=ds)
        else:
            raise RuntimeError("Dataset is not a valid input type.")

        # initialize convenience attributes
        self.__init_grid_var_attrs__()

    def __init_grid_var_attrs__(self):
        """Initialize attributes for directly accessing Coordinate and Data
        variables through ugrid conventions.

      Examples
      ----------
      Assuming the mesh node coordinates for longitude are stored with an input
      name of 'mesh_node_x', we store this variable name in the `ds_var_names`
      dictionary with the key 'Mesh2_node_x'. In order to access it:

      >>> x = grid.ds[grid.ds_var_names["Mesh2_node_x"]]

      With the help of this function, we can directly access it through the
      use of a standardized name (ugrid convention)

      >>> x = grid.Mesh2_node_x
        """

        # Set UGRID standardized attributes
        for key, value in self.ds_var_names.items():
            # Present Data Names
            if self.ds.data_vars is not None:
                if value in self.ds.data_vars:
                    setattr(self, key, self.ds[value])

            # Present Coordinate Names
            if self.ds.coords is not None:
                if value in self.ds.coords:
                    setattr(self, key, self.ds[value])

    def __from_vert__(self):
        """Create a grid with one face with vertices specified by the given
        argument.

        Called by :func:`__init__`.
        """
        self.ds = xr.Dataset()
        self.ds["Mesh2"] = xr.DataArray(
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of unstructured mesh",
                "topology_dimension": -1,
                "node_coordinates": "Mesh2_node_x Mesh2_node_y Mesh2_node_z",
                "node_dimension": "nMesh2_node",
                "face_node_connectivity": "Mesh2_face_nodes",
                "face_dimension": "nMesh2_face"
            })
        self.ds.Mesh2.attrs['topology_dimension'] = self.vertices[0].size

        # set default coordinate units to spherical coordinates
        # users can change to cartesian if using cartesian for initialization
        x_units = "degrees_east"
        y_units = "degrees_north"
        if self.vertices[0].size > 2:
            z_units = "elevation"

        x_coord = self.vertices.transpose()[0]
        y_coord = self.vertices.transpose()[1]
        if self.vertices[0].size > 2:
            z_coord = self.vertices.transpose()[2]

        # single face with all nodes
        num_nodes = x_coord.size
        connectivity = [list(range(0, num_nodes))]

        self.ds["Mesh2_node_x"] = xr.DataArray(data=xr.DataArray(x_coord),
                                               dims=["nMesh2_node"],
                                               attrs={"units": x_units})
        self.ds["Mesh2_node_y"] = xr.DataArray(data=xr.DataArray(y_coord),
                                               dims=["nMesh2_node"],
                                               attrs={"units": y_units})
        if self.vertices[0].size > 2:
            self.ds["Mesh2_node_z"] = xr.DataArray(data=xr.DataArray(z_coord),
                                                   dims=["nMesh2_node"],
                                                   attrs={"units": z_units})

        self.ds["Mesh2_face_nodes"] = xr.DataArray(
            data=xr.DataArray(connectivity),
            dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
            attrs={
                "cf_role": "face_node_connectivity",
                "_FillValue": -1,
                "start_index": 0
            })

    def __init_ds_var_names__(self):
        """Populates a dictionary for storing uxarray's internal representation
        of xarray object.

        Note ugrid conventions are flexible with names of variables, see:
        http://ugrid-conventions.github.io/ugrid-conventions/
        """
        self.ds_var_names = {
            "Mesh2": "Mesh2",
            "Mesh2_node_x": "Mesh2_node_x",
            "Mesh2_node_y": "Mesh2_node_y",
            "Mesh2_node_z": "Mesh2_node_z",
            "Mesh2_face_nodes": "Mesh2_face_nodes",
            # initialize dims
            "nMesh2_node": "nMesh2_node",
            "nMesh2_face": "nMesh2_face",
            "nMaxMesh2_face_nodes": "nMaxMesh2_face_nodes"
        }

    # load mesh from a file
    def __from_ds__(self, dataset):
        """Loads a mesh dataset."""
        # call reader as per mesh_filetype
        if self.mesh_filetype == "exo":
            self.ds = _read_exodus(dataset, self.ds_var_names)
        elif self.mesh_filetype == "scrip":
            self.ds = _read_scrip(dataset)
        elif self.mesh_filetype == "ugrid":
            self.ds, self.ds_var_names = _read_ugrid(dataset, self.ds_var_names)
        else:
            raise RuntimeError("unknown file format: ")

        dataset.close()

    def write(self, outfile, grid_type):
        """Writes mesh file as per extension supplied in the outfile string.

        Parameters
        ----------
        outfile : str, required
            Path to output file
        grid_type : str, required
            Grid type of output file.
            Currently supported options are "ugrid", "exodus", and "scrip"

        Raises
        ------
        RuntimeError
            If unsupported grid type provided or directory not found
        """

        if grid_type == "ugrid":
            _write_ugrid(self.ds, outfile, self.ds_var_names)
        elif grid_type == "exodus":
            _write_exodus(self.ds, outfile, self.ds_var_names)
        elif grid_type == "scrip":
            _write_scrip(outfile, self.Mesh2_face_nodes, self.Mesh2_node_x,
                         self.Mesh2_node_y, self.face_areas)
        else:
            raise RuntimeError("Format not supported for writing: ", grid_type)

    def compute_face_areas(self, quadrature_rule="triangular", order=4):
        """Face areas calculation function for grid class, calculates area of
        all faces in the grid.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Area of all the faces in the mesh : np.ndarray

        Examples
        --------
        Open a uxarray grid file

        >>> grid = ux.open_dataset("/home/jain/uxarray/test/meshfiles/outCSne30.ug")

        Get area of all faces in the same order as listed in grid.ds.Mesh2_face_nodes

        >>> grid.get_face_areas
        array([0.00211174, 0.00211221, 0.00210723, ..., 0.00210723, 0.00211221,
            0.00211174])
        """
        if self._face_areas is None:
            # area of a face call needs the units for coordinate conversion if spherical grid is used
            coords_type = "spherical"
            if not "degree" in self.Mesh2_node_x.units:
                coords_type = "cartesian"

            face_nodes = self.Mesh2_face_nodes.data.astype(np.int64)
            dim = self.Mesh2.attrs['topology_dimension']

            # initialize z
            z = np.zeros((self.ds.nMesh2_node.size))

            # call func to cal face area of all nodes
            x = self.Mesh2_node_x.data
            y = self.Mesh2_node_y.data
            # check if z dimension
            if self.Mesh2.topology_dimension > 2:
                z = self.Mesh2_node_z.data

            # call function to get area of all the faces as a np array
            self._face_areas = get_all_face_area_from_coords(
                x, y, z, face_nodes, dim, quadrature_rule, order, coords_type)

        return self._face_areas

    # use the property keyword for declaration on face_areas property
    @property
    def face_areas(self):
        """Declare face_areas as a property."""

        if self._face_areas is None:
            self.compute_face_areas()
        return self._face_areas

    def calculate_total_face_area(self, quadrature_rule="triangular", order=4):
        """Function to calculate the total surface area of all the faces in a
        mesh.

        Parameters
        ----------
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Sum of area of all the faces in the mesh : float
        """

        # call function to get area of all the faces as a np array
        face_areas = self.compute_face_areas(quadrature_rule, order)

        return np.sum(face_areas)

    def integrate(self, var_ds, quadrature_rule="triangular", order=4):
        """Integrates over all the faces of the given mesh.

        Parameters
        ----------
        var_key : str, required
            Name of dataset variable for integration
        quadrature_rule : str, optional
            Quadrature rule to use. Defaults to "triangular".
        order : int, optional
            Order of quadrature rule. Defaults to 4.

        Returns
        -------
        Calculated integral : float

        Examples
        --------
        Open grid file only

        >>> grid = ux.open_dataset("grid.ug", "centroid_pressure_data_ug")

        Open grid file along with data

        >>> integral_psi = grid.integrate("psi")
        """
        integral = 0.0

        # call function to get area of all the faces as a np array
        face_areas = self.compute_face_areas(quadrature_rule, order)

        var_key = list(var_ds.keys())
        if len(var_key) > 1:
            # warning: print message
            print(
                "WARNING: The xarray dataset file has more than one variable, using the first variable for integration"
            )
        var_key = var_key[0]
        face_vals = var_ds[var_key].to_numpy()
        integral = np.dot(face_areas, face_vals)

        return integral
