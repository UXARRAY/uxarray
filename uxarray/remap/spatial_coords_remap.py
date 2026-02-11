import warnings
from typing import Dict, Literal, Optional, Tuple

import xarray as xr

from uxarray.core.dataarray import UxDataArray
from uxarray.grid.grid import Grid

COORD_TYPES = {
    "LON": "lon",
    "LAT": "lat",
    "CART_X": "X",
    "CART_Y": "Y",
}

# CF attributes that indicate coordinate type
CF_LAT_ATTRS = ["latitude", "projection_y_coordinate"]
CF_LON_ATTRS = ["longitude", "projection_x_coordinate"]

# CF units that indicate coordinate type
CF_LAT_UNITS = ["degrees_north", "degree_north", "degree_n"]
CF_LON_UNITS = ["degrees_east", "degree_east", "degree_e"]


class SpatialCoordsRemapper:
    """Ensures remapping spatial coordinates between the source and destination grid for the remapping functions.
    It may include remapping of values, renaming, and removal of some of the coordinates with respect to the
    dimensions of source data & coordinates and the `remap_to` selection."""

    def __init__(
        self,
        source: UxDataArray,
        destination_grid: Grid,
        remap_to: Literal["nodes", "faces", "edges"],
    ):
        """
        Initialize spatial coordinate remapper for UXarray's remapping functions.

        Parameters
        ----------
        source : UxDataArray
            Source data array that is being remapped to the `destination_grid`.
        destination_grid : Grid
            Destination grid that `source` is being remapped to.
        remap_to : str
            Which grid element receives the remapped values, either 'nodes', 'faces', or 'edges'.
        """

        if source is None:
            raise ValueError(
                "`source` must be provided for spatial coordinates remapping."
            )

        if destination_grid is None:
            raise ValueError(
                "`destination_grid` must be provided for spatial coordinates remapping."
            )

        self.destination_grid = destination_grid
        self.source = source
        self.remap_to = remap_to

    def _get_destination_grid_coords(self) -> Dict[str, xr.DataArray]:
        """
        Get the spatial coordinates of the destination grid corresponding to `remap_to`.

        Returns
        -------
        Dict[str, xr.DataArray]
            Dictionary with 'lon' and 'lat' coordinate arrays
        """
        if self.remap_to == "nodes":
            return {
                COORD_TYPES["LON"]: self.destination_grid.node_lon,
                COORD_TYPES["LAT"]: self.destination_grid.node_lat,
                COORD_TYPES["CART_X"]: self.destination_grid.node_x,
                COORD_TYPES["CART_Y"]: self.destination_grid.node_y,
            }
        elif self.remap_to == "faces":
            return {
                COORD_TYPES["LON"]: self.destination_grid.face_lon,
                COORD_TYPES["LAT"]: self.destination_grid.face_lat,
                COORD_TYPES["CART_X"]: self.destination_grid.face_x,
                COORD_TYPES["CART_Y"]: self.destination_grid.face_y,
            }
        elif self.remap_to == "edges":
            return {
                COORD_TYPES["LON"]: self.destination_grid.edge_lon,
                COORD_TYPES["LAT"]: self.destination_grid.edge_lat,
                COORD_TYPES["CART_X"]: self.destination_grid.edge_x,
                COORD_TYPES["CART_Y"]: self.destination_grid.edge_y,
            }
        else:
            raise ValueError(
                f"Unknown `remap_to`: {self.remap_to}. Must be either 'nodes', 'faces', or 'edges'."
            )

    def _find_source_coords(self) -> Dict[str, Tuple[str, str]]:
        """
        Find spatial coordinate variables in `source` by checking their attributes, units, and axes.

        Returns
        -------
        Dict[str, Tuple[str, str]]
            Dictionary with keys as spatial identifiers ('lat' or 'lon') and values as
            (coord_var_name, standard_name) tuples

            Example output would look like:
                {
                    'lat': ('Mesh2_face_y', 'latitude'),
                    'lon': ('Mesh2_face_x', 'longitude')
                }
        """

        source_coords = {}

        # Check all coordinates in `source`
        for coord_name in self.source.coords:
            coord = self.source.coords[coord_name]

            # Skip if in rare case this coordinate doesn't have dimensions or has multiple dimensions
            if not hasattr(coord, "dims") or len(coord.dims) != 1:
                continue

            # Determine if this is a spatial coordinate by checking attributes
            is_spatial = False
            coord_type = None  # will be 'lat' or 'lon' later

            if hasattr(coord, "attrs"):
                # Check `standard_name` first
                if "standard_name" in coord.attrs:
                    std_name = coord.attrs["standard_name"].lower()
                    if std_name in CF_LAT_ATTRS:
                        is_spatial = True
                        coord_type = COORD_TYPES["LAT"]
                    elif std_name in CF_LON_ATTRS:
                        is_spatial = True
                        coord_type = COORD_TYPES["LON"]

                # Check units if standard_name didn't work
                if not is_spatial and "units" in coord.attrs:
                    units = coord.attrs["units"].lower()
                    if any(u in units for u in CF_LAT_UNITS):
                        is_spatial = True
                        coord_type = COORD_TYPES["LAT"]
                    elif any(u in units for u in CF_LON_UNITS):
                        is_spatial = True
                        coord_type = COORD_TYPES["LON"]

                # Check axis attribute as last chance
                if not is_spatial and "axis" in coord.attrs:
                    axis = coord.attrs["axis"].upper()
                    if axis == COORD_TYPES["CART_Y"]:
                        is_spatial = True
                        coord_type = COORD_TYPES["CART_Y"]
                    elif axis == COORD_TYPES["CART_X"]:
                        is_spatial = True
                        coord_type = COORD_TYPES["CART_X"]

            # If a spatial coord is found and `coord_type` is identified in `source`
            if is_spatial and coord_type:
                # Store the coordinate variable
                standard_name = coord.attrs.get("standard_name", coord_type)
                source_coords[coord_type] = (coord_name, standard_name)

        return source_coords

    def _get_element_type_from_dimension(self, dim_name: str) -> Optional[str]:
        """
        Determine element type (i.e. 'nodes', 'faces', or 'edges') from dimension name.

        Parameters
        ----------
        dim_name : str
            Dimension name (e.g., 'n_face', 'nMesh2_face', etc.)

        Returns
        -------
        Optional[str]
            Element type ('nodes', 'faces', 'edges') or None
        """
        dim_lower = dim_name.lower()
        if "face" in dim_lower:
            return "faces"
        elif "node" in dim_lower:
            return "nodes"
        elif "edge" in dim_lower:
            return "edges"
        return None

    def _rename_coord_for_new_dimension(
        self, original_name: str, old_element: str, new_element: str
    ) -> str:
        """
        Rename a coordinate variable when changing from one element type to another, which occurs when the `remap_to`
        element does not match the `source` dimension.

        Parameters
        ----------
        original_name : str
            Original coordinate variable name
        old_element : str
            Old element type ('nodes', 'faces', 'edges')
        new_element : str
            New element type ('nodes', 'faces', 'edges')

        Returns
        -------
        str
            New coordinate name with element type updated
        """
        # Map plural to singular
        element_type_to_coord_name_string = {
            "nodes": "node",
            "faces": "face",
            "edges": "edge",
        }

        old_coord_name_string = element_type_to_coord_name_string[old_element]
        new_coord_name_string = element_type_to_coord_name_string[new_element]

        # Try to replace the old element name in the coordinate name
        # Handle both singular and plural forms
        new_name = original_name

        # Case-sensitive replacements
        # e.g. "*face*" -> "*node*"
        new_name = new_name.replace(old_coord_name_string, new_coord_name_string)
        # e.g. "*faces*" -> "*nodes*"
        new_name = new_name.replace(old_element, new_element)
        # e.g. "*FACE*" -> "*NODE*"
        new_name = new_name.replace(
            old_coord_name_string.upper(), new_coord_name_string.upper()
        )
        # e.g. "*FACES*" -> "*NODES*"
        new_name = new_name.replace(old_element.upper(), new_element.upper())
        # e.g. "*Face*" -> "*Node*"
        new_name = new_name.replace(
            old_coord_name_string.capitalize(), new_coord_name_string.capitalize()
        )
        # e.g. "*Faces*" -> "*Nodes*"
        new_name = new_name.replace(old_element.capitalize(), new_element.capitalize())

        return new_name

    def construct_output_coords(self) -> Dict[str, xr.DataArray]:
        """
        Construct spatial coordinates for the remapped output by finding spatial coordinate variables, if any,
        in `source` and employing a logic as follows:

        Logic:
        ------
        If `remap_to` matches the `source` dimension (e.g. `source` on face centers` and `remap_to="faces"` etc.)
            - Swap values of spatial coords with values of the corresponding coords from `destination_grid`

        Else (if `remap_to` doesn't match `source` dim (e.g. `source` on face centers but `remap_to="nodes"` etc.))
            - Swap values of spatial coords with values of the coords from `destination_grid` that are
            defined on the `remap_to` dimension.
                - Rename these coords to reflect new element type (e.g. 'face_x' → 'node_x')

        Returns
        -------
        Dict[str, xr.DataArray]
            Dictionary mapping output coordinate variables to their new values
        """

        # Find spatial coordinate variables in `source` by checking their attributes
        source_coords = self._find_source_coords()

        if not source_coords:
            warnings.warn(
                "No spatial coordinate variables found in `source`.",
                UserWarning,
                stacklevel=2,
            )
            return {}

        # Get the dimension that `source` is defined on
        source_dims = list(self.source.dims)
        if len(source_dims) == 0:
            raise ValueError("Source data has no dimensions")

        # Find the primary spatial dimension (should be n_face, n_node, or n_edge)
        source_spatial_dim = None
        for dim in source_dims:
            if self._get_element_type_from_dimension(dim) is not None:
                source_spatial_dim = dim
                break

        if source_spatial_dim is None:
            raise ValueError(
                f"Could not identify spatial dimension in `source` dims: {source_dims}"
            )

        source_element_type = self._get_element_type_from_dimension(source_spatial_dim)

        # Get destination grid values for the remap_to element
        dest_grid_coords = self._get_destination_grid_coords()

        output_coords = {}

        # Logic for the remapped spatial coords construction starts here
        # If `remap_to` matches `source` dimension
        if source_element_type == self.remap_to:
            # Swap coords on matching dimension
            for coord_type in COORD_TYPES.values():
                if coord_type in source_coords:
                    source_coord_name, std_name = source_coords[coord_type]
                    out_name = source_coord_name

                    # Assign destination grid values
                    output_coords[out_name] = dest_grid_coords[coord_type].variable

        # `remap_to` differs from `source` dimension
        else:
            warnings.warn(
                f"Coordinates handling as part of remapping: `source` has the dimension:"
                f"('{source_spatial_dim}') but is being remapped to ('{self.remap_to}'). Therefore, "
                f"coordinate values will be swapped to the '{self.remap_to}' coordinates from "
                f"`destination_grid` and renamed accordingly.",
                UserWarning,
                stacklevel=2,
            )

            renamed_coords = []

            # Swap and rename (as needed) coords from source dimension
            for coord_type in COORD_TYPES.values():
                if coord_type in source_coords:
                    source_coord_name, std_name = source_coords[coord_type]

                    # Rename to reflect new element type
                    out_name = self._rename_coord_for_new_dimension(
                        source_coord_name, source_element_type, self.remap_to
                    )
                    if out_name != source_coord_name:
                        renamed_coords.append((source_coord_name, out_name))

                    # Assign destination grid values on remap_to dimension
                    output_coords[out_name] = dest_grid_coords[coord_type].variable

            if renamed_coords:
                for old, new in renamed_coords:
                    warnings.warn(
                        f"Renamed coordinate '{old}' → '{new}' due to dimension change.",
                        UserWarning,
                        stacklevel=2,
                    )

        return output_coords
