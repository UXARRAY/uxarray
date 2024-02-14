import uxarray as ux

from uxarray.constants import INT_FILL_VALUE
import numpy.testing as nt
import os

import pytest

from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

GRID_PATHS = [
    current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc',
    current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc",
    current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
]




def test_minimal():
    """Tests the minimal required variables for constructing a grid using the
    from topology class method."""

    for grid_path in GRID_PATHS:
        uxgrid = ux.open_grid(grid_path)

        uxgrid_ft = ux.Grid.from_topology(node_lon=uxgrid.node_lon.values,
                                          node_lat=uxgrid.node_lat.values,
                                          face_node_connectivity=uxgrid.face_node_connectivity.values,
                                          fill_value=INT_FILL_VALUE,
                                          start_index=0)

        nt.assert_array_equal(uxgrid.node_lon.values, uxgrid_ft.node_lon.values)
        nt.assert_array_equal(uxgrid.node_lat.values, uxgrid_ft.node_lat.values)
        nt.assert_array_equal(uxgrid.face_node_connectivity.values, uxgrid_ft.face_node_connectivity.values)





        pass
