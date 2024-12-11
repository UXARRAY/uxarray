import uxarray as ux
import os
import numpy.testing as nt
import pytest
from pathlib import Path
from uxarray.constants import INT_FILL_VALUE

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

GRID_PATHS = [
    current_path / 'meshfiles' / "mpas" / "QU" / 'oQU480.231010.nc',
    current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc",
    current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
]

def test_minimal_class_method():
    """Tests the minimal required variables for constructing a grid using the
    from topology class method."""
    for grid_path in GRID_PATHS:
        uxgrid = ux.open_grid(grid_path)

        uxgrid_ft = ux.Grid.from_topology(
            node_lon=uxgrid.node_lon.values,
            node_lat=uxgrid.node_lat.values,
            face_node_connectivity=uxgrid.face_node_connectivity.values,
            fill_value=INT_FILL_VALUE,
            start_index=0
        )

        nt.assert_array_equal(uxgrid.node_lon.values, uxgrid_ft.node_lon.values)
        nt.assert_array_equal(uxgrid.node_lat.values, uxgrid_ft.node_lat.values)
        nt.assert_array_equal(uxgrid.face_node_connectivity.values, uxgrid_ft.face_node_connectivity.values)

def test_minimal_api():
    """Tests the minimal required variables for constructing a grid using the
    ``ux.open_dataset`` method."""
    for grid_path in GRID_PATHS:
        uxgrid = ux.open_grid(grid_path)

        uxgrid_ft = ux.Grid.from_topology(
            node_lon=uxgrid.node_lon.values,
            node_lat=uxgrid.node_lat.values,
            face_node_connectivity=uxgrid.face_node_connectivity.values,
            fill_value=INT_FILL_VALUE,
            start_index=0
        )

        grid_topology = {
            'node_lon': uxgrid.node_lon.values,
            'node_lat': uxgrid.node_lat.values,
            'face_node_connectivity': uxgrid.face_node_connectivity.values,
            'fill_value': INT_FILL_VALUE,
            'start_index': 0
        }

        uxgrid_ft = ux.open_grid(grid_topology)

        nt.assert_array_equal(uxgrid.node_lon.values, uxgrid_ft.node_lon.values)
        nt.assert_array_equal(uxgrid.node_lat.values, uxgrid_ft.node_lat.values)
        nt.assert_array_equal(uxgrid.face_node_connectivity.values, uxgrid_ft.face_node_connectivity.values)

def test_dataset():
    uxds = ux.open_dataset(GRID_PATHS[0], GRID_PATHS[0])

    grid_topology = {
        'node_lon': uxds.uxgrid.node_lon.values,
        'node_lat': uxds.uxgrid.node_lat.values,
        'face_node_connectivity': uxds.uxgrid.face_node_connectivity.values,
        'fill_value': INT_FILL_VALUE,
        'start_index': 0,
        'dims_dict': {"nVertices": "n_node"}
    }

    uxds_ft = ux.open_grid(grid_topology, GRID_PATHS[1])

    uxgrid = uxds.uxgrid
    uxgrid_ft = uxds_ft

    nt.assert_array_equal(uxgrid.node_lon.values, uxgrid_ft.node_lon.values)
    nt.assert_array_equal(uxgrid.node_lat.values, uxgrid_ft.node_lat.values)
    nt.assert_array_equal(uxgrid.face_node_connectivity.values, uxgrid_ft.face_node_connectivity.values)

    assert uxds_ft.dims == {'n_face', 'n_node', 'n_max_face_nodes'}
