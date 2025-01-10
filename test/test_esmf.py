import uxarray as ux
import os
from pathlib import Path
import pytest

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

esmf_ne30_grid_path = current_path / 'meshfiles' / "esmf" / "ne30" / "ne30pg3.grid.nc"
esmf_ne30_data_path = current_path / 'meshfiles' / "esmf" / "ne30" / "ne30pg3.data.nc"

def test_read_esmf():
    """Tests the reading of an ESMF grid file and its encoding into the UGRID
    conventions."""
    uxgrid = ux.open_grid(esmf_ne30_grid_path)

    dims = ['n_node', 'n_face', 'n_max_face_nodes']
    coords = ['node_lon', 'node_lat', 'face_lon', 'face_lat']
    conns = ['face_node_connectivity', 'n_nodes_per_face']

    for dim in dims:
        assert dim in uxgrid._ds.dims

    for coord in coords:
        assert coord in uxgrid._ds

    for conn in conns:
        assert conn in uxgrid._ds

def test_read_esmf_dataset():
    """Tests the constructing of a UxDataset from an ESMF Grid and Data
    File."""
    uxds = ux.open_dataset(esmf_ne30_grid_path, esmf_ne30_data_path)

    dims = ['n_node', 'n_face']

    for dim in dims:
        assert dim in uxds.dims
