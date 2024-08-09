import uxarray as ux

import os
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

import numpy.testing as nt

gridfile_geos_cs = current_path / "meshfiles" / "geos-cs" / "c12" / "test-c12.native.nc4"



def test_merge_nodes_geos():

    uxgrid_with_duplicates = ux.open_grid(gridfile_geos_cs)

    uxgrid_without_duplicates = ux.open_grid(gridfile_geos_cs)
    uxgrid_without_duplicates.merge_duplicate_node_indices(inplace=True)

    assert (uxgrid_with_duplicates.face_node_connectivity.values != uxgrid_without_duplicates.face_node_connectivity.values).any()

    _uxgrid = ux.open_grid(gridfile_geos_cs)
    uxgrid_without_duplicates_new_grid = _uxgrid.merge_duplicate_node_indices(inplace=False)

    assert (uxgrid_with_duplicates.face_node_connectivity.values != uxgrid_without_duplicates_new_grid.face_node_connectivity.values).any()
