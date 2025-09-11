import uxarray as ux
import os
from pathlib import Path
import pytest





def test_read_geos_cs_grid(gridpath):
    """Tests the conversion of a CS12 GEOS-CS Grid to the UGRID conventions.

    A CS12 grid has 6 faces, each with 12x12 faces and 13x13 nodes each.
    """
    uxgrid = ux.open_grid(gridpath("geos-cs", "c12", "test-c12.native.nc4"))

    n_face = 6 * 12 * 12
    n_node = 6 * 13 * 13

    assert uxgrid.n_face == n_face
    assert uxgrid.n_node == n_node

def test_read_geos_cs_uxds(gridpath):
    """Tests the creating of a UxDataset from a CS12 GEOS-CS Grid."""
    uxds = ux.open_dataset(gridpath("geos-cs", "c12", "test-c12.native.nc4"), gridpath("geos-cs", "c12", "test-c12.native.nc4"))

    assert uxds['T'].shape[-1] == uxds.uxgrid.n_face
