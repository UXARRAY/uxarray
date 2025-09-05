import uxarray as ux
import os
from pathlib import Path
import pytest

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *

gridfile_geos_cs = GEOS_CS_C12_GRID

def test_read_geos_cs_grid():
    """Tests the conversion of a CS12 GEOS-CS Grid to the UGRID conventions.

    A CS12 grid has 6 faces, each with 12x12 faces and 13x13 nodes each.
    """
    uxgrid = ux.open_grid(gridfile_geos_cs)

    n_face = 6 * 12 * 12
    n_node = 6 * 13 * 13

    assert uxgrid.n_face == n_face
    assert uxgrid.n_node == n_node

def test_read_geos_cs_uxds():
    """Tests the creating of a UxDataset from a CS12 GEOS-CS Grid."""
    uxds = ux.open_dataset(gridfile_geos_cs, gridfile_geos_cs)

    assert uxds['T'].shape[-1] == uxds.uxgrid.n_face
