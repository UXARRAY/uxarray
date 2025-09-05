import os
from pathlib import Path
import numpy as np
import uxarray as ux
from uxarray.grid.geometry import _build_polygon_shells, _build_corrected_polygon_shells
from uxarray.core.dataset import UxDataset, UxDataArray
import pytest

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *

gridfile_ne30 = OUTCSNE30_GRID
dsfile_var2_ne30 = OUTCSNE30_VAR2
dsfile_v1_geoflow = GEOFLOW_V1

def test_to_dataset():
    """Tests the conversion of UxDataArrays to a UXDataset."""
    uxds = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)
    uxds_converted = uxds['var2'].to_dataset()

    assert isinstance(uxds_converted, UxDataset)
    assert uxds_converted.uxgrid == uxds.uxgrid

def test_get_dual():
    """Tests the creation of the dual mesh on a data array."""
    uxds = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)
    dual = uxds['var2'].get_dual()

    assert isinstance(dual, UxDataArray)
    assert dual._node_centered()

def test_to_geodataframe():
    """Tests the conversion to ``GeoDataFrame``"""
    # GeoFlow
    uxds_geoflow = ux.open_dataset(GEOFLOW_GRID, dsfile_v1_geoflow)

    # v1 is mapped to nodes, should raise a value error
    with pytest.raises(ValueError):
        uxds_geoflow['v1'].to_geodataframe()

    # grid conversion
    gdf_geoflow_grid = uxds_geoflow.uxgrid.to_geodataframe(periodic_elements='split')

    # number of elements
    assert gdf_geoflow_grid.shape == (uxds_geoflow.uxgrid.n_face, 1)

    # NE30
    uxds_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

    gdf_geoflow_data = uxds_ne30['var2'].to_geodataframe(periodic_elements='split')

    assert gdf_geoflow_data.shape == (uxds_ne30.uxgrid.n_face, 2)

def test_to_polycollection():
    """Tests the conversion to ``PolyCollection``"""
    # GeoFlow
    uxds_geoflow = ux.open_dataset(GEOFLOW_GRID, dsfile_v1_geoflow)

    # v1 is mapped to nodes, should raise a value error
    with pytest.raises(ValueError):
        uxds_geoflow['v1'].to_polycollection()

    # grid conversion
    pc_geoflow_grid = uxds_geoflow.uxgrid.to_polycollection()

    # number of elements
    assert len(pc_geoflow_grid._paths) == uxds_geoflow.uxgrid.n_face

def test_geodataframe_caching():
    uxds = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

    gdf_start = uxds['var2'].to_geodataframe()
    gdf_next = uxds['var2'].to_geodataframe()

    # with caching, they point to the same area in memory
    assert gdf_start is gdf_next

    gdf_end = uxds['var2'].to_geodataframe(override=True)

    # override will recompute the grid
    assert gdf_start is not gdf_end
