import os
from pathlib import Path
import numpy as np
import uxarray as ux
from uxarray.grid.geometry import _build_polygon_shells, _build_corrected_polygon_shells
from uxarray.core.dataset import UxDataset, UxDataArray
import pytest

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
dsfile_v1_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"

def test_to_dataset():
    """Tests the conversion of UxDataArrays to a UXDataset."""
    uxds = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)
    uxds_converted = uxds['psi'].to_dataset()

    assert isinstance(uxds_converted, UxDataset)
    assert uxds_converted.uxgrid == uxds.uxgrid

def test_get_dual():
    """Tests the creation of the dual mesh on a data array."""
    uxds = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)
    dual = uxds['psi'].get_dual()

    assert isinstance(dual, UxDataArray)
    assert dual._node_centered()

def test_to_geodataframe():
    """Tests the conversion to ``GeoDataFrame``"""
    # GeoFlow
    uxds_geoflow = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)

    # v1 is mapped to nodes, should raise a value error
    with pytest.raises(ValueError):
        uxds_geoflow['v1'].to_geodataframe()

    # grid conversion
    gdf_geoflow_grid = uxds_geoflow.uxgrid.to_geodataframe(periodic_elements='split')

    # number of elements
    assert gdf_geoflow_grid.shape == (uxds_geoflow.uxgrid.n_face, 1)

    # NE30
    uxds_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

    gdf_geoflow_data = uxds_ne30['psi'].to_geodataframe(periodic_elements='split')

    assert gdf_geoflow_data.shape == (uxds_ne30.uxgrid.n_face, 2)

def test_to_polycollection():
    """Tests the conversion to ``PolyCollection``"""
    # GeoFlow
    uxds_geoflow = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)

    # v1 is mapped to nodes, should raise a value error
    with pytest.raises(ValueError):
        uxds_geoflow['v1'].to_polycollection()

    # grid conversion
    pc_geoflow_grid = uxds_geoflow.uxgrid.to_polycollection(periodic_elements='split')

    polygon_shells = _build_polygon_shells(
        uxds_geoflow.uxgrid.node_lon.values,
        uxds_geoflow.uxgrid.node_lat.values,
        uxds_geoflow.uxgrid.face_node_connectivity.values,
        uxds_geoflow.uxgrid.n_face, uxds_geoflow.uxgrid.n_max_face_nodes,
        uxds_geoflow.uxgrid.n_nodes_per_face.values)

    corrected_polygon_shells, _ = _build_corrected_polygon_shells(polygon_shells)

    # number of elements
    assert len(pc_geoflow_grid._paths) == len(corrected_polygon_shells)

    # NE30
    uxds_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

    polygon_shells = _build_polygon_shells(
        uxds_ne30.uxgrid.node_lon.values, uxds_ne30.uxgrid.node_lat.values,
        uxds_ne30.uxgrid.face_node_connectivity.values,
        uxds_ne30.uxgrid.n_face, uxds_ne30.uxgrid.n_max_face_nodes,
        uxds_ne30.uxgrid.n_nodes_per_face.values)

    corrected_polygon_shells, _ = _build_corrected_polygon_shells(polygon_shells)

    pc_geoflow_data = uxds_ne30['psi'].to_polycollection(periodic_elements='split')

    assert len(pc_geoflow_data._paths) == len(corrected_polygon_shells)

def test_geodataframe_caching():
    uxds = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

    gdf_start = uxds['psi'].to_geodataframe()
    gdf_next = uxds['psi'].to_geodataframe()

    # with caching, they point to the same area in memory
    assert gdf_start is gdf_next

    gdf_end = uxds['psi'].to_geodataframe(override=True)

    # override will recompute the grid
    assert gdf_start is not gdf_end
