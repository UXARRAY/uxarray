import os

from unittest import TestCase
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

dsfiles_mf_ne30 = str(
    current_path) + "/meshfiles/ugrid/outCSne30/outCSne30_*.nc"

gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
dsfile_v1_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"


class TestGeometryConversions(TestCase):

    def test_to_geodataframe(self):
        """Tests the conversion to ``GeoDataFrame``"""
        ### geoflow
        uxds_geoflow = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)

        # v1 is mapped to nodes, should raise a value error
        with self.assertRaises(ValueError):
            uxds_geoflow['v1'].to_geodataframe()

        # grid conversion
        gdf_geoflow_grid = uxds_geoflow.uxgrid.to_geodataframe()

        # number of elements
        assert gdf_geoflow_grid.shape == (uxds_geoflow.uxgrid.nMesh2_face, 1)

        ### n30
        uxds_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

        gdf_geoflow_data = uxds_ne30['psi'].to_geodataframe()

        assert gdf_geoflow_data.shape == (uxds_ne30.uxgrid.nMesh2_face, 2)

    def test_to_polycollection(self):
        """Tests the conversion to ``PolyCollection``"""
        ### geoflow
        uxds_geoflow = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)

        # v1 is mapped to nodes, should raise a value error
        with self.assertRaises(ValueError):
            uxds_geoflow['v1'].to_polycollection()

        # grid conversion
        pc_geoflow_grid = uxds_geoflow.uxgrid.to_polycollection()

        # number of elements
        assert len(pc_geoflow_grid._paths) == len(
            uxds_geoflow.uxgrid.corrected_polygon_shells)

        # ### n30
        uxds_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

        pc_geoflow_data = uxds_ne30['psi'].to_polycollection()

        assert len(pc_geoflow_data._paths) == len(
            uxds_ne30.uxgrid.corrected_polygon_shells)

    def test_geodataframe_caching(self):
        uxds = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

        gdf_start = uxds['psi'].to_polycollection()

        gdf_next = uxds['psi'].to_polycollection()

        # with caching, they point to the same area in memory
        assert gdf_start is gdf_next

        gdf_end = uxds['psi'].to_polycollection(override_grid=True)

        # override will recompute the grid
        assert gdf_start is not gdf_end
