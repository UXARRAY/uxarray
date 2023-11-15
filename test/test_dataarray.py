import os

from unittest import TestCase
from pathlib import Path

import numpy as np

import uxarray as ux

from uxarray.grid.geometry import _build_polygon_shells, _build_corrected_polygon_shells

from uxarray.core.dataset import UxDataset

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

dsfiles_mf_ne30 = str(
    current_path) + "/meshfiles/ugrid/outCSne30/outCSne30_*.nc"

gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
dsfile_v1_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"


class TestDataArray(TestCase):

    def test_to_dataset(self):
        """Tests the conversion of UxDataArrays to a UXDataset."""
        uxds = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)
        uxds_converted = uxds['psi'].to_dataset()

        assert isinstance(uxds_converted, UxDataset)
        assert uxds_converted.uxgrid == uxds.uxgrid


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
        assert gdf_geoflow_grid.shape == (uxds_geoflow.uxgrid.n_face, 1)

        ### n30
        uxds_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

        gdf_geoflow_data = uxds_ne30['psi'].to_geodataframe()

        assert gdf_geoflow_data.shape == (uxds_ne30.uxgrid.n_face, 2)

    def test_to_polycollection(self):
        """Tests the conversion to ``PolyCollection``"""
        ### geoflow
        uxds_geoflow = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)

        # v1 is mapped to nodes, should raise a value error
        with self.assertRaises(ValueError):
            uxds_geoflow['v1'].to_polycollection()

        # grid conversion
        pc_geoflow_grid, _ = uxds_geoflow.uxgrid.to_polycollection()

        polygon_shells = _build_polygon_shells(
            uxds_geoflow.uxgrid.node_lon.values,
            uxds_geoflow.uxgrid.node_lat.values,
            uxds_geoflow.uxgrid.face_node_connectivity.values,
            uxds_geoflow.uxgrid.n_face, uxds_geoflow.uxgrid.n_max_face_nodes,
            uxds_geoflow.uxgrid.n_nodes_per_face.values)

        corrected_polygon_shells, _ = _build_corrected_polygon_shells(
            polygon_shells)

        # number of elements
        assert len(pc_geoflow_grid._paths) == len(corrected_polygon_shells)

        # ### n30
        uxds_ne30 = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

        polygon_shells = _build_polygon_shells(
            uxds_ne30.uxgrid.node_lon.values, uxds_ne30.uxgrid.node_lat.values,
            uxds_ne30.uxgrid.face_node_connectivity.values,
            uxds_ne30.uxgrid.n_face, uxds_ne30.uxgrid.n_max_face_nodes,
            uxds_ne30.uxgrid.n_nodes_per_face.values)

        corrected_polygon_shells, _ = _build_corrected_polygon_shells(
            polygon_shells)

        pc_geoflow_data, _ = uxds_ne30['psi'].to_polycollection()

        assert len(pc_geoflow_data._paths) == len(corrected_polygon_shells)

    def test_geodataframe_caching(self):
        uxds = ux.open_dataset(gridfile_ne30, dsfile_var2_ne30)

        gdf_start = uxds['psi'].to_geodataframe()

        gdf_next = uxds['psi'].to_geodataframe()

        # with caching, they point to the same area in memory
        assert gdf_start is gdf_next

        gdf_end = uxds['psi'].to_geodataframe(override=True)

        # override will recompute the grid
        assert gdf_start is not gdf_end

    def test_nodal_average(self):

        # test on a node-centered dataset
        uxds = ux.open_dataset(gridfile_geoflow, dsfile_v1_geoflow)

        v1_nodal_average = uxds['v1'].nodal_average()

        # final dimension should match number of faces
        self.assertEquals(v1_nodal_average.shape[-1], uxds.uxgrid.n_face)

        # all other dimensions should remain unchanged
        self.assertEquals(uxds['v1'].shape[0:-1], v1_nodal_average.shape[0:-1])

        # test on a sample mesh with 4 verts
        verts = [[[-170, 40], [180, 30], [165, 25], [-170, 20]]]
        data = [1, 2, 3, 4]

        uxgrid = ux.open_grid(verts, latlon=True)

        uxda = ux.UxDataArray(uxgrid=uxgrid, data=data, dims=('n_node'))

        uxda_nodal_average = uxda.nodal_average()

        # resulting data should be the mean of the corner nodes of the single face
        self.assertEquals(uxda_nodal_average, np.mean(data))
