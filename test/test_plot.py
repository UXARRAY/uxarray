import os
import uxarray as ux

from unittest import TestCase
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
datafile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"

gridfile_mpas = current_path / "meshfiles" / "mpas" / "QU" / "oQU480.231010.nc"

gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"

datafile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

grid_files = [gridfile_geoflow, gridfile_mpas]


class TestPlot(TestCase):

    def test_topology(self):
        """Tests execution on Grid elements."""
        uxgrid = ux.open_grid(gridfile_mpas)

        for backend in ['matplotlib', 'bokeh']:

            uxgrid.plot(backend=backend)

            uxgrid.plot.mesh(backend=backend)

            uxgrid.plot.edges(backend=backend)

            uxgrid.plot.nodes(backend=backend)

            uxgrid.plot.face_centers(backend=backend)

            if uxgrid.edge_lon is not None:
                uxgrid.plot.edge_centers(backend=backend)

    def test_face_centered_data(self):
        """Tests execution of plotting methods on face-centered data."""

        uxds = ux.open_dataset(gridfile_mpas, gridfile_mpas)

        for backend in ['matplotlib', 'bokeh']:

            uxds['bottomDepth'].plot(backend=backend)

            uxds['bottomDepth'].plot.polygons(backend=backend)

            uxds['bottomDepth'].plot.points(backend=backend)

            uxds['bottomDepth'].plot.rasterize(method='polygon',
                                               backend=backend)

    def test_face_centered_remapped_dim(self):
        """Tests execution of plotting method on a data variable whose
        dimension needed to be re-mapped."""
        uxds = ux.open_dataset(gridfile_ne30, datafile_ne30)

        for backend in ['matplotlib', 'bokeh']:

            uxds['psi'].plot(backend=backend)

            uxds['psi'].plot.polygons(backend=backend)

            uxds['psi'].plot.points(backend=backend)

            uxds['psi'].plot.rasterize(method='polygon', backend=backend)

    def test_node_centered_data(self):
        """Tests execution of plotting methods on node-centered data."""

        uxds = ux.open_dataset(gridfile_geoflow, datafile_geoflow)

        for backend in ['matplotlib', 'bokeh']:
            uxds['v1'][0][0].plot(backend=backend)

            uxds['v1'][0][0].plot.points(backend=backend)

            uxds['v1'][0][0].nodal_average().plot.polygons(backend=backend)


    def test_clabel(self):
        """Tests the execution of passing in a custom clabel."""

        uxds = ux.open_dataset(gridfile_geoflow, datafile_geoflow)

        raster_no_clabel = uxds['v1'][0][0].plot.rasterize(method='point')

        raster_with_clabel = uxds['v1'][0][0].plot.rasterize(method='point', clabel='Foo')



class TestXarrayMethods(TestCase):

    def test_dataset(self):
        """Tests whether a Xarray DataArray method can be called through the
        UxDataArray plotting accessor."""
        uxds = ux.open_dataset(gridfile_geoflow, datafile_geoflow)

        # plot.hist() is an xarray method
        assert hasattr(uxds['v1'].plot, 'hist')


    def test_dataarray(self):
        """Tests whether a Xarray Dataset method can be called through the
        UxDataset plotting accessor."""
        uxds = ux.open_dataset(gridfile_geoflow, datafile_geoflow)

        # plot.scatter() is an xarray method
        assert hasattr(uxds.plot, 'scatter')
