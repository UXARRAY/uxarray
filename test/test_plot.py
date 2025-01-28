import os
import uxarray as ux
import holoviews as hv
import pytest
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
datafile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"
gridfile_mpas = current_path / "meshfiles" / "mpas" / "QU" / "oQU480.231010.nc"
gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
datafile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

grid_files = [gridfile_geoflow, gridfile_mpas]
grid_plot_routines = ['points', 'nodes', 'node_coords', '']

def test_topology():
    """Tests execution on Grid elements."""
    uxgrid = ux.open_grid(gridfile_mpas)

    for backend in ['matplotlib', 'bokeh']:
        uxgrid.plot(backend=backend)
        uxgrid.plot.mesh(backend=backend)
        uxgrid.plot.edges(backend=backend)
        uxgrid.plot.nodes(backend=backend)
        uxgrid.plot.node_coords(backend=backend)
        uxgrid.plot.corner_nodes(backend=backend)
        uxgrid.plot.face_centers(backend=backend)
        uxgrid.plot.face_coords(backend=backend)
        uxgrid.plot.edge_centers(backend=backend)
        uxgrid.plot.edge_coords(backend=backend)

def test_face_centered_data():
    """Tests execution of plotting methods on face-centered data."""
    uxds = ux.open_dataset(gridfile_mpas, gridfile_mpas)

    for backend in ['matplotlib', 'bokeh']:
        assert isinstance(uxds['bottomDepth'].plot(backend=backend, dynamic=True), hv.DynamicMap)
        assert isinstance(uxds['bottomDepth'].plot.polygons(backend=backend, dynamic=True), hv.DynamicMap)
        assert isinstance(uxds['bottomDepth'].plot.points(backend=backend), hv.Points)

def test_face_centered_remapped_dim():
    """Tests execution of plotting method on a data variable whose dimension needed to be re-mapped."""
    uxds = ux.open_dataset(gridfile_ne30, datafile_ne30)

    for backend in ['matplotlib', 'bokeh']:
        assert isinstance(uxds['psi'].plot(backend=backend, dynamic=True), hv.DynamicMap)
        assert isinstance(uxds['psi'].plot.polygons(backend=backend, dynamic=True), hv.DynamicMap)
        assert isinstance(uxds['psi'].plot.points(backend=backend), hv.Points)

def test_node_centered_data():
    """Tests execution of plotting methods on node-centered data."""
    uxds = ux.open_dataset(gridfile_geoflow, datafile_geoflow)

    for backend in ['matplotlib', 'bokeh']:
        assert isinstance(uxds['v1'][0][0].plot(backend=backend), hv.Points)
        assert isinstance(uxds['v1'][0][0].plot.points(backend=backend), hv.Points)
        assert isinstance(uxds['v1'][0][0].topological_mean(destination='face').plot.polygons(backend=backend, dynamic=True), hv.DynamicMap)

def test_clabel():
    """Tests the execution of passing in a custom clabel."""
    uxds = ux.open_dataset(gridfile_geoflow, datafile_geoflow)

    raster_no_clabel = uxds['v1'][0][0].plot.rasterize(method='point')
    raster_with_clabel = uxds['v1'][0][0].plot.rasterize(method='point', clabel='Foo')

def test_engine():
    """Tests different plotting engines."""
    uxds = ux.open_dataset(gridfile_mpas, gridfile_mpas)
    _plot_sp = uxds['bottomDepth'].plot.polygons(rasterize=True, dynamic=True, engine='spatialpandas')
    _plot_gp = uxds['bottomDepth'].plot.polygons(rasterize=True, dynamic=True, engine='geopandas')

    assert isinstance(_plot_sp, hv.DynamicMap)
    assert isinstance(_plot_gp, hv.DynamicMap)

def test_dataset_methods():
    """Tests whether a Xarray DataArray method can be called through the UxDataArray plotting accessor."""
    uxds = ux.open_dataset(gridfile_geoflow, datafile_geoflow)

    # plot.hist() is an xarray method
    assert hasattr(uxds['v1'].plot, 'hist')

def test_dataarray_methods():
    """Tests whether a Xarray Dataset method can be called through the UxDataset plotting accessor."""
    uxds = ux.open_dataset(gridfile_geoflow, datafile_geoflow)

    # plot.scatter() is an xarray method
    assert hasattr(uxds.plot, 'scatter')

def test_line():
    uxds = ux.open_dataset(gridfile_mpas, gridfile_mpas)
    _plot_line = uxds['bottomDepth'].zonal_average().plot.line()
    assert isinstance(_plot_line, hv.Curve)

def test_scatter():
    uxds = ux.open_dataset(gridfile_mpas, gridfile_mpas)
    _plot_line = uxds['bottomDepth'].zonal_average().plot.scatter()
    assert isinstance(_plot_line, hv.Scatter)
