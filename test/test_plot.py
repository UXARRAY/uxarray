import os
import uxarray as ux
import holoviews as hv
import pytest
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def test_topology(gridpath):
    """Tests execution on Grid elements."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))

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

def test_face_centered_data(gridpath):
    """Tests execution of plotting methods on face-centered data."""
    mesh_path = gridpath("mpas", "QU", "oQU480.231010.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)

    for backend in ['matplotlib', 'bokeh']:
        assert isinstance(uxds['bottomDepth'].plot(backend=backend, dynamic=True), hv.DynamicMap)
        assert isinstance(uxds['bottomDepth'].plot.polygons(backend=backend, dynamic=True), hv.DynamicMap)
        assert isinstance(uxds['bottomDepth'].plot.points(backend=backend), hv.Points)

def test_face_centered_remapped_dim(gridpath, datasetpath):
    """Tests execution of plotting method on a data variable whose dimension needed to be re-mapped."""
    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"))

    for backend in ['matplotlib', 'bokeh']:
        assert isinstance(uxds['psi'].plot(backend=backend, dynamic=True), hv.DynamicMap)
        assert isinstance(uxds['psi'].plot.polygons(backend=backend, dynamic=True), hv.DynamicMap)
        assert isinstance(uxds['psi'].plot.points(backend=backend), hv.Points)

def test_node_centered_data(gridpath, datasetpath):
    """Tests execution of plotting methods on node-centered data."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))

    for backend in ['matplotlib', 'bokeh']:
        assert isinstance(uxds['v1'][0][0].plot(backend=backend), hv.Points)
        assert isinstance(uxds['v1'][0][0].plot.points(backend=backend), hv.Points)
        assert isinstance(uxds['v1'][0][0].topological_mean(destination='face').plot.polygons(backend=backend, dynamic=True), hv.DynamicMap)


def test_engine(gridpath):
    """Tests different plotting engines."""
    mesh_path = gridpath("mpas", "QU", "oQU480.231010.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    _plot_sp = uxds['bottomDepth'].plot.polygons(rasterize=True, dynamic=True, engine='spatialpandas')
    _plot_gp = uxds['bottomDepth'].plot.polygons(rasterize=True, dynamic=True, engine='geopandas')

    assert isinstance(_plot_sp, hv.DynamicMap)
    assert isinstance(_plot_gp, hv.DynamicMap)

def test_dataset_methods(gridpath, datasetpath):
    """Tests whether a Xarray DataArray method can be called through the UxDataArray plotting accessor."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))

    # plot.hist() is an xarray method
    assert hasattr(uxds['v1'].plot, 'hist')

def test_dataarray_methods(gridpath, datasetpath):
    """Tests whether a Xarray Dataset method can be called through the UxDataset plotting accessor."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))

    # plot.scatter() is an xarray method
    assert hasattr(uxds.plot, 'scatter')

def test_line(gridpath):
    mesh_path = gridpath("mpas", "QU", "oQU480.231010.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    _plot_line = uxds['bottomDepth'].zonal_average().plot.line()
    assert isinstance(_plot_line, hv.Curve)

def test_scatter(gridpath):
    mesh_path = gridpath("mpas", "QU", "oQU480.231010.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    _plot_line = uxds['bottomDepth'].zonal_average().plot.scatter()
    assert isinstance(_plot_line, hv.Scatter)



def test_to_raster(gridpath):

    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.Robinson()},
        constrained_layout=True,
        figsize=(10, 5),
    )

    mesh_path = gridpath("mpas", "QU", "oQU480.231010.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)

    raster = uxds['bottomDepth'].to_raster(ax=ax)

    assert isinstance(raster, np.ndarray)


def test_collections_projection_kwarg(gridpath):
    import cartopy.crs as ccrs
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    with pytest.warns(FutureWarning):
        pc = uxgrid.to_polycollection(projection=ccrs.PlateCarree())
        lc = uxgrid.to_linecollection(projection=ccrs.PlateCarree())
