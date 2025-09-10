import os
import uxarray as ux
import xarray as xr
import holoviews as hv
import pytest
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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



def test_to_raster():

    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.Robinson()},
        constrained_layout=True,
        figsize=(10, 5),
    )

    uxds = ux.open_dataset(gridfile_mpas, gridfile_mpas)

    raster = uxds['bottomDepth'].to_raster(ax=ax)

    assert isinstance(raster, np.ndarray)


def test_to_raster_reuse_mapping():

    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.Robinson()},
        constrained_layout=True,
        figsize=(10, 5),
    )

    uxds = ux.open_dataset(gridfile_mpas, gridfile_mpas)

    # Returning
    raster1, pixel_mapping = uxds['bottomDepth'].to_raster(
        ax=ax, pixel_ratio=0.5, return_pixel_mapping=True
    )
    assert isinstance(raster1, np.ndarray)
    assert isinstance(pixel_mapping, xr.DataArray)

    # Reusing
    with pytest.warns(UserWarning, match="Pixel ratio mismatch"):
        raster2 = uxds['bottomDepth'].to_raster(
            ax=ax, pixel_ratio=0.1, pixel_mapping=pixel_mapping
        )
    np.testing.assert_array_equal(raster1, raster2)

    # Data pass-through
    raster3, pixel_mapping_returned = uxds['bottomDepth'].to_raster(
        ax=ax, pixel_mapping=pixel_mapping, return_pixel_mapping=True
    )
    np.testing.assert_array_equal(raster1, raster3)
    assert pixel_mapping_returned is not pixel_mapping
    xr.testing.assert_identical(pixel_mapping_returned, pixel_mapping)
    assert np.shares_memory(pixel_mapping_returned, pixel_mapping)

    # Passing array-like pixel mapping works,
    # but now we need pixel_ratio to get the correct raster
    raster4_bad = uxds['bottomDepth'].to_raster(
        ax=ax, pixel_mapping=pixel_mapping.values.tolist()
    )
    raster4 = uxds['bottomDepth'].to_raster(
        ax=ax, pixel_ratio=0.5, pixel_mapping=pixel_mapping.values.tolist()
    )
    np.testing.assert_array_equal(raster1, raster4)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(raster1, raster4_bad)


@pytest.mark.parametrize(
    "r1,r2",
    [
        (0.01, 0.07),
        (0.1, 0.5),
        (1, 2),
    ],
)
def test_to_raster_pixel_ratio(r1, r2):
    assert r2 > r1

    _, ax = plt.subplots(
        subplot_kw={'projection': ccrs.Robinson()},
        constrained_layout=True,
    )

    uxds = ux.open_dataset(gridfile_mpas, gridfile_mpas)

    ax.set_extent((-20, 20, -10, 10), crs=ccrs.PlateCarree())
    raster1 = uxds['bottomDepth'].to_raster(ax=ax, pixel_ratio=r1)
    raster2 = uxds['bottomDepth'].to_raster(ax=ax, pixel_ratio=r2)

    assert isinstance(raster1, np.ndarray) and isinstance(raster2, np.ndarray)
    assert raster1.ndim == raster2.ndim == 2
    assert raster2.size > raster1.size
    fna1 = np.isnan(raster1).sum() / raster1.size
    fna2 = np.isnan(raster2).sum() / raster2.size
    assert fna1 != fna2
    assert fna1 == pytest.approx(fna2, abs=0.06 if r1 == 0.01 else 1e-3)

    f = r2 / r1
    d = np.array(raster2.shape) - f * np.array(raster1.shape)
    assert (d >= 0).all() and (d <= f - 1).all()


def test_collections_projection_kwarg():
    import cartopy.crs as ccrs
    uxgrid = ux.open_grid(gridfile_ne30)

    with pytest.warns(FutureWarning):
        pc = uxgrid.to_polycollection(projection=ccrs.PlateCarree())
        lc = uxgrid.to_linecollection(projection=ccrs.PlateCarree())
