import uxarray as ux
import xarray as xr
import holoviews as hv
import pytest
import numpy as np
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from uxarray.grid.geometry import _central_longitude_of


def _cartopy_projections(convert_to='instance', *, include_local=True, errors='ignore', skip_private=True):
    """returns a list of all cartopy.crs projections.
    (Useful helper function for looping across all projections during tests.)

    convert_to: 'instance', 'class', or 'name', default 'instance'
        'instance' --> return a list of instances of the projection classes.
                    (instantiated with no arguments.)
        'class' --> return a list of the projection classes themselves.
        'name' --> return a list of the names of the projection classes.
    include_local: bool or 'only', default True
        whether to include projections local to a particular area.
            These are: EuroPP, LambertZoneII, OSGB, OSNI.
        True --> yes, include them.
        False --> no, do not include them.
        'only' --> only include these local projections; exclude all other projections.
    errors: 'ignore', 'warn', or 'raise', default 'ignore'
        'ignore' --> skip any projection classes that can't be instantiated with no arguments.
        'warn' --> issue a warning for any failures.
        'raise' --> raise an exception for any failures.
    skip_private: bool, default True
        if True, skip any projection classes whose names start with '_'.
    """
    result = []
    LOCAL_PROJECTIONS = {'EuroPP', 'LambertZoneII', 'OSGB', 'OSNI'}
    for name in dir(ccrs):
        if skip_private and name.startswith('_'):
            continue
        if include_local == 'only' and name not in LOCAL_PROJECTIONS:
            continue
        elif not include_local and name in LOCAL_PROJECTIONS:
            continue
        obj = getattr(ccrs, name)
        if isinstance(obj, type) and issubclass(obj, ccrs.Projection) and obj is not ccrs.Projection:
            if convert_to == 'instance':
                try:
                    result.append(obj())
                except Exception as err:
                    if errors == 'raise':
                        raise
                    elif errors == 'warn':
                        warnings.warn(f"Could not instantiate {name}: {err}")
                    else:
                        assert errors == 'ignore'
                        continue
            elif convert_to == 'class':
                result.append(obj)
            else:
                assert convert_to == 'name'
                result.append(name)
    return result


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

import hvplot.xarray  # registers .hvplot accessor

def test_line(gridpath):
    mesh_path = gridpath("mpas", "QU", "oQU480.231010.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    _plot_line = uxds['bottomDepth'].zonal_average().hvplot.line()
    assert isinstance(_plot_line, hv.Curve)

def test_scatter(gridpath):
    mesh_path = gridpath("mpas", "QU", "oQU480.231010.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    _plot_scatter = uxds['bottomDepth'].zonal_average().hvplot.scatter()
    assert isinstance(_plot_scatter, hv.Scatter)



def test_to_raster(gridpath):

    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.Robinson()},
        constrained_layout=True,
        figsize=(10, 5),
    )

    mesh_path = gridpath("mpas", "QU", "oQU480.231010.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)

    with pytest.warns(UserWarning, match=r"Axes extent was default"):
        raster = uxds['bottomDepth'].to_raster(ax=ax)

    assert isinstance(raster, np.ndarray)


def test_to_raster_with_extra_dims(gridpath):
    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.Robinson()},
        constrained_layout=True,
        figsize=(10, 5),
    )

    mesh_path = gridpath("mpas", "QU", "oQU480.231010.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)

    da = uxds['bottomDepth'].expand_dims(time=[0])
    with pytest.warns(UserWarning, match=r"Axes extent was default"):
        raster = da.to_raster(ax=ax)

    assert isinstance(raster, np.ndarray)



def test_to_raster_reuse_mapping(gridpath, tmpdir):

    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.Robinson()},
        constrained_layout=True,
        figsize=(10, 5),
    )

    mesh_path = gridpath("mpas", "QU", "oQU480.231010.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)

    # Returning
    with pytest.warns(UserWarning, match=r"Axes extent was default"):
        raster1, pixel_mapping = uxds['bottomDepth'].to_raster(
            ax=ax, pixel_ratio=0.5, return_pixel_mapping=True
        )
    assert isinstance(raster1, np.ndarray)
    assert isinstance(pixel_mapping, xr.DataArray)

    # Reusing (passed pixel ratio overridden by pixel mapping attr)
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

    # Recover attrs from disk
    p = tmpdir / "pixel_mapping.nc"
    pixel_mapping.to_netcdf(p)
    with xr.open_dataarray(p) as da:
        xr.testing.assert_identical(da, pixel_mapping)
        for v1, v2 in zip(da.attrs.values(), pixel_mapping.attrs.values()):
            assert type(v1) is type(v2)
            if isinstance(v1, np.ndarray):
                assert v1.dtype == v2.dtype

    # Modified pixel mapping raises error
    pixel_mapping.attrs["ax_shape"] = (2, 3)
    with pytest.raises(ValueError, match=r"Provided pixel_mapping values incompatible with ax raster attrs: shape \(2, 3\) !="):
        _ = uxds['bottomDepth'].to_raster(
            ax=ax, pixel_mapping=pixel_mapping
        )



@pytest.mark.parametrize(
    "r1,r2",
    [
        (0.01, 0.07),
        (0.1, 0.5),
        (1, 2),
    ],
)
def test_to_raster_pixel_ratio(gridpath, r1, r2):
    assert r2 > r1

    _, ax = plt.subplots(
        subplot_kw={'projection': ccrs.Robinson()},
        constrained_layout=True,
    )

    mesh_path = gridpath("mpas", "QU", "oQU480.231010.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)

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

def test_matplotlib_backend_restored_after_switch(monkeypatch):
    """Regression test for #1537: switching HoloViews to matplotlib must restore
    the Matplotlib backend captured beforehand.

    A bare pytest can't reproduce the Jupyter inline-hook flip, so we simulate
    hv.extension flipping the backend and assert assign() restores it.
    """
    import holoviews as hv

    from uxarray.plot.utils import HoloviewsBackend

    original = matplotlib.get_backend()
    try:
        matplotlib.use("svg")  # the "user's" backend before plotting

        # Simulate hv.extension("matplotlib") flipping the active backend to agg.
        monkeypatch.setattr(hv.Store, "current_backend", "bokeh", raising=False)
        monkeypatch.setattr(hv, "extension", lambda *a, **k: matplotlib.use("agg"))

        be = HoloviewsBackend()
        be.assign("matplotlib")

        # assign() must restore the backend that was active before the switch.
        assert matplotlib.get_backend() == "svg"
    finally:
        matplotlib.use(original)


def test_inline_backend_reactivated_via_shell(monkeypatch):
    """Inside IPython, restoring an inline backend must re-run the shell's own
    backend activation (the public equivalent of ``%matplotlib inline``), which
    rebuilds the full display integration that ``hv.extension`` tore down.

    Simply calling ``mpl.use`` is not enough: it restores the backend name but
    leaves last-line figure auto-display broken (see #1537 / #1538).
    """
    import sys
    import types

    from uxarray.plot.utils import HoloviewsBackend

    inline_backend = "module://matplotlib_inline.backend_inline"
    calls = []

    class FakeShell:
        def enable_matplotlib(self, gui):
            calls.append(("enable_matplotlib", gui))

    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: FakeShell()

    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setattr(
        matplotlib, "use", lambda backend: calls.append(("use", backend))
    )

    be = HoloviewsBackend()
    be.matplotlib_backend = inline_backend
    be.reset_mpl_backend()

    # The module:// inline backend must be mapped to the "inline" gui name and
    # restored through the shell, without falling back to mpl.use.
    assert calls == [("enable_matplotlib", "inline")]


def test_reset_backend_falls_back_to_mpl_use_outside_ipython(monkeypatch):
    """Outside IPython (get_ipython() is None), restoration falls back to
    ``mpl.use`` with the stored backend."""
    import sys
    import types

    from uxarray.plot.utils import HoloviewsBackend

    calls = []
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setattr(matplotlib, "use", lambda backend: calls.append(("use", backend)))

    be = HoloviewsBackend()
    be.matplotlib_backend = "svg"
    be.reset_mpl_backend()

    assert calls == [("use", "svg")]


def test_plot_with_features(gridpath, datasetpath):
    """ensure can render a multiplot layout with geoviews features.
    Regression test for issue #1542.
    """
    gridpath = gridpath("ugrid", "geoflow-small", "grid.nc")
    uxds1 = ux.open_dataset(gridpath, datasetpath("ugrid", "geoflow-small", "v1.nc"))
    uxds2 = ux.open_dataset(gridpath, datasetpath("ugrid", "geoflow-small", "v2.nc"))
    uxds1 = uxds1.isel(time=0, meshLayers=0)  # plot currently expects 1D along n_node
    uxds2 = uxds2.isel(time=0, meshLayers=0)
    plot1 = uxds1["v1"].plot(features=['coastline'])
    plot2 = uxds2["v2"].plot(features=['coastline'])
    plot = plot1 + plot2
    # the crash associated with issue #1542 only occurs when actually trying to render:
    renderer = hv.renderer("matplotlib")
    renderer.get_plot(plot)


def test_central_longitude_of_handles_cartopy_026_platecarree():
    """Cartopy 0.26 changed PlateCarree from ``proj=eqc`` to ``proj=latlong``,
    which carries the prime meridian as ``pm`` and drops ``lon_0`` entirely.

    Reading ``lon_0`` directly raised ``KeyError: 'lon_0'`` on every plotting
    call that defaulted to PlateCarree. Stub the params rather than branching on
    the installed cartopy, so both layouts stay covered on either version.
    """
    class _FakeProjection:
        def __init__(self, params):
            self.proj4_params = params

    # cartopy >= 0.26 PlateCarree: pm, no lon_0
    assert _central_longitude_of(_FakeProjection({"proj": "latlong", "pm": 0.0})) == 0.0
    assert _central_longitude_of(_FakeProjection({"proj": "latlong", "pm": 30})) == 30.0

    # cartopy < 0.26 PlateCarree, and projections that still use lon_0
    assert _central_longitude_of(_FakeProjection({"proj": "eqc", "lon_0": 0.0})) == 0.0
    assert _central_longitude_of(_FakeProjection({"proj": "robin", "lon_0": 45})) == 45.0

    # neither key: fall back to 0.0 rather than crash, but do at least raise a warning.
    with pytest.warns(UserWarning, match=r"Could not determine central longitude"):
        assert _central_longitude_of(_FakeProjection({"proj": "weird"})) == 0.0


def test_central_longitude_of_for_all_nonlocal_projections():
    """Ensure _central_longitude_of correctly identify the central longitude
    for all non-local cartopy projections.
    """
    projections = _cartopy_projections("class", include_local=False)
    # At time of writing, this finds 34 projection classes.
    # Put a reasonably-high lower bound to account for possible future cartopy updates,
    # while also ensuring the _cartopy_projections helper is actually finding projections.
    assert len(projections) >= 30

    couldnt_instantiate = {}
    for cls in projections:
        for cenlon in [0, 30, -45, 270]:
            try:
                proj = cls(central_longitude=cenlon)
            except Exception as err:
                couldnt_instantiate[cls] = err
                continue
            assert _central_longitude_of(proj) == cenlon, f"Failed for {cls}, central_longitude={cenlon}"

    # At time of writing, this finds len(couldnt_instantiate)==3.
    # Put a reasonably-low upper bound to account for possible future cartopy updates,
    # while also ensuring that most cases were actually instantiated and tested above.
    assert len(couldnt_instantiate) <= 5


def test_plot_with_nonzero_central_longitude():
    """Ensure can render a plot using a projection with a nonzero central longitude.
    Regression test for issue #1795.
    """
    arr = ux.tutorial.open_dataset("outCSne30-vortex")['psi']

    # simple test first (easier to debug a single explicit case, if crash)
    plot_obj = arr.plot(projection=ccrs.Robinson(central_longitude=100))
    renderer = hv.renderer("matplotlib")
    renderer.get_plot(plot_obj)

    # exhaustively check all non-local projections which can be instantiated with
    # a central_longitude, and check across a few different central_longitude values.
    projections = _cartopy_projections("class", include_local=False)
    assert len(projections) >= 30  # ensure actually found most projections

    couldnt_instantiate = {}
    for cls in projections:
        for cenlon in [0, 30, -45, 270]:
            try:
                proj = cls(central_longitude=cenlon)
            except Exception as err:
                couldnt_instantiate[cls] = err
                continue
            plot_obj = arr.plot(projection=proj)
            renderer.get_plot(plot_obj)

    assert len(couldnt_instantiate) <= 5  # ensure most projections were actually tested


def test_plot_topology_with_explicit_projection(gridpath):
    """`plot.edges` with an explicit projection exercises the central-longitude
    lookup that regressed under cartopy 0.26.
    Regression test for issue #1780.
    """
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))
    plot0 = uxgrid.plot.edges(backend="matplotlib", projection=ccrs.PlateCarree())
    plot1 = uxgrid.plot.edges(backend="matplotlib", projection=ccrs.Robinson())
    plot2 = uxgrid.plot.edges(backend="matplotlib", projection=ccrs.Orthographic())

    # the crash associated with issue #1780 only occurs when actually trying to render,
    # and only for some projections, but definitely for at least one of the above,
    # so, try to render all three of them.
    renderer = hv.renderer("matplotlib")
    renderer.get_plot(plot0)
    renderer.get_plot(plot1)
    renderer.get_plot(plot2)
