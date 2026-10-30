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


def _face_latitude(uxgrid):
    """Face-centered data equal to the latitude of each face center."""
    return ux.UxDataArray(
        xr.DataArray(uxgrid.face_lat.values, dims="n_face", name="lat"), uxgrid=uxgrid
    )


def _contour_lines(contours):
    """(level, x, y) of each line of a contour plot, with or without labels."""
    if isinstance(contours, hv.Overlay):
        contours = contours.get(0)
    name = contours.vdims[0].name
    lines = []
    # each path holds the lines of one level, separated by NaN
    for p in contours.data:
        level, x, y = float(np.atleast_1d(p[name])[0]), np.asarray(p["x"]), np.asarray(p["y"])
        breaks = np.flatnonzero(np.isnan(x))
        lines += [(level, xs, ys) for xs, ys in zip(np.split(x, breaks), np.split(y, breaks))]
    # np.split keeps the NaN at the start of every line after the first
    return [(level, x[np.isfinite(x)], y[np.isfinite(y)]) for level, x, y in lines]


@pytest.mark.parametrize("method", ["interpolated", "edges"])
def test_contour_face_centered(gridpath, datasetpath, method):
    """Tests that contours of face-centered data are returned on both backends and can be overlaid."""
    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"))

    for backend in ['matplotlib', 'bokeh']:
        contours = uxds['psi'].plot.contour(method=method, backend=backend, color="black")

        # the lines, with the level as their value, and their labels
        lines, labels = contours
        assert isinstance(lines, hv.Contours) and isinstance(labels, hv.Labels)
        assert len(lines.data) > 0
        assert lines.vdims[0].name == "level" and lines.vdims[0].label == "psi"

        hv.renderer(backend).get_plot(uxds['psi'].plot(backend=backend) * contours)


def test_contour_node_centered(gridpath, datasetpath):
    """Tests contours of node-centered data, which are interpolated by default."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))
    v1 = uxds['v1'][0][0]

    contours = v1.plot.contour(backend="matplotlib", labels=False)
    assert isinstance(contours, hv.Contours)
    assert len(contours.data) > 0
    assert len(contours.data) == len(v1.plot.contour(method="interpolated", labels=False).data)


def test_contour_levels(gridpath, datasetpath):
    """Tests that explicit levels are used and levels outside the data range give no lines."""
    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"))

    for method in ["interpolated", "edges"]:
        contours = uxds['psi'].plot.contour(levels=[0.6, 0.9, 1.2, 99.0], method=method)
        assert {level for level, _, _ in _contour_lines(contours)} == {0.6, 0.9, 1.2}
        # one path per level, holding all the lines of that level
        assert len(contours.get(0).data) == 3
        assert _contour_lines(uxds['psi'].plot.contour(levels=[99.0], method=method)) == []

        # an integer asks for a number of levels, which then fall inside the data range
        levels = {level for level, _, _ in _contour_lines(uxds['psi'].plot.contour(levels=4, method=method))}
        assert 1 <= len(levels) <= 6
        assert uxds['psi'].min() < min(levels) and max(levels) < uxds['psi'].max()

        # a level given twice is drawn once
        once = _contour_lines(uxds['psi'].plot.contour(levels=[0.9], method=method))
        twice = _contour_lines(uxds['psi'].plot.contour(levels=[0.9, 0.9], method=method))
        assert len(twice) == len(once)
        assert all(np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) for a, b in zip(once, twice))

    # a level at the smallest value: "edges" draws the line around the faces at that value
    # (here the faces outside a band of latitude), "interpolated" draws nothing
    uxgrid = uxds.uxgrid
    inside = (np.abs(uxgrid.face_lat.values) < 30).astype(float)
    mask = ux.UxDataArray(xr.DataArray(inside, dims="n_face", name="mask"), uxgrid=uxgrid)
    assert {level for level, _, _ in _contour_lines(mask.plot.contour(levels=[0.0], method="edges"))} == {0.0}
    assert _contour_lines(mask.plot.contour(levels=[0.0], method="interpolated")) == []
    assert _contour_lines(mask.plot.contour(levels=[1.0], method="edges")) == []


# the first grid has nodes on the antimeridian, the second has cells that cross it
GLOBAL_GRIDS = [("ugrid", "outCSne30", "outCSne30.ug"), ("mpas", "QU", "mesh.QU.1920km.151026.nc")]


@pytest.mark.parametrize("grid", GLOBAL_GRIDS)
def test_contour_interpolated_values(gridpath, grid):
    """Contours of a field equal to latitude lie on that latitude, all the way around the globe."""
    uxgrid = ux.open_grid(gridpath(*grid))
    levels = [-30.5, 10.5, 45.5]

    lines = _contour_lines(_face_latitude(uxgrid).plot.contour(levels=levels, method="interpolated"))
    for level in levels:
        at_level = [(x, y) for line_level, x, y in lines if line_level == level]
        for x, y in at_level:
            np.testing.assert_allclose(y, level, atol=1e-8)
            assert np.abs(x).max() <= 180
        # no gap at the antimeridian
        assert sum(np.abs(np.diff(x)).sum() for x, _ in at_level) == pytest.approx(360)


@pytest.mark.parametrize("grid", GLOBAL_GRIDS)
def test_contour_edges_follow_grid(gridpath, grid):
    """Edge contours are made of the edges between faces on either side of the level."""
    uxgrid = ux.open_grid(gridpath(*grid))
    uxda = _face_latitude(uxgrid)
    level = 10.5

    # "edges" is the default method for face-centered data
    lines = _contour_lines(uxda.plot.contour(levels=[level]))

    # -180 and 180 are the same longitude
    def points(lon, lat):
        lon = np.radians(lon)
        return list(zip(np.round(np.cos(lon), 6), np.round(np.sin(lon), 6), np.round(lat, 6)))

    # every point is a node of the grid, except where a line is cut at the antimeridian
    nodes = set(points(uxgrid.node_lon.values, uxgrid.node_lat.values))
    n_segments, n_cut_ends = 0, 0
    for _, x, y in lines:
        assert np.abs(x).max() <= 180
        not_a_node = np.array([point not in nodes for point in points(x, y)])
        assert (np.abs(x[not_a_node]) == 180).all()
        n_cut_ends += np.count_nonzero(not_a_node)
        n_segments += len(x) - 1

    # one segment per edge whose two faces are on either side of the level,
    # and two for an edge that is cut at the antimeridian
    edge_faces = uxgrid.edge_face_connectivity.values
    above = uxda.values > level
    between = above[edge_faces[:, 0]] != above[edge_faces[:, 1]]
    lon = uxgrid.node_lon.values[uxgrid.edge_node_connectivity.values]
    cut = between & (np.abs(lon[:, 0] - lon[:, 1]) > 180) & (np.abs(lon) < 180).all(axis=1)
    assert n_cut_ends == 2 * np.count_nonzero(cut)
    assert n_segments == np.count_nonzero(between) + np.count_nonzero(cut)


def test_contour_join_segments():
    """Segments are joined into lines, open lines first from one of their ends, then closed loops."""
    from uxarray.plot.contour import _join_segments

    # a closed loop 0-1-2, and an open line 5-6-7 given in reverse
    nodes, offsets = _join_segments(np.array([0, 1, 2, 6, 7]), np.array([1, 2, 0, 5, 6]))
    assert [line.tolist() for line in np.split(nodes, offsets[1:-1])] == [[5, 6, 7], [0, 1, 2, 0]]
    assert offsets.tolist() == [0, 3, 7]

    # no segments give no lines
    nodes, offsets = _join_segments(np.array([], dtype=int), np.array([], dtype=int))
    assert len(nodes) == 0 and offsets.tolist() == [0]


def test_contour_labels(gridpath):
    """Lines are labeled with their level by default, each level at the middle of its longest lines."""
    uxda = _face_latitude(ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug")))
    levels = [-30.5, 10.5, 45.5]

    lines, labels = uxda.plot.contour(levels=levels, method="interpolated")
    text = labels.dimension_values("text")

    # every level is labeled, at most three times, and each label is on a line of its level
    assert sorted(set(text)) == sorted(f"{level:g}" for level in levels)
    assert max(np.count_nonzero(text == label) for label in set(text)) <= 3
    np.testing.assert_allclose(labels.dimension_values("y"), text.astype(float), atol=1e-8)

    # without labels, the lines alone are returned
    assert isinstance(uxda.plot.contour(levels=levels, labels=False), hv.Contours)

    # a fraction labels that share of the levels, spread evenly from the lowest
    five_levels = [-60.5, -30.5, 0.5, 30.5, 60.5]
    labels = uxda.plot.contour(levels=five_levels, labels=0.5).get(1)
    assert sorted(set(labels.dimension_values("text")), key=float) == ["-60.5", "0.5", "60.5"]
    assert isinstance(uxda.plot.contour(levels=levels, labels=0), hv.Contours)
    for bad in (1.5, -0.5, "half"):
        with pytest.raises(ValueError, match="labels"):
            uxda.plot.contour(levels=levels, labels=bad)


@pytest.mark.parametrize("name", ["x", "y"])
def test_contour_data_named_like_a_coordinate(gridpath, name):
    """Data named "x" or "y" gives the same lines as any other name, and hover shows its name."""
    uxda = _face_latitude(ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug")))
    levels = [-30.5, 10.5]

    expected = _contour_lines(uxda.plot.contour(levels=levels, backend="bokeh"))
    contours = uxda.rename(name).plot.contour(levels=levels, backend="bokeh")
    lines = _contour_lines(contours)

    assert len(lines) == len(expected)
    for (level, x, y), (expected_level, expected_x, expected_y) in zip(lines, expected):
        assert level == expected_level
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(y, expected_y)

    tools = hv.renderer("bokeh").get_plot(contours.get(0)).state.tools
    tooltips = [tool.tooltips for tool in tools if type(tool).__name__ == "HoverTool"][0]
    assert name in [label for label, _ in tooltips]


def test_contour_triangles_of_dented_faces(gridpath):
    """Dented faces are split into triangles that stay inside them, also when no corner can see
    the whole face, and on the dual cells of a FESOM mesh, some of which are dented."""
    from uxarray.plot.contour import _fan_triangles, _wrap_longitude

    def areas(lon, lat, triangles):
        x, y = lon[triangles], lat[triangles]
        return 0.5 * ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))

    # a dart, whose first corner cannot see the corner in the dent; and a C shape, where
    # no corner can see every other one
    for corners in ([(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)], [(0, 0), (5, 0), (5, 1), (1, 1), (1, 2), (5, 2), (5, 3), (0, 3)]):
        lon, lat = np.array(corners, dtype=float).T
        triangles = _fan_triangles(np.arange(len(lon))[None, :], np.array([len(lon)]), lon, lat)
        face = 0.5 * np.sum(lon * np.roll(lat, -1) - np.roll(lon, -1) * lat)
        assert len(triangles) == len(lon) - 2
        assert np.all(areas(lon, lat, triangles) * np.sign(face) > 0)
        np.testing.assert_allclose(areas(lon, lat, triangles).sum(), face)

    # on the dual of a FESOM mesh, the triangles cover the faces exactly once (with a fan from
    # every face's first corner they covered 2.8% more, where they reached outside dented faces)
    dual = ux.open_grid(gridpath("fesom", "soufflet-netcdf", "grid.nc")).get_dual()
    connectivity, counts = dual.face_node_connectivity.values, dual.n_nodes_per_face.values
    lon, lat = _wrap_longitude(dual.node_lon.values), dual.node_lat.values
    triangles = _fan_triangles(connectivity, counts, lon, lat)
    face_area = sum(
        abs(np.sum(lon[nodes] * np.roll(lat[nodes], -1) - np.roll(lon[nodes], -1) * lat[nodes])) / 2
        for nodes in (connectivity[face, : counts[face]] for face in range(dual.n_face))
    )
    np.testing.assert_allclose(np.abs(areas(lon, lat, triangles)).sum(), face_area)


def test_contour_options(gridpath):
    """Options have one spelling on both backends, and Bokeh shows the level of a line on hover."""
    uxda = _face_latitude(ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")))

    for backend, line_width in [("matplotlib", "linewidth"), ("bokeh", "line_width")]:
        contours = uxda.plot.contour(levels=[10.5], backend=backend, line_width=3, color="black")
        assert contours.get(0).opts.get(backend=backend, defaults=False).kwargs[line_width] == 3
        hv.renderer(backend).get_plot(contours)

    def tools(contours):
        return [type(tool).__name__ for tool in hv.renderer("bokeh").get_plot(contours).state.tools]

    assert "HoverTool" in tools(uxda.plot.contour(levels=[10.5], backend="bokeh"))
    assert "HoverTool" not in tools(uxda.plot.contour(levels=[10.5], backend="bokeh", hover=False))


def test_contour_projection(gridpath):
    """Tests that a projection gives GeoViews elements that render, without the lines that are not on the map."""
    import geoviews as gv

    uxda = _face_latitude(ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")))
    levels = [-45.5, 45.5]

    projection = ccrs.Robinson()
    contours = uxda.plot.contour(levels=levels, projection=projection, color="black", backend="matplotlib")
    lines, labels = contours
    assert isinstance(lines, gv.Contours) and isinstance(labels, gv.Labels)
    hv.renderer("matplotlib").get_plot(uxda.plot(projection=projection, backend="matplotlib") * contours)

    # seen from above the north pole, the southern line is not on the map.
    # GeoViews raises an IndexError for such a line, so it is dropped
    for method in ["interpolated", "edges"]:
        n_lines = len(_contour_lines(uxda.plot.contour(levels=levels, method=method)))
        contours = uxda.plot.contour(levels=levels, method=method, projection=ccrs.Orthographic(0, 90), backend="matplotlib")
        assert 0 < len(_contour_lines(contours)) < n_lines
        hv.renderer("matplotlib").get_plot(contours)


@pytest.mark.parametrize("projection", [
    ccrs.Orthographic(0, 90), ccrs.PlateCarree(central_longitude=180), ccrs.LambertConformal(), ccrs.Robinson(-150)
])
def test_contour_projection_drops_same_lines_as_projecting_each(gridpath, projection):
    """Lines found on the map from their points alone are the lines whose projection is not empty."""
    import shapely
    from uxarray.plot.contour import _compute_contours, _drop_lines_outside_projection

    uxda = _face_latitude(ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug")))
    for method in ["interpolated", "edges"]:
        contours = _compute_contours(uxda, levels=[-45.5, 0.5, 10.5, 45.5], method=method)
        expected = [
            line for _, line in contours
            if not projection.project_geometry(shapely.LineString(line), ccrs.PlateCarree()).is_empty
        ]
        kept = [line for _, line in _drop_lines_outside_projection(contours, projection)]
        assert len(kept) == len(expected)
        assert all(a is b for a, b in zip(kept, expected))


def test_contour_frame(gridpath):
    """Lines alone get the frame of .plot(): equal axis scaling on a map, 700 x 300 without a
    projection. Overlaid on a shaded plot, the shaded plot's frame and area are kept."""
    from bokeh.models import Plot

    uxda = _face_latitude(ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")))
    projection = ccrs.Robinson()

    def frame(element):
        figure = list(hv.renderer("bokeh").get_plot(element).state.select({"type": Plot}))[0]
        return figure.frame_width or figure.width, figure.frame_height or figure.height, figure.match_aspect

    shading = uxda.plot(projection=projection, backend="bokeh")
    contours = uxda.plot.contour(levels=[-30.5, 30.5], projection=projection, backend="bokeh")
    width, height, equal = frame(contours)
    assert equal and height == frame(shading)[1]
    assert abs(width - frame(shading)[0]) <= 0.02 * frame(shading)[0]
    assert frame(shading * contours) == frame(shading)

    assert frame(uxda.plot.contour(levels=[-30.5, 30.5], backend="bokeh"))[:2] == (700, 300)
    assert frame(uxda.plot.contour(levels=[-30.5, 30.5], backend="bokeh", width=500))[0] == 500

    # nor do the lines widen the shaded area, whose polygons are in single precision
    def ranges(element):
        figure = hv.renderer("bokeh").get_plot(element).state
        return figure.x_range.start, figure.x_range.end, figure.y_range.start, figure.y_range.end

    uxda = _face_latitude(ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc")))
    shading = uxda.plot(backend="bokeh")
    assert ranges(shading * uxda.plot.contour(levels=[-30.5, 30.5], backend="bokeh")) == ranges(shading)


def test_contour_missing_values(gridpath):
    """Faces without data are left out instead of raising."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    uxda = _face_latitude(uxgrid)

    for method in ["interpolated", "edges"]:
        lines = _contour_lines(uxda.where(uxgrid.face_lon.values < 0).plot.contour(levels=[10.0], method=method))
        assert len(lines) > 0
        assert all(x.max() < 5 for _, x, _ in lines)

        # no data at all gives no lines
        assert _contour_lines((uxda * np.nan).plot.contour(method=method)) == []

    # nor does a grid whose dual has no faces to split into triangles
    two_faces = _face_latitude(ux.open_grid(gridpath("scrip", "scrip_radians", "scrip_radians_grid.nc")))
    assert two_faces.uxgrid.get_dual().n_face == 0
    assert _contour_lines(two_faces.plot.contour(method="interpolated")) == []


def test_contour_invalid_input(gridpath, datasetpath):
    """Tests the errors for unsupported methods, levels, options, dimensions and data locations."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))
    v1 = uxds['v1'][0][0]

    with pytest.raises(ValueError, match="Unsupported method"):
        v1.plot.contour(method="smooth")

    with pytest.raises(ValueError, match="levels must be at least 1"):
        v1.plot.contour(levels=0)
    # neither a number of levels nor a sequence of them
    for levels in (10.0, None):
        with pytest.raises(ValueError, match="levels must be an integer or a one-dimensional sequence"):
            v1.plot.contour(levels=levels)

    # the Matplotlib spelling of an option; the error names the one to use
    with pytest.raises(ValueError, match="line_width"):
        v1.plot.contour(linewidth=2)

    # more than one dimension that is not of length 1
    with pytest.raises(ux.errors.DimensionError):
        uxds['v1'].plot.contour()

    # "edges" needs face-centered data, and edge-centered data is not supported
    with pytest.raises(ux.errors.DataCenteringError):
        v1.plot.contour(method="edges")
    uxgrid = uxds.uxgrid
    edge_data = ux.UxDataArray(xr.DataArray(np.zeros(uxgrid.n_edge), dims="n_edge"), uxgrid=uxgrid)
    with pytest.raises(ux.errors.DataCenteringError):
        edge_data.plot.contour()


def test_contour_data_not_on_grid(gridpath):
    """Data without a grid dimension is still contoured by xarray, as it was before plot.contour() was added."""
    from matplotlib.contour import QuadContourSet

    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))
    values = np.random.default_rng(0).random((4, 6, uxgrid.n_face))
    uxda = ux.UxDataArray(xr.DataArray(values, dims=("time", "lev", "n_face"), name="temp"), uxgrid=uxgrid)

    for section in (uxda.mean("n_face"), uxda.weighted_mean()):
        assert section.dims == ("time", "lev")
        assert isinstance(section.plot.contour(), QuadContourSet)
        assert isinstance(section.plot.contour(levels=5, colors="k"), QuadContourSet)
    plt.close("all")


def test_contour_faces_with_repeated_nodes(gridpath):
    """Faces padded by repeating a node do not give triangles that Matplotlib cannot contour."""
    from uxarray.plot.contour import _triangulation

    # every face of this grid is a quadrilateral stored with five nodes, the last one repeated
    uxgrid = ux.open_grid(gridpath("ugrid", "ne120_TCsubset", "ne120_TCsubset.ug"))
    face_nodes = uxgrid.face_node_connectivity.values
    assert (face_nodes[:, -1] == face_nodes[:, -2]).all()

    for dim in ("n_face", "n_node"):
        _, _, triangles, _ = _triangulation(uxgrid, dim)
        assert len(triangles) > 0
        # no repeated corners, and no edge in more than two triangles
        assert (np.sort(triangles, axis=1)[:, 1:] != np.sort(triangles, axis=1)[:, :-1]).all()
        edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        assert np.unique(edges, axis=0, return_counts=True)[1].max() <= 2

    level = float(uxgrid.face_lat.mean())
    lines = _contour_lines(_face_latitude(uxgrid).plot.contour(levels=[level], method="interpolated"))
    assert len(lines) > 0
    for _, _, y in lines:
        np.testing.assert_allclose(y, level, atol=1e-8)
