import uxarray as ux
import xarray as xr
import holoviews as hv
import pytest
import numpy as np

import matplotlib
matplotlib.use("Agg")
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
    from uxarray.grid.geometry import _central_longitude_of

    class _FakeProjection:
        def __init__(self, params):
            self.proj4_params = params

    # cartopy >= 0.26 PlateCarree: pm, no lon_0
    assert _central_longitude_of(_FakeProjection({"proj": "latlong", "pm": 0.0})) == 0.0
    assert _central_longitude_of(_FakeProjection({"proj": "latlong", "pm": 30})) == 30.0

    # cartopy < 0.26 PlateCarree, and projections that still use lon_0
    assert _central_longitude_of(_FakeProjection({"proj": "eqc", "lon_0": 0.0})) == 0.0
    assert _central_longitude_of(_FakeProjection({"proj": "robin", "lon_0": 45})) == 45.0

    # neither key: fall back rather than raise
    assert _central_longitude_of(_FakeProjection({"proj": "weird"})) == 0.0


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
    """(level, x, y) of each line in a ``Contours`` element."""
    name = contours.vdims[0].name
    return [(float(np.atleast_1d(p[name])[0]), np.asarray(p["x"]), np.asarray(p["y"])) for p in contours.data]


@pytest.mark.parametrize("method", ["interpolated", "edges"])
def test_contour_face_centered(gridpath, datasetpath, method):
    """Tests that contours of face-centered data are returned on both backends and can be overlaid."""
    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"))

    for backend in ['matplotlib', 'bokeh']:
        contours = uxds['psi'].plot.contour(method=method, backend=backend)
        assert isinstance(contours, hv.Contours)
        assert len(contours.data) > 0
        assert contours.vdims[0].name == "psi"

        overlay = uxds['psi'].plot(backend=backend) * contours.opts(color="black")
        hv.renderer(backend).get_plot(overlay)


def test_contour_node_centered(gridpath, datasetpath):
    """Tests contours of node-centered data, which only support interpolation."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))
    v1 = uxds['v1'][0][0]

    # the default method for node-centered data is "interpolated"
    contours = v1.plot.contour(backend="matplotlib")
    assert isinstance(contours, hv.Contours)
    assert len(contours.data) > 0
    assert len(contours.data) == len(v1.plot.contour(method="interpolated").data)

    with pytest.raises(ux.errors.DataCenteringError):
        v1.plot.contour(method="edges")


def test_contour_levels(gridpath, datasetpath):
    """Tests that explicit levels are used and levels outside the data range give no lines."""
    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"))

    for method in ["interpolated", "edges"]:
        contours = uxds['psi'].plot.contour(levels=[0.6, 0.9, 1.2, 99.0], method=method)
        assert {level for level, _, _ in _contour_lines(contours)} == {0.6, 0.9, 1.2}

        # an integer asks for a number of levels, which then fall inside the data range
        levels = {level for level, _, _ in _contour_lines(uxds['psi'].plot.contour(levels=4, method=method))}
        assert 1 <= len(levels) <= 6
        assert uxds['psi'].min() < min(levels) and max(levels) < uxds['psi'].max()

    assert len(uxds['psi'].plot.contour(levels=[99.0]).data) == 0


def test_contour_interpolated_values(gridpath):
    """Contours of a field equal to latitude lie on that latitude."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))
    levels = [-30.0, 0.0, 45.0]

    lines = _contour_lines(_face_latitude(uxgrid).plot.contour(levels=levels, method="interpolated"))
    assert {level for level, _, _ in lines} == set(levels)
    for level, _, y in lines:
        np.testing.assert_allclose(y, level, atol=1e-8)


def test_contour_edges_follow_grid(gridpath):
    """Edge contours are made of the edges between faces on either side of the level."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    uxda = _face_latitude(uxgrid)
    level = 20.0

    # "edges" is the default method for face-centered data
    lines = _contour_lines(uxda.plot.contour(levels=[level]))

    # every point on the contour is a node of the grid
    nodes = set(zip(np.round(uxgrid.node_lon.values, 6), np.round(uxgrid.node_lat.values, 6)))
    n_segments = 0
    for _, x, y in lines:
        assert set(zip(np.round(x, 6), np.round(y, 6))) <= nodes
        n_segments += len(x) - 1

    # one segment per edge whose two faces are on either side of the level
    edge_faces = uxgrid.edge_face_connectivity.values
    edge_nodes = uxgrid.edge_node_connectivity.values
    above = uxda.values > level
    crossing = above[edge_faces[:, 0]] != above[edge_faces[:, 1]]
    not_wrapped = np.abs(np.diff(uxgrid.node_lon.values[edge_nodes], axis=1)[:, 0]) < 180
    assert n_segments == np.count_nonzero(crossing & not_wrapped)


def test_contour_projection(gridpath, datasetpath):
    """Tests that a projection returns a GeoViews element that renders with the projected data."""
    import geoviews as gv

    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"))
    projection = ccrs.Robinson()

    contours = uxds['psi'].plot.contour(projection=projection, color="black", backend="matplotlib")
    assert isinstance(contours, gv.Contours)

    overlay = uxds['psi'].plot(projection=projection, backend="matplotlib") * contours
    hv.renderer("matplotlib").get_plot(overlay)


def test_contour_projection_line_on_map_edge():
    """Lines that lie along the edge of the map are dropped, since they cannot be projected."""
    uxgrid = ux.Grid.from_healpix(zoom=4)
    lon, lat = np.radians(uxgrid.face_lon.values), np.radians(uxgrid.face_lat.values)
    values = np.cos(2 * lat) * np.sin(3 * lon) + 0.5 * np.sin(lat)
    uxda = ux.UxDataArray(xr.DataArray(values, dims="n_face", name="wave"), uxgrid=uxgrid)

    # without a projection, some lines run along the antimeridian
    lines = _contour_lines(uxda.plot.contour(levels=5, method="edges"))
    assert any(np.all(np.abs(x) == 180) for _, x, _ in lines)

    # rendering used to fail in GeoViews with an IndexError for those lines
    contours = uxda.plot.contour(levels=5, method="edges", projection=ccrs.Robinson(), backend="matplotlib")
    assert 0 < len(contours.data) < len(lines)
    hv.renderer("matplotlib").get_plot(contours)


def test_contour_missing_values(gridpath):
    """Faces without data are left out instead of raising."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    uxda = _face_latitude(uxgrid)
    uxda = uxda.where(uxgrid.face_lon.values < 0)

    for method in ["interpolated", "edges"]:
        lines = _contour_lines(uxda.plot.contour(levels=[10.0], method=method))
        assert len(lines) > 0
        assert all(x.max() < 5 for _, x, _ in lines)


def test_contour_invalid_input(gridpath, datasetpath):
    """Tests the errors for unsupported methods, dimensions and data locations."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))

    with pytest.raises(ValueError, match="Unsupported method"):
        uxds['v1'][0][0].plot.contour(method="smooth")

    # more than one dimension that is not of length 1
    with pytest.raises(ux.errors.DimensionError):
        uxds['v1'].plot.contour()

    # edge-centered data
    uxgrid = uxds.uxgrid
    edge_data = ux.UxDataArray(xr.DataArray(np.zeros(uxgrid.n_edge), dims="n_edge"), uxgrid=uxgrid)
    with pytest.raises(ux.errors.DataCenteringError):
        edge_data.plot.contour()
