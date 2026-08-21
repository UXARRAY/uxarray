import uxarray as ux
import xarray as xr
import pytest

# import pooch  # not necessary to actually import, but tests would likely fail if this fails.
# (commented here for future reference, since pooch is included in uxarray[dev]
#    dependencies, but not imported explicitly anywhere in uxarray.)

@pytest.mark.parametrize("ds_name", ["air_temperature", "ersstv5"])
def test_read_structured_grid_from_ds(ds_name):
    ds = xr.tutorial.open_dataset(ds_name)
    uxgrid = ux.Grid.from_structured(ds)

    assert uxgrid.n_face == ds.sizes['lon'] * ds.sizes['lat']

    assert uxgrid.validate()


@pytest.mark.parametrize("ds_name", ["air_temperature", "ersstv5"])
def test_read_structured_grid_from_latlon(ds_name):
    ds = xr.tutorial.open_dataset(ds_name)
    uxgrid = ux.Grid.from_structured(lon=ds.lon, lat=ds.lat)
    assert uxgrid.n_face == ds.sizes['lon'] * ds.sizes['lat']
    assert uxgrid.validate()

@pytest.mark.parametrize("ds_name", ["air_temperature", "ersstv5"])
def test_read_structured_uxds_from_ds(ds_name):
    # Load the dataset using xarray's tutorial module
    ds = xr.tutorial.open_dataset(ds_name)

    # Create a uxarray Grid from the structured dataset
    uxds = ux.UxDataset.from_structured(ds)

    assert "n_face" in uxds.dims

    assert "lon" not in uxds.dims
    assert "lat" not in uxds.dims

    assert uxds.uxgrid.validate()


@pytest.mark.parametrize("ds_name", ["air_temperature"])
def test_from_xarray_with_grid_from_latlon(ds_name):
    """Regression test for GH #1410: a Grid built via ``from_structured(lon=, lat=)``
    must record its source dimensions so ``UxDataset.from_xarray`` flattens the
    structured (lon, lat) data variables onto ``n_face``."""
    ds = xr.tutorial.open_dataset(ds_name)

    uxgrid = ux.Grid.from_structured(lon=ds.lon, lat=ds.lat)

    # The grid must carry the structured spec and the source-dim mapping.
    assert uxgrid.source_grid_spec == "Structured"
    assert uxgrid._source_dims_dict == {"n_face": (ds.lon.dims[0], ds.lat.dims[0])}

    uxds = ux.UxDataset.from_xarray(ds, uxgrid=uxgrid)

    # Data must now be mapped onto n_face, not left on lon/lat.
    assert "n_face" in uxds.dims
    assert "lon" not in uxds.dims
    assert "lat" not in uxds.dims
    assert uxds["air"].sizes["n_face"] == ds.sizes["lon"] * ds.sizes["lat"]

    # The flatten must preserve data order: each face value must equal the
    # original (lat, lon) cell value at that face. n_face is stacked as
    # (lat, lon) C-order, so face k corresponds to (k // n_lon, k % n_lon).
    n_lon = ds.sizes["lon"]
    original = ds["air"].isel(time=0).values  # (lat, lon)
    flattened = uxds["air"].isel(time=0).values  # (n_face,)
    for k in (0, n_lon + 1, flattened.size - 1):  # first, an interior, last
        i, j = k // n_lon, k % n_lon
        assert flattened[k] == original[i, j]

    # End-to-end: the mapped data must be subsettable (the symptom in #1410).
    subset = uxds["air"].isel(time=0).subset.bounding_circle((-100.0, 40.0), 5)
    assert "n_face" in subset.dims
    assert subset.sizes["n_face"] > 0


def test_global_structured_grid_merges_poles_and_seam():
    """Nodes coincident on the sphere must be merged, even though their
    (lon, lat) pairs differ. Regression test for issue #1689."""
    import numpy as np

    n_lon, n_lat = 36, 18
    d_lat = 180.0 / n_lat
    lon = np.linspace(-180, 180, n_lon, endpoint=False)
    lat = np.linspace(-90 + d_lat / 2, 90 - d_lat / 2, n_lat)

    uxgrid = ux.Grid.from_structured(lon=lon, lat=lat)

    # Every duplicated pole node and antimeridian node must be gone.
    assert uxgrid.n_node < (n_lon + 1) * (n_lat + 1)
    assert np.isclose(uxgrid.node_lat.values, 90.0).sum() == 1
    assert np.isclose(uxgrid.node_lat.values, -90.0).sum() == 1

    # A closed sphere: V - E + F == 2.
    assert uxgrid.n_node - uxgrid.n_edge + uxgrid.n_face == 2

    # The pole is now a real singularity touching every longitude column, and
    # its faces are triangles rather than quads with a repeated corner.
    face_nodes = uxgrid.face_node_connectivity.values
    n_nodes_per_face = uxgrid.n_nodes_per_face.values
    assert (n_nodes_per_face == 3).sum() == 2 * n_lon

    for face, n_nodes in zip(face_nodes, n_nodes_per_face):
        nodes = face.tolist()[:n_nodes]
        assert len(set(nodes)) == n_nodes

    pole = int(np.flatnonzero(np.isclose(uxgrid.node_lat.values, 90.0))[0])
    assert (face_nodes == pole).any(axis=1).sum() == n_lon


def test_regional_structured_grid_is_unchanged():
    """A grid that touches neither pole nor the antimeridian must keep every
    node and stay entirely quadrilateral."""
    import numpy as np

    lon = np.linspace(-50, -10, 20)
    lat = np.linspace(10, 40, 15)

    uxgrid = ux.Grid.from_structured(lon=lon, lat=lat)

    assert uxgrid.n_node == 21 * 16
    assert uxgrid.n_face == 20 * 15
    assert (uxgrid.n_nodes_per_face.values == 4).all()
