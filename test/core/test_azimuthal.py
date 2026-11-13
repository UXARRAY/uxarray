import dask.array as da
import numpy as np
import pytest
import xarray as xr
from xarray.core import indexing

import uxarray as ux
from uxarray.core.azimuthal import _ring_membership
from uxarray.grid.coordinates import _lonlat_rad_to_xyz


@pytest.fixture
def uxds(gridpath, datasetpath):
    return ux.open_dataset(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc"),
    )


def _center_xyz(lon, lat):
    return np.array(_lonlat_rad_to_xyz(np.deg2rad(lon), np.deg2rad(lat)))


def _chords(outer_radius, radius_step):
    radii = np.arange(0.0, outer_radius + radius_step, radius_step)
    return 2.0 * np.sin(np.deg2rad(radii) / 2.0)


def _reference_rings(points, center_xyz, chord_radii):
    """Brute-force ring of each point: the first i with d <= r_i, or -1."""
    dist2 = ((points - center_xyz) ** 2).sum(axis=1)
    ring_of = np.full(len(points), -1)
    # squared as an array, as a numpy scalar's ** can round differently
    for i, bound in reversed(list(enumerate(chord_radii**2))):
        ring_of[dist2 <= bound] = i
    return ring_of


@pytest.mark.parametrize("name, sign", [("gaussian", -1), ("inverse_gaussian", 1)])
def test_monotone_profile(uxds, name, sign):
    res = uxds[name].azimuthal_mean(
        center_coord=(45, 0), outer_radius=2, radius_step=0.5
    )

    # gaussian decreases away from the center, inverse_gaussian increases. Only
    # the empty ring at radius 0 is NaN, so any other NaN fails the comparison.
    np.testing.assert_array_less(-1e-12, sign * res[1:].diff("radius").values)


@pytest.mark.parametrize("chunks", [None, {"n_face": 50, "time": 1}])
@pytest.mark.parametrize("center", [(45, 0), (-45, 0)])  # in and outside the subset
def test_matches_reference(uxds, center, chunks):
    # (n_face, time) data with the grid dim first and a few NaNs
    stacked = (
        xr.concat([uxds["gaussian"], uxds["inverse_gaussian"]], dim="time")
        .transpose("n_face", "time")
        .copy()
    )
    stacked[[3, 50, 120], 1] = np.nan
    uxda = ux.UxDataArray(stacked.assign_coords(time=[10, 20]), uxgrid=uxds.uxgrid)
    if chunks:
        uxda = uxda.chunk(chunks)

    res, hits = uxda.azimuthal_mean(
        center_coord=center, outer_radius=2, radius_step=0.25, return_hit_counts=True
    )

    assert res.dims == ("radius", "time")
    assert hits.dims == ("radius",) and hits.sizes["radius"] == res.sizes["radius"]
    np.testing.assert_array_equal(res["time"], [10, 20])
    if chunks:
        assert isinstance(res.data, da.Array)
        assert res.chunks[1] == (1, 1)

    # Per-ring float64 oracle: nansum(x * area) / sum(area)
    ring_of = _reference_rings(
        uxda.uxgrid._get_scipy_kd_tree().data, _center_xyz(*center), _chords(2, 0.25)
    )
    areas = uxda.uxgrid.face_areas.values.astype(np.float64)
    values = uxda.values.astype(np.float64)
    expected = np.full(res.shape, np.nan)
    expected_hits = np.zeros(res.sizes["radius"], dtype=int)
    for i in range(res.sizes["radius"]):
        ring = ring_of == i
        expected_hits[i] = ring.sum()
        if ring.any():
            weighted = np.nansum(values[ring] * areas[ring, None], axis=0)
            expected[i] = weighted / areas[ring].sum()

    np.testing.assert_array_equal(hits, expected_hits)
    np.testing.assert_allclose(res.values, expected, rtol=1e-12, equal_nan=True)
    if center == (45, 0):
        assert np.all(hits[1:] > 1)
    else:
        assert np.all(hits == 0)


class _CountingBackendArray(xr.backends.BackendArray):
    """Serves an in-memory array the way a file engine (e.g. netCDF4) does,
    recording the shape of each read."""

    def __init__(self, array):
        self.array, self.shape, self.dtype = array, array.shape, array.dtype
        self.read_shapes = []

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._read
        )

    def _read(self, key):
        # numpy matches outer indexing here, as at most one axis is fancy indexed
        array = self.array[key]
        self.read_shapes.append(array.shape)
        return array


# (center, outer_radius): 12 faces scattered over a span of 133, every face,
# and none
@pytest.mark.parametrize(
    "center, outer_radius", [((45, 0), 0.5), ((45, 0), 2), ((-45, 0), 2)]
)
def test_file_backed_reads_are_bounded(uxds, center, outer_radius):
    data = ux.UxDataArray(
        xr.concat([uxds["gaussian"], uxds["inverse_gaussian"]] * 2, dim="time"),
        uxgrid=uxds.uxgrid,
    )
    # (time, n_face) data loaded lazily without dask, as a file engine serves it
    backend = _CountingBackendArray(data.values)
    lazy = ux.UxDataArray(
        xr.Variable(data.dims, indexing.LazilyIndexedArray(backend)),
        uxgrid=uxds.uxgrid,
    )

    res, hits = lazy.azimuthal_mean(
        center_coord=center,
        outer_radius=outer_radius,
        radius_step=0.25,
        return_hit_counts=True,
    )

    # every time step is read for each face, but at most twice the selected faces
    n_selected = int(hits.sum())
    assert all(shape[0] == data.sizes["time"] for shape in backend.read_shapes)
    assert sum(shape[1] for shape in backend.read_shapes) <= 2 * n_selected
    if center == (45, 0) and outer_radius == 0.5:
        assert n_selected == 12  # spread over 133 faces of the array

    expected = data.azimuthal_mean(
        center_coord=center, outer_radius=outer_radius, radius_step=0.25
    )
    np.testing.assert_array_equal(res.values, expected.values)


def _ring_edge_cases(kdtree, n_cases=200):
    """Radii exactly at face distances, so faces sit on ring edges, where the
    kd-tree's own rounding can disagree with d <= r."""
    rng = np.random.default_rng(0)
    for _ in range(n_cases):
        center = kdtree.data[rng.integers(kdtree.n)] + rng.normal(scale=1e-3, size=3)
        offsets = kdtree.data[rng.integers(kdtree.n, size=5)] - center
        yield center, np.sort(np.append(np.sqrt((offsets**2).sum(axis=1)), 0.0))


# (outer_radius, radius_step), past 180 degrees too, or None for ring-edge ties
@pytest.mark.parametrize("radii", [(2, 0.1), (60, 1), (250, 10), (360, 3), None])
def test_ring_membership(gridpath, radii):
    kdtree = ux.open_grid(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc")
    )._get_scipy_kd_tree()
    if radii is None:
        cases = _ring_edge_cases(kdtree)
    else:
        chords = _chords(*radii)
        cases = [(_center_xyz(*c), chords) for c in [(45, 0), (44.7, 0.6), (-45, 0)]]

    for center, chords in cases:
        ring_of = _reference_rings(kdtree.data, center, chords)
        faces, ring_ids = _ring_membership(kdtree, center, chords)
        np.testing.assert_array_equal(faces, np.flatnonzero(ring_of >= 0))
        np.testing.assert_array_equal(ring_ids, ring_of[faces])
