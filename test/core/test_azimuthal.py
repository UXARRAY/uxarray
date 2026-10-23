import dask.array as da
import numpy as np
import pytest
import xarray as xr

import uxarray as ux
from uxarray.core.azimuthal import _ring_membership


def test_gaussian(gridpath, datasetpath):
    uxds = ux.open_dataset(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
    )

    res = uxds['gaussian'].azimuthal_mean(center_coord=(45, 0), outer_radius=2, radius_step=0.5)

    # Expects decreasing values from center
    valid_vals = res[1:]

    np.testing.assert_array_less(
        valid_vals.diff("radius").values, 1e-12
    )



def test_inverse_gaussian(gridpath, datasetpath):

    uxds =  ux.open_dataset(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
    )

    res = uxds['inverse_gaussian'].azimuthal_mean(center_coord=(45, 0), outer_radius=2, radius_step=0.5)

    # Expects increasing values from center
    atol = 1e-12
    diffs = res[1:].diff("radius").values
    diffs = diffs[np.isfinite(diffs)]
    np.testing.assert_array_less(-atol, diffs)

def test_non_zero_hit_counts(gridpath, datasetpath):
    uxds =  ux.open_dataset(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
    )

    res, hit_counts = uxds['inverse_gaussian'].azimuthal_mean(center_coord=(45, 0), outer_radius=2, radius_step=0.5, return_hit_counts=True)

    # At least one hit after the first circle
    assert np.all(hit_counts[1:] > 1)

    assert 'radius' in hit_counts.dims
    assert hit_counts.sizes['radius'] == res.sizes['radius']

def test_zero_hit_counts(gridpath, datasetpath):
    uxds =  ux.open_dataset(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
    )

    # Outside of grid domain
    res, hit_counts = uxds['inverse_gaussian'].azimuthal_mean(center_coord=(-45, 0), outer_radius=2, radius_step=0.5, return_hit_counts=True)

    assert 'radius' in hit_counts.dims
    assert hit_counts.sizes['radius'] == res.sizes['radius']

    # No hits
    assert np.all(hit_counts == 0)

    print(res)


def _reference_azimuthal_mean(uxda, center_coord, outer_radius, radius_step):
    """Per-ring float64 oracle: nansum(x * area) / sum(area) over each ring."""
    from uxarray.grid.coordinates import _lonlat_rad_to_xyz

    kdtree = uxda.uxgrid._get_scipy_kd_tree()
    lon, lat = np.deg2rad(center_coord)
    center = np.array(_lonlat_rad_to_xyz(lon, lat))
    radii = np.arange(0.0, outer_radius + radius_step, radius_step)
    chords = 2.0 * np.sin(np.deg2rad(radii) / 2.0)

    areas = uxda.uxgrid.face_areas.values.astype(np.float64)
    values = uxda.transpose(..., "n_face").values.astype(np.float64)
    out = np.full(values.shape[:-1] + (radii.size,), np.nan)
    previous = set()
    for i, chord in enumerate(chords):
        within = set(kdtree.query_ball_point(center, chord))
        ring = np.array(sorted(within - previous), dtype=int)
        previous = within
        if ring.size:
            out[..., i] = (
                np.nansum(values[..., ring] * areas[ring], axis=-1) / areas[ring].sum()
            )
    return out


def _multi_dim(uxds):
    """(n_face, time) data with the grid dim first and a few NaNs."""
    stacked = (
        xr.concat(
            [uxds["gaussian"].to_xarray(), uxds["inverse_gaussian"].to_xarray()],
            dim="time",
        )
        .transpose("n_face", "time")
        .copy()
    )
    stacked[[3, 50, 120], 1] = np.nan
    return ux.UxDataArray(stacked.assign_coords(time=[10, 20]), uxgrid=uxds.uxgrid)


def test_matches_reference(gridpath, datasetpath):
    uxds = ux.open_dataset(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc"),
    )
    uxda = _multi_dim(uxds)

    res = uxda.azimuthal_mean(center_coord=(45, 0), outer_radius=2, radius_step=0.25)

    assert res.dims == ("radius", "time")
    np.testing.assert_array_equal(res["time"], [10, 20])
    expected = _reference_azimuthal_mean(uxda, (45, 0), 2, 0.25)
    np.testing.assert_allclose(
        res.transpose("time", "radius").values, expected, rtol=1e-12
    )


def test_dask_lazy_and_matches_numpy(gridpath, datasetpath):
    uxds = ux.open_dataset(
        gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc"),
    )
    uxda = _multi_dim(uxds)
    kwargs = dict(
        center_coord=(45, 0), outer_radius=2, radius_step=0.25, return_hit_counts=True
    )

    eager, eager_hits = uxda.azimuthal_mean(**kwargs)
    lazy, lazy_hits = uxda.chunk(n_face=50, time=1).azimuthal_mean(**kwargs)

    assert isinstance(lazy.data, da.Array)
    assert lazy.chunks[lazy.get_axis_num("time")] == (1, 1)
    np.testing.assert_array_equal(lazy_hits, eager_hits)
    np.testing.assert_allclose(lazy.values, eager.values, rtol=1e-14, equal_nan=True)


def _per_radius_rings(kdtree, center_xyz, chord_radii):
    """The original ring assignment: one kd-tree query per radius."""
    faces, ring_ids = [], []
    processed = np.array([], dtype=np.int_)
    for i, r_chord in enumerate(chord_radii):
        within = np.sort(
            np.array(kdtree.query_ball_point(center_xyz, r_chord), dtype=np.int_)
        )
        ring = np.setdiff1d(within, processed, assume_unique=True)
        if ring.size:
            processed = within
        faces.append(ring)
        ring_ids.append(np.full(ring.size, i))
    return np.concatenate(faces), np.concatenate(ring_ids)


@pytest.mark.parametrize(
    "outer_radius, radius_step", [(2, 0.1), (60, 1), (250, 10), (360, 3)]
)
def test_ring_membership_matches_per_radius_queries(
    gridpath, outer_radius, radius_step
):
    from uxarray.grid.coordinates import _lonlat_rad_to_xyz

    uxgrid = ux.open_grid(gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"))
    kdtree = uxgrid._get_scipy_kd_tree()
    radii = np.arange(0.0, outer_radius + radius_step, radius_step)
    chord_radii = 2.0 * np.sin(np.deg2rad(radii) / 2.0)

    for lon, lat in [(45, 0), (44.7, 0.6), (-45, 0)]:
        center = np.array(_lonlat_rad_to_xyz(np.deg2rad(lon), np.deg2rad(lat)))
        expected = _per_radius_rings(kdtree, center, chord_radii)
        actual = _ring_membership(kdtree, center, chord_radii)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


def test_ring_membership_ties_on_ring_edges(gridpath):
    """Radii exactly at face distances, where the kd-tree's pruning rounding decides."""
    uxgrid = ux.open_grid(gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"))
    kdtree = uxgrid._get_scipy_kd_tree()
    rng = np.random.default_rng(0)

    for _ in range(200):
        center = kdtree.data[rng.integers(uxgrid.n_face)].copy()
        offsets = kdtree.data[rng.integers(uxgrid.n_face, size=5)] - center
        chord_radii = np.sort(np.append(np.sqrt((offsets * offsets).sum(axis=1)), 0.0))
        expected = _per_radius_rings(kdtree, center, chord_radii)
        actual = _ring_membership(kdtree, center, chord_radii)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
