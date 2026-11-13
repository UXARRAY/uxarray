# Verifying findings in UXarray

## Setup

Use a scratch directory outside the repo (`$SCRATCH` below), e.g. the agent's scratchpad.

```bash
# PR worktree (never switch branches in the user's checkout). --detach: the PR branch may already be
# checked out in another worktree, e.g. when reviewing your own or a colleague's branch.
git fetch origin
git worktree add --detach $SCRATCH/pr<N> origin/main
(cd $SCRATCH/pr<N> && gh pr checkout <N> --detach)

# Clean base worktree, for "does this also happen on main?" comparisons
git worktree add --detach $SCRATCH/base-wt origin/<base>
```

**Environment.** Reuse an existing conda env built from `ci/environment.yml` (`uxarray_build`) if there
is one; otherwise create it (several minutes):

```bash
conda env create -f ci/environment.yml          # env "uxarray_build": test deps, geo + viz extras
PY=$(conda run -n uxarray_build which python)
```

Don't install the worktree into the env. Select the tree under test with `PYTHONPATH` and check it took
effect: the env may hold a stale non-editable copy, or an editable install of the user's checkout, and
running from the repo root hides that.

```bash
cd $SCRATCH && PYTHONPATH=$SCRATCH/pr<N> $PY -c "import uxarray; print(uxarray.__file__)"   # must print pr<N>
```

`uxarray.__version__` comes from installed metadata, so it does not identify the tree; use `__file__`.

**Faster test runs** (optional). The env has no `pytest-xdist`. Overlay a venv on it and pin to the env's
versions, or uv upgrades numpy past what numba supports:

```bash
$PY -m pip list --format=freeze | grep -vi '^uxarray' > $SCRATCH/constraints.txt
uv venv -q --system-site-packages --python $PY $SCRATCH/ux-env
VIRTUAL_ENV=$SCRATCH/ux-env uv pip install -q -c $SCRATCH/constraints.txt pytest-xdist
export PROJ_DATA=$(dirname $(dirname $PY))/share/proj   # overlay venvs miss it: pyproj DataDirError
```

The full suite (~1250 tests, no network needed) takes about 3 min serially and under 1 min with `-n auto`.
Run the touched area first, the full suite at the end.

**Numba cache.** `@njit(cache=True)` writes its cache into `__pycache__` inside each tree, so a fresh
worktree compiles cold (first runs can be ~10x slower). Never time a first run.

## Techniques

Pick the ones that fit the candidate. The first two apply to almost every PR.

1. **Mutation test each source hunk.** Copy the PR tree (`rsync -a --exclude .git`), revert one hunk (or
   replace the new logic with the old behavior), and run the PR's tests with `PYTHONPATH` pointing at the
   copy. If the tests still pass, that hunk is untested: run an experiment to show what breaks without it,
   and report it as a coverage finding with that consequence.
   - "The new tests fail on main" is weak evidence when they fail only because a new symbol does not
     exist yet (ImportError). Mutate the behavior, not the existence.
2. **Sweep the inputs and options the change touches.** Write a short script that runs the changed path,
   on the PR and on base, over the axes the caller allows, and report the table:
   - meshes: an all-quad mesh *and* a mixed-size mesh with fill values (see the table below);
   - geometry: a face crossing the antimeridian, a face containing a pole, a vertex exactly on lon 0 or
     ±180, and the same face **rotated** by a few degrees (rotation must not change the topology of the
     result: this kind of sweep exposed the pole-detection bug in #1657);
   - data: node / edge / face-centered; extra dims (`time`, `lev`) before and after the grid dim;
     numpy and chunked;
   - options: `periodic_elements`, `destination`, `remap_to`, `backend`, `latlon`, `inverse_indices`.
   Build small custom faces inline: `ux.Grid.from_face_vertices(np.array([[[lon, lat], ...]]), latlon=True)`.
3. **Compare against an independent answer.** Analytic values (total area of a global mesh is 4π on the
   unit sphere: `ux.open_grid(<outCSne30.ug>).calculate_total_face_area()` matches to ~1e-11), an
   independent formula for a single face (remember UXarray edges are great-circle arcs), the accuracy
   baselines in `test/grid/geometry/`, or `Grid.validate()` for grid construction.
4. **Base vs PR equivalence.** For a refactor or performance change: run the same script with
   `PYTHONPATH` at each tree, save outputs, and compare with `np.testing.assert_allclose` or
   `xr.testing.assert_identical` (attrs and coords too; losing them is a past bug class). Compare grids
   with `Grid.__eq__` or the connectivity arrays directly.
5. **Grid and laziness invariants.**
   - `uxgrid` survives: `(v * 2).uxgrid is v.uxgrid`; type stays `UxDataArray`; after a subset or
     reorder, `result.uxgrid.n_face == result.sizes["n_face"]`.
   - Chunked input stays lazy — count graph executions:
     ```python
     import dask, dask.array
     class Counter(dask.callbacks.Callback):
         n = 0
         def _start(self, dsk): Counter.n += 1
     vc = v.chunk({"n_face": 1000})
     with Counter(): out = (vc - vc.mean()) * 2
     assert isinstance(out.data, dask.array.Array) and Counter.n == 0
     ```
     Some functions use `apply_ufunc` with the grid dim as a core dim (e.g. `gradient`), which rejects
     multiple chunks along `n_face`; for those, chunk a non-grid dim instead.
   - `import uxarray` stays cheap: `test/test_dependencies.py` and `test/utils/test_imports.py` cover
     kernel compilation and optional-dependency imports; also time `python -X importtime -c "import uxarray"`
     on base vs PR if imports changed.
6. **Timing.** For a quick A/B, a `timeit` script run under each tree's `PYTHONPATH`, after a warm-up call
   so numba compilation is excluded. Identical code differs by ~5% between runs: repeat and alternate
   the order before calling anything a regression. ASV for a single benchmark without building:
   ```bash
   cd $SCRATCH/pr<N>/benchmarks && asv machine --yes
   PYTHONPATH=$SCRATCH/pr<N> asv run --python=same --quick --bench "quad_hexagon.QuadHexagon.time_open_grid"
   ```
   Importing `benchmarks/helpers/_fixtures.py` downloads ~107 MB of meshes into `benchmarks/` (not
   gitignored); do it only in a scratch worktree. For full ASV results in CI, suggest the `run-benchmark`
   label.
7. **Debug kernels.** `NUMBA_DISABLE_JIT=1` runs jitted code as Python for real tracebacks and `pdb`. Don't
   time with it.
8. **Docs.** For docstring or `docs/` changes, a fast build without executing notebooks, run inside each
   worktree (the build writes into `docs/`), needs sphinx and the extensions from `ci/docs.yml`:
   ```bash
   cd $SCRATCH/pr<N>/docs && PYTHONPATH=$SCRATCH/pr<N> sphinx-build -b html -q \
       -D nb_execution_mode=off -D nbsphinx_execute=never . $SCRATCH/docs-pr 2> $SCRATCH/docs-pr.log
   ```
   Do the same on base and diff the sorted warning lists: `main` has some warnings already; only new ones
   are findings. Check new public names in `docs/api.rst` (nothing else checks it). Run a changed notebook
   with `jupyter nbconvert --execute --to notebook --output $SCRATCH/nb.ipynb <path>`, or point to the
   Read the Docs preview: `https://uxarray--<N>.org.readthedocs.build/en/<N>/`.
9. **Lint.** ruff excludes `test/` and `benchmarks/`; use the version pinned in `.pre-commit-config.yaml`:
   ```bash
   git -C $SCRATCH/pr<N> diff --name-only --diff-filter=AMR origin/<base>...HEAD -- 'uxarray/*.py' '*.ipynb' \
       | (cd $SCRATCH/pr<N> && xargs uvx ruff@<pinned> check) # then the same with `format --check`
   ```

### Test meshes (`test/meshfiles/`)

| Path | Use it for |
|---|---|
| `ugrid/quad-hexagon/grid.nc` (+ `data.nc`, `random-{node,edge,face}-data.nc`, `multi_dim_data.nc`) | Tiny (4 faces) mixed quad/hexagon grid with fill values; first choice for repros |
| `mpas/QU/mesh.QU.1920km.151026.nc` | Small global MPAS mesh (162 faces), fill values |
| `ugrid/outCSne30/outCSne30.ug` (+ `_vortex.nc`, `_var2.nc`) | Global cubed sphere (5400 quads), area = 4π |
| `ugrid/outRLL1deg/outRLL1deg.ug` | Lat-lon grid with pole rows |
| `scrip/outCSne8/`, `exodus/outCSne8/`, `exodus/mixed/`, `esmf/ne30/`, `geos-cs/c12/`, `icon/R02B04/` (largest, 15 MB), `fesom/` | Reader-specific checks |
| `scrip/scrip_radians/`, `scrip/oasis/` | SCRIP edge cases (radian units, OASIS masks) |
| `structured/`, `points/`, `shp/`, `geojson/`, `healpix/` | `from_structured`, `from_points`, geopandas readers, HEALPix data (grid via `ux.Grid.from_healpix(zoom=...)`) |

### What can't be verified locally

YAC remapping (built from source only in `yac-optional.yml`), the DYAMOND benchmarks (NCAR GLADE only),
and OS-specific behavior (Windows int dtypes, paths) unless you have that OS: label those PLAUSIBLE and
point to the CI job that covers them. In `gh pr checks`, the min-deps result shows in `report-failed`
(the `min-deps` job itself is `continue-on-error`).

Keep every script you write in `$SCRATCH`: they are the evidence, and the author can be pointed to them.

## Cleanup

```bash
git worktree remove --force $SCRATCH/pr<N>        # after the PR merges, if a re-review was possible
git worktree remove --force $SCRATCH/base-wt
```

`--force` is needed because the numba cache, docs build output and benchmark downloads leave untracked
files in the worktrees.
