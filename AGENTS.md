# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Copilot, Cursor, etc.) working in this repository. The
repository is the source of truth: if anything here disagrees with the code, follow the code and point
out the stale line.

## About

UXarray is an xarray extension for analysis and visualization of **unstructured** grids, mostly from
climate and weather models. It is pure Python; hot loops are JIT-compiled with numba.

- `UxDataset` / `UxDataArray` subclass `xr.Dataset` / `xr.DataArray` and carry a `Grid` (`.uxgrid`).
- `Grid` (`uxarray/grid/grid.py`) is **not** an xarray subclass: it wraps a plain `xr.Dataset` of
  UGRID-named variables in `Grid._ds`. Maintainers don't want `Grid` to grow Dataset-like API (#1647).
- Every supported format (UGRID, MPAS, ICON, SCRIP, ESMF, Exodus, FESOM2, GEOS-CS, HEALPix, shapefile /
  GeoJSON, points via Delaunay/Voronoi, ...) is converted to one internal UGRID-style representation. The
  minimum is `node_lon`, `node_lat`, `face_node_connectivity`.
- Scope: horizontally 2D grids on the sphere (vertical levels are ordinary dims), convex faces (#1597).
  Structured grids enter only through explicit APIs such as `Grid.from_structured()`, never through
  `open_dataset()` / `open_grid()` (#1887).
- Mirror xarray's API, names and behavior wherever there is a natural analogue (#1461, #1573).

## Commands

```bash
# Dev environment (conda env "uxarray_build", as CI uses), then install the working tree
conda env create -f ci/environment.yml
conda activate uxarray_build
pip install -e . --no-deps
# or, without conda: pip install -e ".[all]"     (extras: dev, geo, viz, all/complete)

python -m pytest test                                  # full suite (testpaths = ["test"])
python -m pytest test/grid/geometry                    # one area
python -m pytest test/core/test_dataset.py::test_x -x  # one test
pre-commit run --all-files                             # ruff lint + format, yaml, whitespace, ...
```

- `test_optional_deps/` is deliberately outside `testpaths`; CI runs it in separate installs with no
  extras, `[geo]`, `[viz]` and `[geo,viz]` (`ci.yml`, job `test-optional-deps`).
- Docs build in their own env: `conda env create -f ci/docs.yml` (env `uxarray-docs`), then
  `cd docs && make html`. Notebooks execute during the build, so it is slow and needs network.
- No pytest markers are registered; select tests by path.

## Repo map

| Path | Contents |
|---|---|
| `uxarray/__init__.py` | The public top-level API. Not listed here → not top-level public. |
| `uxarray/core/` | `UxDataArray`, `UxDataset`, `api.py` (`open_grid`, `open_dataset`, `open_mfdataset`, ...), `accessors.py` (groupby/resample/... wrappers), aggregation, gradient, zonal |
| `uxarray/grid/` | `Grid` and the topology/geometry numerics: connectivity, coordinates, geometry, arcs, intersections, area, bounds, neighbors (trees, spatial hash), dual, `slice.py`, validation, point_in_face, integrate |
| `uxarray/conventions/` | **Naming authority** for UGRID dims/coords/connectivity (`ugrid.py`), descriptors, and `attrs.py` |
| `uxarray/io/` | One reader per format (`_<fmt>.py`) + `utils.py` (format detection) |
| `uxarray/remap/`, `subset/`, `cross_sections/`, `plot/` | Accessor-based features (`.remap`, `.subset`, `.cross_section`, `.plot`) |
| `uxarray/utils/` | `computing.py` (compensated arithmetic), `numba_math.py`, `parallel.py` (`parallel_njit`), `imports.py` (optional deps), `warnings.py` |
| `test/` | pytest suite; data in `test/meshfiles/<format>/<dataset>/`; ruff-excluded |
| `benchmarks/` | ASV suite (`helpers/` holds fixtures, warm-up, peakmem helpers); ruff-excluded |
| `docs/` | Sphinx + myst-nb; `api.rst` is the hand-maintained API index |

## Core data model

- A `UxDataset` and the `UxDataArray`s taken from it share **one `Grid` by reference**; only a deep copy
  clones it.
- `self.uxgrid` raises `GridInvalidError` when no grid is attached; use it in anything grid-aware.
  `self._uxgrid` may be `None` (xarray's `_replace`/`_construct_direct` rebuild objects without
  kwargs) and is for copy/replace/init plumbing only. `UxDataArray(...)` without `uxgrid=` succeeds and
  fails later at the first grid-aware call.
- Grid preservation through xarray operations is explicit: `__getattribute__` in `dataarray.py` /
  `dataset.py` wraps `groupby`, `groupby_bins`, `resample`, `rolling`, `coarsen`, `weighted`,
  `rolling_exp`, `cumulative` with the classes in `core/accessors.py`, and `where`/`fillna` are
  overridden. A new xarray method that returns an object needs the same treatment, or results lose the
  grid.
- Combining objects on **different grids** raises `GridsMismatchError`; never silently keep one grid.
- Results that no longer have a grid dimension: `zonal_mean` returns plain xarray, but the general rule
  is still undecided (#1733). Ask instead of picking one.
- `UxDataArray.isel(..., ignore_grid=False, inverse_indices=False)` slices the grid along with the data
  through `Grid.isel` (`grid/slice.py`). `Grid.inverse_indices` maps a subset back to its source grid.
- Operations that change topology return a **new** `Grid`; adding derived variables to an existing grid
  is fine.
- File writing goes through xarray (`uxds.to_xarray().to_netcdf()`, `uxgrid.to_xarray()`); don't add
  write APIs such as `Grid.to_netcdf()` (#1836).

## Grid variables are declared in `uxarray/conventions`

`conventions/ugrid.py` declares every UGRID dim, coordinate and connectivity name with its `dims` tuple
and `attrs`; `conventions/descriptors.py` does the same for descriptors. The name lists there
(`DIM_NAMES`, `SPHERICAL_COORD_NAMES`, `CARTESIAN_COORD_NAMES`, `CONNECTIVITY_NAMES`,
`DESCRIPTOR_NAMES`) drive `Grid.dims`/`.coordinates`/`.connectivity`/`.descriptors`, `repr(grid)` and
the HTML repr: a variable that isn't registered works but stays invisible. Use these constants
rather than literal attrs dicts.

There is no `cached_property`: presence in `Grid._ds` is the cache.

```python
@property
def edge_node_connectivity(self) -> xr.DataArray:
    if "edge_node_connectivity" not in self._ds:
        _populate_edge_node_connectivity(self)
    return self._ds["edge_node_connectivity"]
```

followed by `x = x.setter(make_setter("x"))`. To add a derived variable: declare `_DIMS`/`_ATTRS` and
register the name in `conventions/`; write `_populate_<name>(grid)` in the matching `grid/` module
(kernel in a separate jitted function); add the guarded property + setter in `grid.py`; list it in
`docs/api.rst`; test that it populates on demand with the right dims. Some variables are built in
pairs: computing `edge_node_connectivity` also derives `face_edge_connectivity` in the same edge order,
so code that renumbers edges must keep both consistent (see the `Grid.edge_node_connectivity`
docstring).

Non-UGRID derived state (spatial trees, antimeridian face indices, plotting caches) is cached in private
attributes set in `Grid.__init__`; reuse `Grid.get_ball_tree()`, `get_kd_tree()` and
`get_spatial_hash()` rather than building new trees.

## Numeric conventions (`uxarray/constants.py`)

- Index arrays use `INT_DTYPE` (`np.intp`); ragged connectivity is padded with `INT_FILL_VALUE`
  (`np.iinfo(np.intp).min`), never `np.nan`, `-1` or `0`. Only these two constants are top-level public.
- Tolerances: `ERROR_TOLERANCE` (1e-8) for geometric "is this degenerate/normalized/at the pole" checks;
  `MACHINE_EPSILON` only for near-machine-precision arc/intersection math. They are not interchangeable.
- Geometry is computed on the **unit sphere**; the physical radius is `Grid.sphere_radius`.
- Spherical coordinates are **degrees** with longitude in **[-180, 180]**; internal radian helpers use
  **[0, 2π)**. Mixing the two ranges is a recurring bug.
- Projected grids (coordinates in meters) skip longitude wrapping with a one-time warning; spherical
  geometry does not apply to them (`grid/coordinates.py::_set_desired_longitude_range`).
- Antimeridian-crossing faces: `Grid.antimeridian_face_indices`; plotting and `to_geodataframe` take
  `periodic_elements="exclude" | "split" | "ignore"`.
- Accuracy-sensitive geometry (great-circle arcs, intersections, areas) uses the compensated / error-free
  primitives in `uxarray/utils/computing.py` (`two_sum`, `diff_of_products`, `accucross`, ...), checked
  by `test/grid/geometry/test_eft_accuracy.py` and `test_accusphgeom_baseline.py`. Don't replace them
  with naive arithmetic, don't use `fastmath=True` (it breaks compensation), and don't claim
  "robust"/"exact" in comments unless a test shows it.

## numba and dask

- Kernels use `@njit(cache=True)`. Parallel kernels use `@parallel_njit` (`utils/parallel.py`), not raw
  `njit(parallel=True)`: it runs the kernel under `numba_pool()`, which keeps numba's thread pool safe
  when called from dask threads. It is only for kernels called from Python, not from other jitted code.
- `import uxarray` must not compile any kernel or start numba's threading layer
  (`test/test_dependencies.py::test_no_numba_kernels_built_on_import`), so no module-level work that
  triggers compilation.
- Inside kernels, don't allocate small numpy arrays in loops; use scalars or tuples and the 3-vector
  helpers in `utils/numba_math.py` (#1648).
- Keep chunked (dask) data lazy: no `.values` / `np.asarray` on whole arrays, no silent rechunking.
  Prefer `xr.apply_ufunc` over numba kernels and go through xarray (`.compute()`), not dask directly.
  Some global-topology and spatial-index operations must materialize; say so when they do.
- Setting `NUMBA_DISABLE_JIT=1` steps through kernels in a debugger; there is no project-level toggle.

## Optional dependencies

Hard dependencies are listed in `pyproject.toml` `dependencies`. Plotting and geo packages (cartopy,
geopandas, healpix, pyproj, spatialpandas, datashader, matplotlib, geoviews, holoviews, hvplot) are the
optional extras `[geo]` / `[viz]` (#1548). Rules, enforced by `test/utils/test_imports.py`:

- Import optional packages **inside functions only**, never at module level (`if TYPE_CHECKING:` is
  exempt).
- Before importing them, call `_raise_hint_if_optional_deps_missing("pkg", ...)`
  (`uxarray/utils/imports.py`), naming every optional package the function imports. It raises
  `OptionalDependencyNotFoundError` with a `pip install "uxarray[viz]"`-style hint. A new optional
  dependency must also be added to `_OPTIONAL_DEPS_TO_EXTRAS`.
- YAC is a remap **backend** (`backend="yac"`), not in any extra; it is imported lazily and raises
  `YacNotAvailableError`, and its tests run only in `.github/workflows/yac-optional.yml`.
- Lower bounds on dependencies are added only for a known failure, with a comment saying what breaks.
  Min-deps CI (`min-deps-ci.yml`) tests the oldest allowed versions on every PR, so using a newer API of
  a dependency may require raising its bound.

## Errors and warnings

- Domain errors use the types in `uxarray/errors.py`: `DataCenteringError`, `DimensionError`,
  `GridInvalidError`, `GridsMismatchError` (all `ValueError` subclasses), `YacNotAvailableError`,
  `OptionalDependencyNotFoundError`. New custom types go there. Plain `ValueError` / `TypeError` /
  `FileNotFoundError` stay correct for generic argument checks.
- Messages state expected **and** actual (e.g. "expected node-centered data, got edge-centered") and
  tell the user what to do; they don't mention private names. Don't validate with `assert`, and don't
  wrap calls in broad `try/except` that hides the original error. Invalid input raises; it never
  silently produces a different result.
- Docstrings name custom errors as e.g. "DimensionError (subclass of ValueError)".
- New warnings use `stacklevel=find_stack_level()` (`utils/warnings.py`). Deprecations use
  `FutureWarning`, and deprecated code is removed after about a year (#1707).

## Adding features

- **Grid format**: detection (`_is_<fmt>` + the if/elif chain in `io/utils.py::_parse_grid_type`) and
  dispatch (`Grid.from_dataset`) are two ordered chains, not a registry; first match wins. The reader
  `io/_<fmt>.py::_read_<fmt>` returns `(grid_ds, source_dims_dict)`. Add a small mesh under
  `test/meshfiles/<fmt>/`, tests under `test/io/`, and the format in `docs/user-guide/grid-formats.rst`.
- **Remap / subset / plot**: add the method on the accessor (`remap/accessor.py`,
  `subset/grid_accessor.py` + `subset/dataarray_accessor.py`, `plot/accessor.py`), keep its `__repr__`
  method list current, and list it in `docs/api.rst` (accessor methods use
  `:template: autosummary/accessor_method.rst`). Plotting calls `plotting_backend.assign(backend)` and
  imports hvplot lazily through `_ensure_hvplot_imported()`; keep that, as `import uxarray` time is
  watched.

## Tests

- pytest only (no `unittest.TestCase`). Mostly module-level `def test_*`; plain `class Test...` groups
  exist for variants. Layout loosely mirrors the package (`test/core/`, `test/grid/...`, `test/io/`).
- Get data paths through the `test/conftest.py` fixtures (`gridpath`, `datasetpath`, `test_data_dir`,
  `mesh_constants`): `gridpath("ugrid", "quad-hexagon", "grid.nc")`. Passing a **list** of filenames
  makes a missing file skip the test instead of failing.
- A bug fix needs a regression test that **fails on `main`** and passes with the fix; reviewers check
  this. Tests must assert what their name says, not just call the function.
- Prefer existing meshfiles or a tiny hand-built mesh over new large files; compare floats with
  `np.testing.assert_allclose` / `np.isclose` and justify tolerances and expected values (analytic
  values, a reference implementation) rather than magic numbers.
- Don't assume dimension order or 1-D data: cover extra dims (e.g. `time`) and use `.sizes` / `.isel`.
- Optional-dependency imports in tests: `try/except ImportError` + `pytest.skip`, or a `skipif` reusing
  the source's own check. Tests must be fast; use `monkeypatch` rather than mutating global state.

## Benchmarks

ASV under `benchmarks/` (`asv.conf.json`: Python 3.12, env from `ci/environment.yml`). PR benchmarks run
when the `run-benchmark` label is added, in two workflows split for security (#1547): don't merge them.
Put fixture building and numba warm-up in `setup()`, not in timed methods; use `number = 1` when repeat
calls would hit a cache; no assertions. Reviewers expect ASV results (and an accuracy check) on
performance PRs, and a new benchmark when a hot path has none. Peakmem results are noisy.

## Lint and formatting

ruff (lint, import sort, format) via pre-commit, also on notebooks. `test/` and `benchmarks/` are
excluded from ruff (`extend-exclude`), so don't reformat them in a functional PR. pre-commit.ci reports
on PRs but doesn't push fixes; comment `pre-commit.ci autofix` to get them.

## Docs

- Docstrings are **numpydoc** (`sphinx.ext.napoleon`). `autodoc_typehints = "none"`, so types reach the
  docs only through the `Parameters` section. Examples are not doctested. Private helpers may have
  short docstrings.
- New public API is listed by hand in `docs/api.rst`; internal names start with `_`.
- A new user-facing feature also gets a user-guide section or notebook; reviewers ask for this before
  merge. User-guide notebooks go in `docs/user-guide/` and both lists in `docs/userguide.rst`; gallery
  examples go in `docs/examples/` plus `docs/gallery.yml`, `docs/gallery.rst` and a thumbnail.
- Clear notebook outputs before committing, except notebooks in `nb_execution_excludepatterns`
  (`docs/conf.py`), whose stored outputs are what the docs render. Notebooks run on Read the Docs, so
  they can't use data that exists only on NCAR's GLADE.
- Docs are concise reference for people who know their models, not a tutorial on unstructured grids.
- Vocabulary: node, edge, face, connectivity, fill value (`docs/user-guide/terminology.rst`).
- `docs/contributing.rst` is out of date (being rewritten, #1831): it prescribes Google-style
  docstrings, `unittest` and `docs/user_api/index.rst`. The code uses numpydoc, pytest and
  `docs/api.rst`; follow the code.

## Pull requests

- Follow `.github/PULL_REQUEST_TEMPLATE.md`. The title is a descriptive summary. Use "Closes #N" only if
  the PR fully resolves the issue, otherwise "Related to #N". Every PR has a linked issue.
- **AI disclosure**: fill in the "AI Usage:" line (tools, optionally model and how they were used). The
  responsibility checkbox is the human author's to tick; leave it unticked in drafts you write.
- Open as draft until ready. Don't resolve review threads you didn't start; reviewers resolve their own.
  Two approvals is the norm; the author usually merges.
- Keep PRs focused: split large multi-bug PRs and open issues for follow-ups. Behavior changes that
  aren't the point of the PR (ordering, defaults, dtypes) go in the description.
- Reviewers push back on typical LLM output: unneeded defensive code, over-general helpers and
  hedging, boilerplate module docstrings, comments that don't match the code, and long bullet-list PR
  descriptions. Write the minimal change, and keep docs, comments and descriptions short and accurate.
- Decisions are often made in maintainer meetings and posted back on the issue ("From internal
  discussion..."); read the whole thread before proposing a design.

## Agent skills

Task-specific instructions live in `.agents/skills/` (see `.agents/README.md`): `uxarray-pr-review`
(review a PR and verify each finding by running code) and `uxarray-issue-triage`. Claude Code finds them
through symlinks in `.claude/skills/` and reads this file itself. Keep this the only project
instructions file (no `CLAUDE.md`), so every agent sees the same text.
