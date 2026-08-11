# CLAUDE.md

Guidance for Claude working in the UXarray repository.

UXarray is an Xarray extension for **unstructured** climate and global weather data. `UxDataArray`/`UxDataset` subclass `xarray.DataArray`/`Dataset` and carry a `Grid` object describing the mesh topology. The package is **pure Python** (no compiled extensions); hot loops are JIT-compiled with numba. Grid representation follows the **UGRID conventions**, with readers that translate MPAS, ICON, SCRIP, ESMF, FESOM2, Exodus, HEALPix, GEOS-CS, and structured formats into that internal representation.

Mirroring Xarray's API and idioms wherever a natural analogue exists is a core design principle, and it is what a user might expect.

---

## 1. Freshness and precedence

This file is a snapshot, not an oracle. Where it contradicts the repository, the repository wins, but establish that by dates, not by instinct.

Before leaning on a specific claim here, and **especially** before repeating a claim that some other document is out of date:

```bash
git log -1 --format='%ai %h' -- CLAUDE.md
git log -1 --format='%ai %h' -- docs/contributing.rst
git log -1 --format='%ai %h' -- <the files you are about to change>
```

| Situation | What to do |
| --- | --- |
| CLAUDE.md is newer than what it describes | Follow this file. |
| A source file it describes is newer | Read that file before acting on the claim about it. |
| `docs/contributing.rst` is newer than CLAUDE.md | **Treat this file as the suspect one.** Follow the contributor's guide, and tell the user which parts of CLAUDE.md now look stale. |

The last case is the dangerous one: a contributor's guide that has just been revised is more likely to be current than an assistant-facing file that hasn't. Never assert that the guide is stale without checking the dates first.

---

## 2. Working agreements

| Do | Don't |
| --- | --- |
| Keep chat commentary short and factual. Lead with the answer or the change. | Don't narrate each step, restate the request, or summarize what the user just read. |
| Write comments that explain *why* the code does something non-obvious. | **Never leave advice for the user inside the code.** No `# NOTE: you may want to...`, `# TODO(you):`, `# I changed this because...`, `# Claude:`. Code review talk belongs in the chat reply. |
| Match the surrounding file's comment density, naming, and style. | Don't add docstrings/type hints/comments to untouched neighboring code as a drive-by. |
| Change what was asked. Report anything else you noticed. | Don't reformat, rename, or "clean up" beyond the request — `ruff` owns formatting. |
| Reuse existing helpers (§6, §7, §8). | Don't reimplement connectivity building, coordinate conversion, tree queries, or compensated arithmetic. |
| Prefer editing an existing module over creating a new one. | Don't create new top-level modules, scratch scripts, or summary `.md` files unless asked. |
| State uncertainty plainly ("I haven't verified X"). | Don't assert behavior you haven't read or run. |

Beware that LLMs frequently hallucinate in critical areas, like with compensated arithmetic. Alert users to this when changes are needed in accuracy-sensitive regions of the codebase.

When acting as a **reviewer**, see §17. When helping a **user** with analysis rather than development, prefer the public API (`ux.open_dataset`, accessors, `Grid` properties) and never reach into `Grid._ds` or `_uxgrid` in example code.

**AI disclosure is mandatory in PRs.** `.github/PULL_REQUEST_TEMPLATE.md` has an `## AI Disclosure` section requiring the tool/model be named (e.g. `AI Usage: Claude (Opus 5)`) plus two checkboxes — the author takes responsibility for, and has tested, all AI-generated content. If you draft a PR body, insert the model names and versions that were used on the AI line, and leave the boxes **exclusively** as the user's responsibility.

---

## 3. Environment and commands

Work from the repository root. The conda environment defined by `ci/environment.yml` is named `uxarray_build`.

```bash
conda env create -f ci/environment.yml   # first time only
```

Always run with `PYTHONPATH=$PWD` so you exercise the working tree, not a stale installed copy:

```bash
PYTHONPATH=$PWD python -m pytest test
```

| Task | Command |
| --- | --- |
| Full test suite (what CI runs) | `python -m pytest test` |
| One area | `python -m pytest test/grid/geometry` |
| One test | `python -m pytest test/core/test_dataset.py::test_name -x` |
| Lint + format everything | `pre-commit run --all-files` |
| Ruff only | `ruff check . && ruff format .` |
| Build docs | `cd docs && make html` |
| Benchmarks | `cd benchmarks && asv machine --yes && asv continuous --split <base-sha> <head-sha>` |

There are **no registered pytest markers** (no `pytest.ini`, no `[tool.pytest.ini_options]`). Don't write `-m slow` or invent markers; select by path instead.

---

## 4. Repo map

| Path | Contents | Notes |
| --- | --- | --- |
| `uxarray/__init__.py` | The entire public API (`__all__`) | If it isn't here, it isn't top-level public |
| `uxarray/core/` | `UxDataArray`, `UxDataset`, `api.py` (`open_grid`/`open_dataset`/…), gradient, zonal, aggregations | |
| `uxarray/grid/` | The `Grid` class plus all topology/geometry math: `connectivity.py`, `coordinates.py`, `geometry.py`, `arcs.py`, `intersections.py`, `area.py`, `bounds.py`, `neighbors.py`, `dual.py`, `validation.py`, `slicer.py`, `point_in_face.py`, `integrate.py` | Where most numerics live |
| `uxarray/conventions/` | `ugrid.py`, `descriptors.py` — **the naming authority** (§6) | Data, not logic |
| `uxarray/io/` | One reader per format (`_ugrid.py`, `_mpas.py`, `_scrip.py`, `_esmf.py`, `_icon.py`, `_fesom2.py`, `_exodus.py`, `_geos.py`, `_healpix.py`, `_topology.py`, `_vertices.py`, `_structured.py`, `_geopandas.py`) + `utils.py` dispatch | §10 |
| `uxarray/remap/` | `accessor.py` + one module per method; `weights.py` caches loaded weight files; `yac.py` optional backend | §10 |
| `uxarray/subset/` | Real subsetting logic, `Grid.subset` / `UxDataArray.subset` accessors | Add new subsetting here |
| `uxarray/cross_sections/` | Accessor methods here emit `DeprecationWarning` and forward to `.subset.*`. `sample.py` holds live geodesic-sampling helpers | Don't add features to the accessors |
| `uxarray/plot/` | hvplot/holoviews-backed accessors; `utils.py` owns the `backend` singleton | §10 |
| `uxarray/utils/computing.py` | Compensated / error-free floating-point primitives (numba) | §8 |
| `uxarray/tutorial/` | Hardcoded `DATASETS` registry; resolves from `test/meshfiles/` or `UXARRAY_DATA_DIR`, else downloads to `~/.cache/uxarray/tutorial` via plain `urllib` | Network access on cache miss |
| `uxarray/formatting_html.py` | Jupyter `_repr_html_` for `Grid`/`UxDataset`/`UxDataArray` | Reads the §6 name lists |
| `test/` | 70 files, ~546 tests; data committed under `test/meshfiles/` | §12 — **ruff-excluded** |
| `benchmarks/` | Flat ASV suite + `asv.conf.json` | §13 — **ruff-excluded** |
| `docs/` | Sphinx; `api.rst` is a hand-maintained API index | §15 |
| `ci/` | `environment.yml`, `asv.yml`, `docs.yml`, `install-upstream.sh` | |

Ignore `build/`, `uxarray.egg-info/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`.

---

## 5. The core data model

- `Grid` (`uxarray/grid/grid.py`) is **not** an Xarray subclass. It wraps a plain `xr.Dataset` of UGRID variables in `self._ds`, and exposes everything through properties.
- `UxDataArray` / `UxDataset` subclass their Xarray counterparts and must declare `__slots__` (Xarray requires it); that is why `uxgrid` cannot be an ordinary attribute.
- A `UxDataset` and every `UxDataArray` pulled out of it share **one `Grid` instance by reference**. Only `_copy(deep=True)` clones it.

**The `_uxgrid` vs `uxgrid` contract** — get this backwards and you either crash in the wrong place or silently propagate `None`:

| Use | When |
| --- | --- |
| `self._uxgrid` | Non-grid-aware internals only: `_copy`, `_replace`, `__init__`, binary-op plumbing. May legitimately be `None` because Xarray's `_replace`/`_construct_direct` reconstruct objects without kwargs. |
| `self.uxgrid` | Everything grid-aware (`integrate`, `plot`, `remap`, `face_areas`, …). Raises `GridInvalidError` if unset. |

Constructing `UxDataArray(...)` without `uxgrid=` succeeds silently and fails later on first grid-aware use. When debugging a `GridInvalidError` reading *"Maybe you forgot to provide uxgrid"*, look for a construction path that dropped the grid.

---

## 6. Grid variable names are declared, not invented

`uxarray/conventions/ugrid.py` is the single source of truth for every dimension, coordinate, and connectivity name, along with its `dims` tuple and CF-style `attrs`. `uxarray/conventions/descriptors.py` does the same for descriptors.

| Category | Names | Registry to update |
| --- | --- | --- |
| Dims | `n_node`, `n_edge`, `n_face`, `n_max_face_nodes`, `n_max_face_edges`, … | `DIM_NAMES` |
| Spherical coords (degrees) | `node_lon`/`node_lat`, `edge_*`, `face_*` | `SPHERICAL_COORDS`, `SPHERICAL_COORD_NAMES` |
| Cartesian coords | `node_x`/`node_y`/`node_z`, `edge_*`, `face_*` | `CARTESIAN_COORDS`, `CARTESIAN_COORD_NAMES` |
| Connectivity | `face_node_connectivity`, `edge_node_connectivity`, `face_edge_connectivity`, `face_face_connectivity`, `edge_face_connectivity`, `node_face_connectivity`, `node_edge_connectivity` | `CONNECTIVITY`, `CONNECTIVITY_NAMES` (`UGRID_COMPLIANT_CONNECTIVITY_NAMES` is the strict-UGRID subset; the two `node_*` ones are UXarray extensions) |
| Descriptors | `n_nodes_per_face`, `face_areas`, boundary/hole indices, … | `DESCRIPTOR_NAMES` |

Never hardcode a name or an `attrs` dict at a use site — import from `uxarray.conventions`. These lists drive `Grid.dims`/`.coordinates`/`.connectivity`/`.descriptors`, `repr(grid)`, and the HTML repr, so forgetting to register a new variable makes it work but stay invisible.

`constants.GRID_DIMS` is a *narrower* list (`n_node`, `n_edge`, `n_face`) used only to enforce "one grid dimension per `.isel()`". Don't confuse it with `DIM_NAMES`.

### The lazy-populate idiom

There is no `cached_property`. `Grid._ds` **is** the cache; presence in `_ds` short-circuits recomputation:

```python
@property
def edge_node_connectivity(self) -> xr.DataArray:
    if "edge_node_connectivity" not in self._ds:
        _populate_edge_node_connectivity(self)
    return self._ds["edge_node_connectivity"]

edge_node_connectivity = edge_node_connectivity.setter(make_setter("edge_node_connectivity"))
```

`make_setter` (`uxarray/grid/utils.py`) type-checks for `xr.DataArray` and assigns into `_ds`. Populate functions write back with the declared constants:

```python
grid._ds["face_edge_connectivity"] = xr.DataArray(
    data=face_edges,
    dims=ugrid.FACE_EDGE_CONNECTIVITY_DIMS,
    attrs=ugrid.FACE_EDGE_CONNECTIVITY_ATTRS,
)
```

**To add a derived connectivity/coordinate, touch all of:**
1. `conventions/ugrid.py` (or `descriptors.py`) — add `_ATTRS` + `_DIMS`, then register the name in the matching list/dict.
2. `grid/<area>.py` — write `_populate_<name>(grid)`; put heavy loops in a separate module-level `@njit` function.
3. `grid/grid.py` — import `_populate_<name>` in the top import block.
4. `grid/grid.py` — add the guarded `@property` in the right section, then the `make_setter` line immediately after.
5. `docs/api.rst` — add the name (§15), or it won't be documented.
6. `test/` — add a test asserting it populates on demand and has the expected dims.

---

## 7. Numeric conventions

| Constant (`uxarray/constants.py`) | Meaning / rule |
| --- | --- |
| `INT_DTYPE = np.intp` | The index dtype. Chosen because numpy indexing is written for `intp` — don't substitute `int32`/`int64`. |
| `INT_FILL_VALUE = np.iinfo(INT_DTYPE).min` | Padding sentinel for ragged connectivity (e.g. triangles in a mesh padded to `n_max_face_nodes`). **Never `np.nan`** — these arrays are integer dtype. Test with `== INT_FILL_VALUE`, and set `_FillValue` in the variable's `attrs`. |
| `ERROR_TOLERANCE = 1e-8` | Geometric "is this degenerate/normalized" checks: near-zero face area, coordinate-norm deviation, pole snapping. |
| `MACHINE_EPSILON` | Only for near-machine-precision comparisons in arc/intersection math. Not interchangeable with `ERROR_TOLERANCE`. |
| `WGS84_CRS` | Used by the geopandas reader. |

**Coordinate invariants:**
- All internal computation is on a **unit sphere** (radius 1.0). The physical radius is preserved separately as `Grid.sphere_radius`.
- Spherical coords are **degrees**, longitude normalized to **[-180, 180]** (`_set_desired_longitude_range`, re-applied after any centroid/edge-coordinate population).
- Internal radian helpers normalize longitude to **[0, 2π]** instead. Degrees and radians use *different* canonical ranges — a recurring bug source when mixing them.
- Cartesian coords are unit-length; `_check_normalization` compares `x²+y²+z²-1` against `ERROR_TOLERANCE` and caches into `grid._normalized`.
- Near the poles (|z| ≈ 1) longitude is forced to `0.0` and latitude snapped to ±π/2 rather than trusting `arctan2`.
- Conversion helpers take `normalize: bool = True`; pass `False` on hot paths where inputs are already unit vectors.

Only `INT_DTYPE` and `INT_FILL_VALUE` are re-exported at top level — those two are user-facing (for hand-building connectivity); the rest are internal.

---

## 8. numba

- Style is `@njit(cache=True)`, sometimes with `parallel=True` + `prange`, `nogil=True`, `inline="always"`, or `error_model="numpy"`.
- JIT functions must be **module-level** and take/return plain arrays and scalars. `self`, `xr.DataArray`, and dicts cannot cross the boundary — the `_populate_*` wrapper unwraps to numpy, calls the kernel, and re-wraps.
- Cast inputs to a concrete numeric dtype before the call; object arrays fail.
- Heterogeneous containers need explicit numba types (`types.UniTuple`, `numba.typed.List`). Deliberately simple algorithms (e.g. insertion sort for small arrays) are sometimes correct here because numba can't compile `sorted(key=...)`.
- There is **no project-level JIT toggle**. To step through a kernel, set `NUMBA_DISABLE_JIT=1` in the environment.
- First call to a kernel pays compilation cost. That is why benchmarks warm up in `setup()` (§13), and why a "slow" first call in a profile is often compilation rather than the algorithm.
- Before writing new spherical arithmetic, check `uxarray/utils/computing.py` — it holds compensated primitives (`two_sum`, `diff_of_products`, `accucross`, `acc_sqrt_re`, …) that exist specifically to avoid catastrophic cancellation for near-parallel great-circle arcs. Naive `@njit` arithmetic there is the bug this module was written to fix.
- Be aware that dask and numpy may conflict with what works best for numba. In general, we prefer numba to work at a lower level in hot paths, and dask/numpy as chunked or non-chunked dispatchers.

---

## 9. Errors

All custom types live in `uxarray/errors.py` (see issue #1556) — define new ones there, never inline elsewhere.

| Exception | Base | Raise when |
| --- | --- | --- |
| `DataCenteringError` | `ValueError` | Data centering mismatch (expected face-centered, got node-centered, …) |
| `DimensionError` | `ValueError` | Wrong dimension name, count, size, or ndim |
| `GridInvalidError` | `ValueError` | Grid isn't a valid UXarray grid: unrecognized format, duplicate nodes, negative face area, failed `validate()`, missing `uxgrid` |
| `GridsMismatchError` | `ValueError` | An operation spans two incompatible `Grid` objects |
| `YacNotAvailableError` | `RuntimeError` | YAC backend requested but unavailable |

Plain `TypeError`/`ValueError` remain correct for generic argument validation — the custom types are for these domain concepts, not a blanket replacement. Note the layering in `Grid.validate()`: low-level `_check_*` helpers in `validation.py` `warnings.warn(...)` and return a bool; only the top-level `validate()` raises.

---

## 10. Extension checklists

### New grid file format
Detection and dispatch are **two separate if/elif chains**, not a registry — order matters, first match wins.
1. `io/utils.py` — add `_is_<fmt>(dataset)` and a branch in `_parse_grid_type` (raises `GridInvalidError` if nothing matches).
2. `io/_<fmt>.py` — add `_read_<fmt>(dataset)` returning `(grid_ds: xr.Dataset, source_dims_dict: dict)` mapping source dim names → UGRID dim names.
3. `grid/grid.py` — import the reader and add the matching branch in `Grid.from_dataset`.
4. `io/utils.py::_get_source_dims_dict` — add a case if *data* variables need dim renaming too.
5. `core/utils.py::_map_dims_to_ugrid` — add a branch only for exotic dim handling (e.g. GEOS-CS cube-sphere stacking). `match_chunks_to_ugrid` in the same file needs the dims resolvable for dask chunking to work.
6. `test/io/` + a small mesh file under `test/meshfiles/<fmt>/`, and list the format in `docs/user-guide/grid-formats.rst`.

Formats that come from something other than a dataset (`from_topology`, `from_structured`, `from_points`, `from_face_vertices`, `from_healpix`, geopandas via file extension, FESOM2 ASCII directories) bypass `_parse_grid_type` with their own `Grid` classmethod. HEALPix additionally skips `_validate_minimum_ugrid`, because `pixels_only=True` grids have only `face_lon`/`face_lat`.

### New remap method
Add `_<name>_remap(source, destination_grid, remap_to="faces", **kwargs)` in `remap/<name>.py`, reusing `remap/utils.py` (`_assert_dimension`, `_to_dataset`, `_get_remap_dims`, `_construct_remapped_ds`, `LABEL_TO_COORD`, `KDTREE_DIM_MAP`). Add the public method to `RemapAccessor`, update its `__repr__` listing, and register in `docs/api.rst`. Note YAC is a *backend* (`backend="yac"`), not a method — every accessor method takes `backend`/`yac_method`/`yac_options` and lazily imports `_yac_remap` inside the branch. Reuse `remap/weights.py` (module-level LRU cache) rather than re-parsing weight files.

### New subsetting method
Implement on `GridSubsetAccessor` returning `self.uxgrid.isel(n_face=..., inverse_indices=...)`, add the mirrored method on `DataArraySubsetAccessor`, update both `__repr__` listings. Accessors are attached with `UncachedAccessor(...)` on `Grid` and `UxDataArray`. Do **not** add to `uxarray/cross_sections/` — those accessor methods only forward with deprecation warnings.

### New plot method
Add to the relevant accessor in `plot/accessor.py`; call `plotting_backend.assign(backend)` first (the `backend` singleton in `plot/utils.py` — don't call `hv.extension()` directly), build a (geo)pandas frame, delegate to `.hvplot.*`. Keep heavy imports (`cartopy.crs`) inside the method.

`uxarray/plot/*` carries a ruff `E402`/`F401` exemption in `pyproject.toml`. The relevant case is `import hvplot.xarray` inside `_ensure_hvplot_imported()`: it is imported for its **side effect** of registering the `.hvplot` accessor and is never referenced by name, and it is deferred because eager import measurably slowed `import uxarray` (see the comment at the top of `plot/accessor.py`). Preserve that laziness — `test/test_dependencies.py` asserts plotting stacks aren't pulled in by `import uxarray`.

---

## 11. Optional dependencies — three different rules

This distinction matters and is easy to get wrong.

| Area | Idiom | Example |
| --- | --- | --- |
| **Source** (`uxarray/`) | Lazy import inside the function; on failure raise a **typed error from `errors.py`** with an actionable message. Never skip silently, never `pytest` anything. | `remap/yac.py::_import_yac()` → `YacNotAvailableError("YAC backend requested but 'yac.core' is not available…")` |
| **Tests** (`test/`) | `try: import x / except ImportError: pytest.skip(...)`, or a module-level skip via the source's own probe, or `@pytest.mark.skipif` reusing a source capability flag. | `test/test_remap_yac.py` calls `_import_yac()` and skips with `allow_module_level=True` |
| **Benchmarks** (`benchmarks/`) | No dependency guarding — the ASV env is assumed complete. Only *data* availability is conditional (params shrink when NCAR/Glade paths are absent). | `bench_connectivity.py` |

`pytest.importorskip` is not used anywhere in this repo; follow the existing `try/except` + `pytest.skip` form.

**YAC is the only genuinely optional dependency.** It's absent from both `pyproject.toml` and `ci/environment.yml`, is built from source (YAXT + YAC + MPI) only in `.github/workflows/yac-optional.yml`, and that workflow triggers solely on changes to `uxarray/remap/**`, `test/test_remap_yac.py`, or itself.

Everything else that looks optional is not: `hvplot`, `holoviews`, `cartopy`, `geoviews`, `geopandas`, `datashader`, `spatialpandas`, `pyproj` are **hard** dependencies in `pyproject.toml` that are merely imported lazily for startup speed. Don't wrap them in `try/except ImportError` or add "please install" messages. (`IPython` is the one exception — guarded defensively in `plot/utils.py` with a silent fallback.)

---

## 12. Tests

- **Style: bare pytest functions.** 457 module-level `def test_*` versus 89 methods, and **zero** `unittest.TestCase`. Write new tests as module-level functions. A handful of plain `class Test…:` containers exist purely for grouping variants (no `setUp`); use one only if you're extending such a group.
- Layout: `test/<area>/test_<topic>.py`, mirroring the source (`test/core/`, `test/grid/grid/`, `test/grid/geometry/`, `test/grid/integrate/`, `test/io/`). Two files don't follow the convention (`test/precomputed_weights_test.py`, `test/test_placeholder.py`) — follow the majority pattern.
- **Test data** is committed under `test/meshfiles/<format>/<dataset>/`. Always reach it through the fixtures in `test/conftest.py` (the only conftest) — never build paths from `__file__` in a test:

```python
def test_something(gridpath, datasetpath):
    uxds = ux.open_dataset(
        gridpath("ugrid", "quad-hexagon", "grid.nc"),
        datasetpath("ugrid", "quad-hexagon", "data.nc"),
    )
```

- Use the `mesh_constants` fixture for expected node counts/areas instead of new magic numbers; `test_data_dir` gives the base directory.
- `@pytest.mark.parametrize` is used sparingly but is welcome for value sweeps.
- Compare floats with `np.testing.assert_allclose` / `ERROR_TOLERANCE`-scaled tolerances, not `==`.
- Coverage is soft-gated: `.codecov.yml` disables patch/changes checks and allows a 0.2% project drop, with no PR comment. Write tests because the behavior needs pinning, not to move the number.

## 13. Benchmarks

ASV, flat files under `benchmarks/`, configured by `benchmarks/asv.conf.json` (conda, Python 3.11, env from `ci/asv.yml`).

- Shape: a base class with `params` / `param_names` and `setup()` / `teardown()`, subclasses adding `time_*` / `peakmem_*` methods that consume what `setup` built. `timeraw_*` for process-level measurements (`benchmarks/import.py`).
- `setup()` is not timed — put fixture construction **and numba warm-up** there. `bench_connectivity.py` does this explicitly, its comment citing ~240 ms of JIT compilation that would otherwise be charged to whichever sample ran first.
- Set `number = 1` when repeat calls would measure a cache hit rather than real work.
- Extend params by concatenation when subclassing: `param_names = DatasetBenchmark.param_names + ["exclude_antimeridian"]`.
- **No assertions** — benchmarks measure, tests validate.
- Data downloads on first run into `benchmarks/` itself via `urllib` (the `current_path = Path(os.path.dirname(os.path.realpath(__file__)))` idiom, which is correct here and wrong in `test/`). Some params only appear on Glade.
- PR benchmarking runs only with the `run-benchmark` label, in two workflows split for security: the PR job runs untrusted code with a read-only token and uploads an artifact; `asv-benchmarking-comment.yml` posts the comment afterward (issue #1547). Don't merge them.

## 14. Lint and formatting

`pyproject.toml` sets `[tool.ruff] extend-exclude = ["test", "benchmarks"]`, and ruff honors that config when invoked via pre-commit too. Practically:

| | `uxarray/` | `test/`, `benchmarks/` |
| --- | --- | --- |
| ruff lint, import sort (`I`), format | Enforced | **Not enforced** |
| `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`, `check-docstring-first`, `debug-statements` | Enforced | Enforced |

So don't "fix" formatting or import order in `test/`/`benchmarks/` — it's out of scope and creates noise. Do still avoid trailing whitespace, missing final newlines, and stray `breakpoint()`/`pdb` (that last hook will block the commit). `pre-commit.ci` reports on PRs but does not auto-push fixes (`autofix_prs: false`).

`E731` (lambda assignment) is globally ignored; `docs/*` and `uxarray/plot/*` additionally ignore `E402`/`F401`.

---

## 15. Docs and docstrings

**Use numpydoc.** The source is uniformly numpydoc and `sphinx.ext.napoleon` renders it. Because `docs/conf.py` sets `autodoc_typehints = "none"`, the `Parameters` section is the *only* place types reach the rendered docs — document the type in prose even when the signature is annotated.

```python
def open_dataset(grid_filename_or_obj, filename_or_obj=None, chunks=None) -> UxDataset:
    """Wraps ``xarray.open_dataset()`` for loading in a dataset paired with a grid file.

    Parameters
    ----------
    grid_filename_or_obj : str | os.PathLike[Any] | dict | xr.Dataset
        Grid information for the ``UxDataset``. Strings and Path objects are
        interpreted as a path to a grid file.
    chunks : int, dict, 'auto' or None, default: None
        If provided, used to load the grid into dask arrays.

    Returns
    -------
    uxds : uxarray.UxDataset
        Dataset with linked `uxgrid` property of type `Grid`.

    Notes
    -----
    Optional extra context.

    Examples
    --------
    >>> import uxarray as ux
    >>> ux_ds = ux.open_dataset("grid_file.nc", "data_file.nc")
    """
```

- Section order: one-line summary → optional prose → `Parameters` → `Returns` → `Notes` → `Examples`. Double-backtick code, single-backtick cross-references.
- **`Examples` are not doctested** — no doctest runner exists in CI. They must be illustrative and correct-looking, but need not be executable; inventing plausible printed output is worse than omitting it.
- Underscore-prefixed internals often have a one-line docstring or none. That's acceptable; full numpydoc is for public API.

**New public API is invisible until you list it.** `docs/api.rst` is a hand-maintained `autosummary` index. Append the dotted name under the right section/subsection:

```rst
Methods
~~~~~~~
.. autosummary::
   :toctree: generated/

   Grid.compute_face_areas
   Grid.validate
```

Accessor methods use `:template: autosummary/accessor_method.rst` and list both the accessor and its methods (`Grid.subset`, `Grid.subset.bounding_box`, …). Private functions are simply omitted — the PR checklist requires internal names to start with `_`.

**Notebooks** are executed at build time by `myst-nb` (`nb_execution_timeout = 120`), except those in `nb_execution_excludepatterns`. Rules:
- User-guide notebook → `docs/user-guide/`, then add it to the toctree/list in `docs/userguide.rst`.
- Gallery example → `docs/examples/`, register in `docs/gallery.rst` **and** `docs/gallery.yml`, plus a thumbnail in `docs/_static/thumbnails/`.
- Quickstart → `docs/getting-started/`, referenced from `docs/quickstart.rst`.
- **Clear all cell outputs before committing.**

## 16. Terminology

Use the project's vocabulary consistently — in code, docstrings, docs, commit messages, and PR text.

| Use | Not |
| --- | --- |
| **node** | vertex, corner (acceptable only as a parenthetical gloss) |
| **edge** | segment, side |
| **face** | cell, element (generic prose only) |
| **connectivity** | adjacency, mapping |
| **fill value** | missing/masked value |
| **grid**, "unstructured grid" | mesh (reserved for specific terms like "dual mesh", "MPAS mesh type") |

Reference definitions: `docs/user-guide/terminology.rst`, `docs/user-guide/representation.rst`.

---

## 17. Reviewing

Beyond correctness and tests, check for:

- [ ] Hardcoded variable names, dims, or `attrs` dicts instead of `uxarray.conventions` constants (§6).
- [ ] New registry entry missing → variable works but is absent from `repr`, `.connectivity`, and the HTML repr.
- [ ] New public API not added to `docs/api.rst`; new private helper missing its `_` prefix.
- [ ] `self.uxgrid` where `self._uxgrid` belongs, or vice versa (§5).
- [ ] `np.nan`, `-1`, or a bare `0` where `INT_FILL_VALUE` belongs; non-`INT_DTYPE` index arrays.
- [ ] `==` on floats, or a hand-rolled tolerance instead of `ERROR_TOLERANCE` / `MACHINE_EPSILON` (and the right one of the two).
- [ ] Degrees/radians mixed, or longitude range assumptions ([-180, 180] vs [0, 2π]) crossed (§7).
- [ ] Naive arithmetic in geometry kernels where `utils/computing.py` primitives exist (§8).
- [ ] `@njit` receiving Python objects, or an expensive `_populate_*` call inside a loop rather than relying on `_ds` caching.
- [ ] Reaching into `Grid._ds`, `_kdtrees`, `_ball_tree`, `_normalized`, or other private state from outside `grid.py`.
- [ ] Eager import of a plotting/heavy dependency at module top (breaks `test_dependencies.py` and import time).
- [ ] `try/except ImportError` around a hard dependency, or a bare `pytest.skip` idiom used in source (§11).
- [ ] Features added to the `cross_sections/` accessors instead of `subset/`.
- [ ] Ruff-driven reformatting of `test/`/`benchmarks/` mixed into a functional change (§14).
- [ ] Comments addressed to the reader/reviewer rather than explaining the code (§2).
- [ ] Notebook outputs committed; missing AI disclosure in the PR body.

## 18. Known pins and divergences

| Thing | Reality |
| --- | --- |
| `docs/contributing.rst` vs the code | The guide currently prescribes Google-style docstrings, `unittest`-based tests, and an API index at `docs/user_api/index.rst`. The code is numpydoc, pytest functions, and `docs/api.rst`. As of this snapshot the guide is the older file, so the code wins — but **run the §1 freshness check before repeating that**. Reconciling the guide is a human task: flag it, don't silently change either side. |
| `pyproject.toml` claims Python 3.10–3.14 | CI tests 3.11–3.13 on ubuntu/macos/windows (9 combos, `fail-fast: false`). |
| `matplotlib<3.11` | Pinned in three places — `pyproject.toml`, `ci/environment.yml`, `ci/asv.yml` — each citing issue #1542. Change all three or none. |
| `numba>=0.63` | Required for Python 3.14 support (issue #1561). |
| upstream-dev CI | Daily cron only (not on PRs), Python 3.13, installs nightly/git-main upstreams. `continue-on-error`, auto-files a `CI`-labeled issue on failure. pandas is deliberately held back due to a geopandas incompatibility (issue #1414). A red run here is not necessarily caused by your PR. |
