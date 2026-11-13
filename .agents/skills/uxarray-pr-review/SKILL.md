---
name: uxarray-pr-review
description: Review a UXarray (UXARRAY/uxarray) pull request like a senior maintainer and verify every finding by running code before reporting it. Use when asked to review, check or comment on a uxarray PR by number, branch or URL.
---

# UXarray PR review (self-verifying)

Review the pull request given as the argument (number, branch or URL). The goal is a review that is
**deep but short**, where every finding has been **checked by running code**, not just by reading it.
Read `AGENTS.md` at the repo root first: it holds the architecture and the repo rules this review
checks against.

Work in four stages. Do not skip stage 3: a finding that has not been through it is not reported as fact.

## Stage 0 — Set up (once per PR)

Follow [references/verification.md](references/verification.md) §Setup: a worktree for the PR, a clean
worktree of the base branch, and a Python environment that imports the tree under test. Never switch
branches in the user's own checkout.

PRs from forks run someone else's code on this machine. For an unfamiliar author, say so and ask
before running anything.

## Stage 1 — Context

1. `gh pr view <n>` (body, author, commits, `closingIssuesReferences`) and every linked issue with its
   comments. Note what the PR **claims** to do: you will check the code against those claims. Maintainer
   decisions are often posted on the issue as "From internal discussion..." comments; the PR must follow
   them.
2. **Review history** — follow [references/history.md](references/history.md): earlier reviews and the
   commit each one reviewed, every inline thread and comment, commits since the last review, description
   edits, force-pushes, CI status. Build the ledger of points raised and their status, and note
   disagreements and unanswered questions. Do this even when the PR looks fresh.
3. `git diff <base>...HEAD --stat`, then the full diff. If there was an earlier review, also the diff since
   that review's commit: it is the least-reviewed code in the PR.
4. For every changed function, **trace its callers up to a public entry point** (`ux.open_*`, a `Grid`
   property or classmethod, a `UxDataArray`/`UxDataset` method, an accessor method such as `.remap.*`,
   `.subset.*`, `.plot.*`). Record:
   - which grid formats and readers reach it, and whether faces can be mixed-size (padded with
     `INT_FILL_VALUE`), cross the antimeridian, contain a pole, or come from a projected grid;
   - which data locations it handles (node / edge / face) and which it rejects;
   - whether it runs on numpy or chunked (dask) data, and whether the caller can pass either;
   - which options select the path (`periodic_elements`, `backend`, `remap_to`, `destination`,
     `latlon`, `inverse_indices`, ...).
   This step decides which findings are possible at all. Skipping it produces false positives.

## Stage 2 — Find candidates

Look at the diff along these dimensions. Use parallel subagents (one per dimension) when the PR is
large; for small PRs, do it in one pass.

- **Correctness and numerics**: degrees vs radians and the two longitude ranges ([-180, 180] vs
  [0, 2π)); `INT_FILL_VALUE` padding handled (not `-1`, `0`, `np.nan`); `ERROR_TOLERANCE` vs
  `MACHINE_EPSILON`; antimeridian and pole faces; unit sphere vs `sphere_radius` scaling; NaN comparison
  semantics; dtype promotion and integer overflow (int32 vs int64, Windows); `np.isclose(a, b, tol)`
  passing `rtol` by position.
- **Accuracy-sensitive geometry**: naive arithmetic replacing `utils/computing.py` primitives,
  `fastmath=True`, branching inside hot kernels, comments that overclaim ("robust", "exact",
  "branch-free") or don't match the code. Compare against an analytic value or the existing accuracy
  baselines (`test/grid/geometry/test_eft_accuracy.py`, `test_accusphgeom_baseline.py`).
- **Grid and data model** (AGENTS.md "Core data model"): `uxgrid` lost or not sliced after an op (e.g.
  `sel` vs `isel`); coords and attrs dropped; objects on different grids combined without
  `GridsMismatchError`; `_uxgrid` vs `uxgrid` misuse; topology mutated in place instead of returning a
  new `Grid`; a connectivity renumbered without its pair; new grid variables not registered in
  `uxarray/conventions` (works but invisible in `repr`).
- **Dask and performance**: `.values` / `np.asarray` on whole arrays, eager compute or silent rechunking
  of chunked input; numba kernels allocating small arrays in loops; raw `njit(parallel=True)` instead of
  `parallel_njit`; work done at import time; heavy imports at module level. Does a hot-path change need
  the `run-benchmark` label or a new ASV benchmark?
- **API and back-compat**: consistency with xarray and existing UXarray signatures, argument order,
  defaults and names; public vs `_private`; behavior changes not stated in the description; removals
  without a `FutureWarning` cycle; structured-grid handling leaking into core APIs; new write APIs
  (AGENTS.md "Core data model").
- **Optional dependencies**: optional packages imported at module level or without
  `_raise_hint_if_optional_deps_missing`; a new optional dep missing from `_OPTIONAL_DEPS_TO_EXTRAS`; a
  newer dependency API used without raising the lower bound (min-deps CI).
- **Errors**: right type (`uxarray/errors.py` for domain errors, built-ins for generic checks), message
  with expected and actual values and no private names, no `assert` for validation, no broad
  `try/except` hiding the cause, invalid input raising rather than silently changing results.
- **Tests**: do the new tests fail without the fix (stage 3 checks this)? Do they assert what their
  name says? Are tolerances and expected values justified? Are mixed-size, antimeridian, pole, extra-dim
  and chunked cases covered where the caller allows them?
- **Docs and PR hygiene**: numpydoc docstrings (types in `Parameters`); new public API in `docs/api.rst`;
  a user-guide section or notebook for a new user-facing feature; notebook outputs cleared (unless the
  notebook is in `nb_execution_excludepatterns`); description matches the code; "Closes" vs "Related
  to"; AI disclosure line filled in and its box ticked.
- **Scope and LLM habits**: unrelated changes, PRs that should be split; unneeded defensive code,
  over-general helpers, hedging, boilerplate module docstrings (AGENTS.md "Pull requests").
- **Scientific-software practice**: reproducibility, units, readability, maintainability.

For each candidate, write down: the claim, `file:line`, the concrete failure scenario, and how you will
verify it. Candidates are hypotheses, not findings.

Check each candidate against the ledger from Stage 1: drop it if someone already raised it, unless you
have new evidence, in which case it becomes a reply in that thread or a line in the summary.

## Stage 3 — Verify

Apply the techniques in [references/verification.md](references/verification.md) §Techniques. Then label
each candidate:

- **CONFIRMED** — reproduced by running code (a failing test, a mutation that survives, an experiment).
- **PLAUSIBLE** — the code path is real but cannot be run here (e.g. needs YAC, NCAR GLADE data or a
  large mesh). State why it cannot be run.
- **Dropped** — disproved. Keep a one-line note of why: dropped candidates show the review was thorough
  and stop the next reviewer from raising them again.

Always run, whatever the candidates: the PR's new and changed tests on the PR; the tests of the touched
area on the PR; a **mutation test of each source hunk** (revert it, rerun the tests; a hunk whose
reversion no test notices is a coverage finding); ruff on changed files under `uxarray/`.

For a PR with **no source hunks** (docs, notebooks, CI, agent files), mutation testing does not apply.
Instead: build the docs if `docs/` or docstrings changed and compare warnings with base; run touched
notebooks; and check that every code reference in the changed text (paths, functions, options, commands)
exists on the PR branch and says what the text claims.

For a **refactor or performance PR** (behavior should not change), compare outputs on base vs PR
(verification.md §Techniques, item 4) and time both (item 6).

## Stage 4 — Report

Write the draft to a file in the scratchpad using [references/report.md](references/report.md), then
**show it to the user and wait**. Post only after explicit approval, exactly as approved, following
report.md §Posting. Then clean up per verification.md §Cleanup (keep the PR worktree until the PR merges
if a re-review is likely).
