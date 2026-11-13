---
name: uxarray-issue-triage
description: Triage a UXarray (UXARRAY/uxarray) GitHub issue like a senior maintainer - classify it, find duplicates and past decisions, try to reproduce it, locate the code involved, and suggest labels and next steps. Use when asked to triage, review or assess a uxarray issue by number or URL.
---

# UXarray issue triage

Triage the issue given as the argument. The output is a short triage note for the maintainer, plus an
optional comment for the issue thread. Read `AGENTS.md` at the repo root first for the architecture and
the settled design decisions.

Nothing is posted or labelled without the user's approval.

## 1. Understand

`gh issue view <n> --comments`. Note what the reporter observed, what they expected, their UXarray
version and install method (the bug template asks for both), the grid format and model (MPAS, ICON,
SCRIP, ...), the data size, whether dask/chunking is involved, and anything they already tried.

Read the discussion as carefully as the report: who has already diagnosed or reproduced it, what was
asked of the reporter and whether they answered, existing labels/assignees, decisions posted from
maintainer meetings ("From internal discussion..."), and PRs or issues that mention it
(`gh api repos/UXARRAY/uxarray/issues/<n>/timeline` → `cross-referenced` events). Build on that rather
than repeating it, and list unanswered questions or disagreements under **Needs your call**.

## 2. Classify

- **Type**: bug, feature request, question/support, docs, performance/scalability, accuracy, CI/infra,
  upstream (an xarray/numba/dask/cartopy change; label `fix upstream`), environment (install, conda vs
  pip, missing optional extra: often not a UXarray bug).
- **Bot issues** ("🤖 Upstream testing failure", "🤖 Minimum dependency testing failure"): find the first
  failing run and the dependency release that broke it; a min-deps failure means a lower bound in
  `pyproject.toml` needs raising (with a comment) or the code needs a compatible path.
- **AI-reported bugs** (e.g. "[Claude] ..." titles, lists of several bugs): check each claim separately
  against current `main`; some may be duplicates or already fixed.
- **Security-sensitive** (exposed credentials, a vulnerability): stop here. Do not draft a public comment
  or reproduce it in public; tell the user, who decides how to handle it privately.
- **Area**: io/reader (which format), grid construction/topology, geometry/accuracy, core
  (`UxDataArray`/`UxDataset` ops, aggregations, calculus), remap, subset/cross-section, plotting, dask,
  docs, packaging/optional deps.
- **Labels**: suggest only from the repo's labels (`gh label list`), e.g. `bug`, `new feature`,
  `improvement`, `documentation`, `scalability`, `accuracy`, `visualization`, `extra dims`, `design`,
  `developer experience`, `fix upstream`, `CI`, `testing`. The repo has no priority, size or "good first
  issue" labels; `may revisit` is for ideas closed as not planned for now.

## 3. Look for duplicates, related work and past decisions

Search open **and** closed issues, PRs and Discussions with 2-3 keyword variants:
`gh search issues --repo UXARRAY/uxarray "<keywords>"`, `gh pr list --state all --search "<keywords>"`,
and Discussions via `gh api graphql` (`repository.discussions` or `search(type: DISCUSSION)`).
Also check whether a PR already claims to fix it (`closedByPullRequestsReferences`:
`gh api graphql -f query='query{repository(owner:"UXARRAY",name:"uxarray"){issue(number:<n>){
closedByPullRequestsReferences(first:10,includeClosedPrs:true){nodes{number title state}}}}}'`).

For feature requests, check them against the settled decisions in AGENTS.md (scope: 2D grids on the
sphere; structured grids only through `from_structured()`; file writing through xarray; `Grid` not
growing Dataset API) and look for an earlier issue closed as not planned or `may revisit`. Link what
you find rather than re-arguing it.

## 4. Locate and reproduce

1. Check whether it is already fixed: get the reporter's version (a dev version such as
   `2026.9.1.dev24+ge01d71e37` names the commit after the `g`; a release maps to tag `v<version>`) and
   run `git log <their-commit>..origin/main -- <files involved>`. If a later commit fixed it, say which.
2. Find the code involved and trace it to the public entry point, as in the PR-review skill's Stage 1
   (`.agents/skills/uxarray-pr-review/SKILL.md`).
3. For a bug, try to reproduce it on current `main` in a clean worktree with the environment from
   `.agents/skills/uxarray-pr-review/references/verification.md` §Setup. Write the smallest script that
   shows it, preferring a mesh from `test/meshfiles/` (table in verification.md) or a face built with
   `Grid.from_face_vertices` over the reporter's large file. Result is one of:
   - **Reproduced** — include the script and its output.
   - **Not reproduced** — say what was tried; ask the reporter for the missing detail (grid file or a
     link to it, version, full traceback).
   - **Needs reporter data** — the bug depends on a grid or dataset that isn't available; say what is
     needed. Many model files live on NCAR GLADE or are large; a subset or the grid file alone often
     suffices.
4. Give a root-cause hypothesis with `file:line`, labelled confirmed or suspected. For geometry bugs, a
   sweep over antimeridian, pole and rotated inputs (verification.md §Techniques, item 2) often shows
   the pattern.

For a **feature request**, instead of reproducing: does it fit UXarray's scope, can it be done today with
existing API (e.g. via `.to_xarray()` and plain xarray), does an xarray analogue define what the API
should look like, and does it overlap open work? For a **question**, find the docs page or user-guide
notebook that answers it; if none does, that is a docs gap worth noting.

## 5. Report

Show the user, in this order and briefly:

```
#<n> <title>
Type / area:            ...
Duplicates / related:   #.. (why), past decisions (link) — or none found
Reproduction:           Reproduced | Not reproduced | Needs reporter data — one line, script path
Likely cause:           file:line — confirmed | suspected
Suggested labels:       ...
Priority (advisory):    high (wrong results, data loss, crash in common path) | medium | low — one-line reason
Size (advisory):        S | M | L — one-line reason
Next step:              who/what, e.g. "needs reporter's grid file", "fix with #PR", "maintainer design call"
Needs your call:        open disagreements or unanswered questions from the thread, if any
```

Priority and size are advice to the maintainer, not repo labels.

Then, if useful, a draft comment for the issue thread (under ~100 words: what was reproduced, what is
needed from the reporter, or the duplicate link). Post the comment or apply labels only after the user
approves, with `gh issue comment` / `gh issue edit --add-label`.
