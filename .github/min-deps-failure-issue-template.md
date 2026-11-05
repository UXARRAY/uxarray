# Minimum dependency testing failure

**Workflow:** {{WORKFLOW}}
**Run:** [{{RUN_ID}}]({{RUN_URL}})
**Date:** {{DATE}}

Tests failed with the oldest dependency versions that `pyproject.toml` still
allows. The versions tested are in the run summary.

To fix:

1. Find the package whose old release causes the failure (the traceback usually
   points to it).
2. Add a lower bound for it in `pyproject.toml`, with a comment saying what
   breaks with older releases.

This issue was automatically generated from the CI Minimum Dependencies workflow.
