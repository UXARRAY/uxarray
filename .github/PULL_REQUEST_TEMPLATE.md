<!--  The PR title should summarize the changes, for example "Add `Grid._build_face_dimension` function".
      Avoid non-descriptive titles such as "Addresses issue #229". -->

<!--  Replace XXX with the issue number resolved by this PR, if this PR fully resolves an issue.
      If it does not fully resolve any issues, replace with something like "Related to #XXX",
          or "Fixes part of #YYY but does not fully close it."
      If it resolves multiple issues, repeat "closes" for each, like "Closes #XXX, closes #YYY." -->
Closes #XXX

## Overview
<!--  Please provide a few bullet points summarizing the changes in this PR. This should include
      points on any bug fixes, new functions, or other changes that have been made. -->

## Expected Usage
<!--  If this PR adds a new feature, please provide a short example of it in action.
      You may ignore this step if it is not applicable (delete this section). -->
```Python
import uxarray as ux

grid_path = "/path/to/grid.nc"
data_path = "/path/to/data.nc"

uxds = ux.open_dataset(grid_path, data_path)

# this is how you use this function
some_output = uxds.some_function()

# this is another way to use this function
other_output = uxds.some_function(some_param = True)
```

## PR Checklist
<!-- Please mark any checkboxes that do not apply to this PR as [N/A]. -->

**General**
- [ ] An issue is created and linked
- [ ] Added appropriate labels (if your uxarray repo permissions allow it)
- [ ] Filled out Overview and Expected Usage (if applicable) sections

**Testing & Benchmarking**
<!--  If this PR does not update any functionality or tests and is unlikely to affect efficiency,
    e.g. by affecting computation time or memory usage, remove this section (Testing & Benchmarking) -->
- [ ] Adequate tests are created if there is new functionality
- [ ] Tests are not too basic (such as simply calling a function and nothing else)
- [ ] Tests cover all major paths in your new functions
- [ ] If this PR could affect performance, ran ASV benchmarks and confirmed they show expected behavior (add a new benchmark if necessary)
<!-- Adding the run-benchmark label (if your uxarray repo permissions allow it) will run ASV benchmarks.
    If you need benchmarks to be run but don't have permissions, leave this item unchecked for now. -->

**Documentation**
<!--  If this PR does not update any functionality or docstrings, remove this section (Documentation) -->
- [ ] Docstrings have been added to all new functions
- [ ] Docstrings have been updated with any function changes
- [ ] User (public) functions have been added to `docs/api.rst`
- [ ] Internal (private) function names start with an underscore (`_`)

**Examples**
<!--  If this PR does not affect any example notebooks, remove this section (Examples) -->
- [ ] **All** notebook examples cleared the output of all cells before committing
- [ ] New notebook examples added to appropriate folder (gallery: `docs/examples/`; guide: `docs/user-guide/`; quickstart: `docs/getting-started/`)
- [ ] New notebook examples referenced in appropriate .rst file (gallery: `docs/gallery.rst`; guide: `docs/userguide.rst`; quickstart: `docs/quickstart.rst`)
- [ ] New notebook gallery examples added entry in `docs/gallery.yml` with appropriate thumbnail photo in `docs/_static/thumbnails/`


## AI Disclosure
<!-- If you did not use AI, please write "AI Usage: N/A" and remove these checklist items or mark as [N/A].
    Otherwise, please briefly specify all tools used and how they were used. Include model if known. Examples:
    "AI Usage: Claude (Fable 5) made all code edits, but I came up with the ideas and design for this feature myself."
    "AI Usage: discussion with Gemini and ChatGPT (5.5 Instant) to help find and understand the cause of this bug."
    "AI Usage: inline code suggestions from GitHub Copilot." -->

AI Usage:

- [ ] I take responsibility for all AI-generated content in my PR.
- [ ] I have tested all AI-generated content in my PR.

<!--
Thank you so much for your PR!  To help us review your contribution, please
consider the following points:

**PR Etiquette Reminders**
- This PR should be listed as a draft PR until you are ready for it to be reviewed

- After making changes in accordance with any reviews, re-request reviews from the same reviewers

- Do *not* mark conversations as resolved if you didn't start them

- Do mark conversations as resolved *if you opened them* and are satisfied with the changes/discussion.
-->
