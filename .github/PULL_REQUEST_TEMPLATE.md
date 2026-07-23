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
      You may ignore this step if it is not applicable (comment out this section). -->
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
<!-- Please mark any checkboxes that do not apply to this PR as [N/A]. If an entire section doesn't
apply to this PR, comment it out or delete it. -->

**General**
- [ ] An issue is created and linked
- [ ] Added appropriate labels (if you have label edit permissions)
- [ ] Filled out Overview and Expected Usage (if applicable) sections

**Testing**
- [ ] Adequate tests are created if there is new functionality
- [ ] Tests cover all possible logical paths in your function
- [ ] Tests are not too basic (such as simply calling a function and nothing else)

**Documentation**
- [ ] Docstrings have been added to all new functions
- [ ] Docstrings have been updated with any function changes
- [ ] User (public) functions have been added to `docs/api.rst`
- [ ] Internal (private) function names start with an underscore (`_`)


**Examples**
- [ ] New notebook examples cleared the output of all cells before committing
- [ ] New notebook examples added to appropriate folder (gallery: `docs/examples/`; guide: `docs/user-guide/`; quickstart: `docs/getting-started/`)
- [ ] New notebook examples referenced in appropriate .rst file (gallery: `docs/gallery.rst`; guide: `docs/userguide.rst`; quickstart: `docs/quickstart.rst`)
- [ ] New notebook gallery examples added entry in `docs/gallery.yml` with appropriate thumbnail photo in `docs/_static/thumbnails/`

<!--
Thank you so much for your PR!  To help us review your contribution, please
consider the following points:

**PR Etiquette Reminders**
- This PR should be listed as a draft PR until you are ready for it to be reviewed

- After making changes in accordance with any reviews, re-request reviews from the same reviewers

- Do *not* mark conversations as resolved if you didn't start them

- Do mark conversations as resolved *if you opened them* and are satisfied with the changes/discussion.
-->
