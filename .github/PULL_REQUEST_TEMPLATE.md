<!--  The PR title should summarize the changes, for example "Add `Grid._build_face_dimension` function".
      Avoid non-descriptive titles such as "Addresses issue #229". -->

<!--  Replace XXX with the number of the issue that this PR will resolve. If this PR closed more than one,
      you may add a comma separated sequence-->
Closes #XXX

## Overview
<!--  Please provide a few bullet points summarizing the changes in this PR. This should include
      points on any bug fixes, new functions, or other changes that have been made. -->

## Expected Usage
<!--  If you are adding a feature into the Internal API, please produce a short example of it in action.
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
- [ ] An issue is linked created and linked
- [ ] Add appropriate labels
- [ ] Filled out Overview and Expected Usage (if applicable) sections

**Testing**
- [ ] Adequate tests are created if there is new functionality
- [ ] Tests cover all possible logical paths in your function
- [ ] Tests are not too basic (such as simply calling a function and nothing else)

**Documentation**
- [ ] Docstrings have been added to all new functions
- [ ] Docstrings have updated with any function changes
- [ ] Internal functions have a preceding underscore (`_`) and have been added to `docs/internal_api/index.rst`
- [ ] User functions have been added to `docs/user_api/index.rst`

**Examples**
- [ ] Any new notebook examples added to `docs/examples/` folder
- [ ] Clear the output of all cells before committing
- [ ] New notebook files added to `docs/examples.rst` toctree
- [ ] New notebook files added to new entry in `docs/gallery.yml` with appropriate thumbnail photo in `docs/_static/thumbnails/`

<!--
Thank you so much for your PR!  To help us review your contribution, please
consider the following points:

**PR Etiquette Reminders**
- This PR should be listed as a draft PR until you are ready to request reviewers

- After making changes in accordance with the reviews, re-request your reviewers

- Do *not* mark conversations as resolved if you didn't start them

- Do mark conversations as resolved *if you opened them* and are satisfied with the changes/discussion.
-->
