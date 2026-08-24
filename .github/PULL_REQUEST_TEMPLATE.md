<!--  Thank you for opening a PR! To help us review your contribution, please follow instructions below
      while filling out this form, and read the etiquette reminders at the bottom. -->

<!--  Please ensure the PR title summarizes the changes, e.g. "Adds `Grid._build_face_dimension` function".
      Avoid non-descriptive titles such as "Addresses issue #229". -->

Closes #XXX
<!--  Replace XXX with the issue number resolved by this PR, if this PR fully resolves an issue.
      If it resolves multiple issues, repeat "closes" for each, like "Closes #XXX, closes #YYY."
      If it does not fully resolve any issues, replace with something like "Related to #XXX",
          or "Fixes part of #YYY but does not fully close it." -->

## Overview
<!--  Please summarize the changes in this PR. How does it solve the original issue? -->


<!--  Does the scope of this PR do anything aside from just solving the original issue?
      And/or, does it not fully solve the issue as originally reported? If so, please clarify. -->


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
<!-- Please mark each item as [X] when completed, or [N/A] if not applicable to this PR. -->

**General**
- [ ] An issue is created and linked
- [ ] Added appropriate labels (if your uxarray repo permissions allow it)
- [ ] Filled out Overview and Expected Usage (if applicable) sections

**Testing & Benchmarking**
- [ ] There is adequate test coverage of changes from this PR (add new tests if needed)
- [ ] If this PR could affect performance, ran ASV benchmarks and confirmed they show expected behavior (add a new benchmark if necessary)
<!-- Adding the run-benchmark label (if your uxarray repo permissions allow it) will run ASV benchmarks.
    If you need benchmarks to be run but don't have permissions, leave this item unchecked for now. -->

**Documentation**
- [ ] Docstrings updated with any function changes, and included in all new functions
- [ ] User (public) functions added to `docs/api.rst`; internal (private) function names start with an underscore (`_`)

**Examples**
- [ ] If touched any notebook examples, cleared the output of all cells before committing
- [ ] If added new notebook examples, put in appropriate folder(s) and referenced in appropriate docs files
<!-- Appropriate folders for reference: (gallery: `docs/examples/`; guide: `docs/user-guide/`; quickstart: `docs/getting-started/`)
    Appropriate .rst files for reference: (gallery: `docs/gallery.rst`; guide: `docs/userguide.rst`; quickstart: `docs/quickstart.rst`)
    For gallery examples also add reference in `docs/gallery.yml` and thumbnail photo in `docs/_static/thumbnails/`-->


## AI Disclosure
<!-- Please specify all AI tools used. Optionally, include model and/or briefly describe usage. Examples:
    "AI Usage: Claude (Fable 5), Gemini"
    "AI Usage: ChatGPT (5.5 Instant) to help understand cause of this bug, but I wrote all updates myself."
    "AI Usage: just GitHub Copilot's inline code suggestions."
    If you did not use AI, please write "AI Usage: N/A" and remove these checklist items or mark as [N/A].-->

AI Usage:

- [ ] I have tested and take responsibility for all AI-generated content in my PR.

<!-- **PR Etiquette Reminders**
- Please list as "draft PR" until ready for review.
- Please do NOT mark any review comment threads as resolved.
- Instead, please notify reviewers after addressing their comments, via @username or the "re-request review" button.
- (Reviewers: please mark your own threads as resolved after confirming your comments have been addressed.)
-->
