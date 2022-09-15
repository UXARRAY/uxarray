.. currentmodule:: uxarray

.. _contributing:


Contributor's Guide
===================

Welcome to the Contributor's Guide for UXarray!

So, welcome to the team! If you are reading this document, we hope that
you are already or soon-to-be a UXarray contributor! Please keep reading!

1. Overview
-----------

UXarray is a community-owned, open-development effort that is a result
of the collaboration between `Project Raijin <https://raijin.ucar.edu/>`_
and DOE’s `SEATS Project <https://seatstandards.org/>`_ ]. Even though the
Uxarray team has initiated and been expanding this package, outside
contributions are welcome and vital for its long-term sustainability.
Therefore, we invite other community members to become part of this
collaboration at any level of contribution.

1.1. Why the name "UXarray"?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have created UXarray by composing `Xarray <https://docs.xarray.dev/en/stable/>`_,
a Pangeo ecosystem package commonly-used  for structured grids
recognition, to support reading and recognizing unstructured grid
model outputs. We picked the name "UXarray" and preferred to
capitalize the first two letters to emphasize it is Xarray for
Unstructured grids.

1.2. Many ways to contribute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are many different ways to contribute to UXarray. Anyone can, for
example,

- Write or revise user documentation (including this document)

- Implement data analysis operators for unstructured grids from scratch
  or from their workflows

- Develop example notebooks that demonstrate how a particular function
  is used

- Answer a support question

- Simply request a feature or report a bug

All of these activities are signicant contributions to the on-going
development and maintenance of UXarray.

1.3. About this guide
^^^^^^^^^^^^^^^^^^^^^

The UXarray content is hosted on GitHub, and through this document,
we aim to ease the community's experience with contributing to this project.
However, this guide might still be missing case-specific details; please do
not hesitate to reach out to us to consult any such cases.

1.4. Project Specific Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some important UXarray resources are as follows:

- `Uxarray GitHub repository <https://github.com/UXARRAY/uxarray>`_ houses
  the open-source code base along with some significant documentation such
  as “Readme”, “Installation”, “Contributing”, and UXarray draft API.

- `UXarray draft API <https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
  tentatively shows the eventual list of UXarray functionalities. It is
  open to community feedback and inputs!

  - Please create a
    GitHub `Discussion <https://github.com/UXARRAY/uxarray/discussions>`_ or
    `Issue <https://github.com/UXARRAY/uxarray/issues>`_ if you feel there
    should be change in any way in this document!

2. Working on the UXarray GitHub Repository
-------------------------------------------

In this section, we provide details about how to work on the UXarray
GitHub repository. This is because most contributions, such as adding a new
function or making changes to an existing one, writing a usage example, or
making some modifications to this Contributor's Guide requires operating
directly on the GitHub repository.

Contributing to a GitHub repository follows almost the same
process used by many open development Python tools maintained on GitHub.
However, it can still seem complex and somewhat varied from one project to
another. As such, we will provide an overview of GitHub here and refer the
reader to other more comprehensive resources for detailed information about
it, but we will provide great details about how to configure a GitHub
development environment for UXarray development and how to contribute.

2.1. Getting started with GitHub and Git
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
