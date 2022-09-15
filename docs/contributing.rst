.. currentmodule:: uxarray

.. _contributing:

*******************
Contributor's Guide
*******************

Welcome to the Contributor's Guide for UXarray!

Welcome to the team! If you are reading this document, we hope that
you are already or soon-to-be a UXarray contributor, please keep reading!

1. Overview
===========

UXarray is a community-owned, open-development effort that is a result
of the collaboration between `Project Raijin <https://raijin.ucar.edu/>`_
and DOE’s `SEATS Project <https://seatstandards.org/>`_ ]. Even though the
UXarray team has initiated and been expanding this package, outside
contributions are welcome and vital for its long-term sustainability.
Therefore, we invite other community members to become part of this
collaboration at any level of contribution.

1.1. Why the name "UXarray"?
----------------------------

We have created UXarray by composing `Xarray <https://docs.xarray.dev/en/stable/>`_,
a Pangeo ecosystem package commonly-used  for structured grids
recognition, to support reading and recognizing unstructured grid
model outputs. We picked the name "UXarray" and preferred to
capitalize the first two letters to emphasize it is Xarray for
Unstructured grids.

1.2. Many ways to contribute
----------------------------

There are many different ways to contribute to UXarray. Anyone can, for
example,

* Write or revise documentation (including this document)
* Implement data analysis operators for unstructured grids from scratch
  or from their workflows
* Develop example notebooks that demonstrate how a particular function
  is used
* Answer a support question
* Simply request a feature or report a bug

All of these activities are signicant contributions to the on-going
development and maintenance of UXarray.

1.3. About this guide
---------------------

The UXarray content is hosted on GitHub, and through this document,
we aim to ease the community's experience with contributing to this project.
However, this guide might still be missing case-specific details; please do
not hesitate to reach out to us to consult any such cases.

.. note::
    Much of the information in this guide has been co-opted from the
    `GeoCAT <https://geocat.ucar.edu/pages/contributing.html>`_ project and
    `Project Pythia <https://projectpythia.org/contributing.html>`_.

1.4. Project Specific Resources
-------------------------------

Some important UXarray resources are as follows:

* `UXarray GitHub repository <https://github.com/UXARRAY/uxarray>`_ houses
  the open-source code base along with some significant documentation such
  as `Readme <https://github.com/UXARRAY/uxarray/blob/main/README.md>`_ and
  `UXarray draft API <https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_.

* `UXarray documentation <https://uxarray.readthedocs.io/en/latest/?badge=latest>`_
  houses significant documentation such as
  `Getting Started <https://uxarray.readthedocs.io/en/latest/quickstart.html>`_,
  `Installation <https://uxarray.readthedocs.io/en/latest/installation.html>`_,
  `Contributor's Guide <https://uxarray.readthedocs.io/en/latest/contributing.html>`_
  (i.e. this document), `Usage Examples <https://uxarray.readthedocs.io/en/latest/examples.html>`_,
  `Tutorials <https://uxarray.readthedocs.io/en/latest/tutorials.html#>`_, and
  `API Reference <https://uxarray.readthedocs.io/en/latest/api.html>`_.

* `UXarray draft API <https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
  tentatively shows the eventual list of UXarray functionalities. It is
  open to community feedback and inputs!

  - Please create a
    GitHub `Discussion <https://github.com/UXARRAY/uxarray/discussions>`_ or
    `Issue <https://github.com/UXARRAY/uxarray/issues>`_ if you feel any changes
    should be made to this document!

2. Configuring GitHub and Git, and Setting Up Python Environment
================================================================

In this section, we provide details about how to configure Github and Git
in order to work with the UXarray GitHub repository. This is because most
contributions, such as adding a new function or making changes to an
existing one, writing a usage example, or making some modifications to this
Contributor's Guide requires operating directly on the GitHub repository.

Contributing to a GitHub repository follows almost the same process by
any open development Python project maintained on GitHub.
However, it can still seem complex and somewhat varied from one project to
another. As such, we will provide an overview of GitHub here and refer the
reader to other more comprehensive resources for detailed information about
it (such as GitHub's this
`Getting Started with Github <https://docs.github.com/en/get-started>`_
guide), but we will provide great details about how to configure a GitHub
development environment for UXarray development and how to contribute.

2.1. Getting started with GitHub and Git
----------------------------------------

Contributing to UXarray requires using GitHub, as already mentioned, and
Git. The latter, Git, is an open source, command line tool for collaborative
software version control, while GitHub is an online, web-accessible service
that greatly simplifies using the powerful, yet often complex, Git.

.. note::
    GitHub operates entirely within a web browser. You do not need to
    install anything, but you will need to set up a free GitHub account. Git
    is a command line tool that most likely will need to be installed on
    your machine and run from a "terminal" window, AKA a "shell".

Using, and even just configuring, Git and GitHub are often the most
daunting aspects of contributing to a GitHub hosted project. Here are
the basic steps for GitHub/Git configuration and Python installation, all
of which must be performed **before** contributing to UXarray.

2.1.1. GitHub Setup
^^^^^^^^^^^^^^^^^^^

Create a free `GitHub <https://github.com/>`_ account. Note GitHub
offers free personal use and paid enterprise accounts. The free account
is all that is needed.

2.1.2. Git Setup
^^^^^^^^^^^^^^^^

If not already installed on your machine, download and install the latest
version of `Git <https://git-scm.com/downloads>`_. Once Git is installed
you will need to open a terminal/shell and type the commands below to
configure Git with a user name and your email. Note, it is recommended
that you use the same user name/email as you did when setting up your GitHub
account, though technically this may not be necessary::

    $ git config --global user.name "Your name here"
    $ git config --global user.email "your_email@example.com"

Don’t type the $. This simply indicates the command line prompt.

Configure your environment to authenticate with GitHub from Git. This is a
complicated process, so suggest that you refer to the details on the `GitHub
site <https://docs.github.com/en/get-started/quickstart/set-up-git#next-steps-authenticating-with-github-from-git>`_
to complete this step.

For further reading see the `Getting Started with Github
<https://docs.github.com/en/get-started>`_.

2.2. Setting Up Python Environment
----------------------------------

Before starting any Python or documentation development, you’ll need to
create a Python environment along with a package manager. When we use the
term “Python environment” here, we are referring to the Python programming
language. Because there are so many Python packages available,
maintaining interoperability between them is a huge challenge. To overcome
some of these difficulties, we strongly recommend the use of `Anaconda
<https://www.anaconda.com/download/>`_ or `Miniconda
<https://conda.io/miniconda.html/>`_ as a package manager to manage your
Python ecosystem. These package managers allow you to create a separate,
custom Python environment for each specific Python set of tools. Yes, this
unfortunately results in multiple copies of Python on your system, but it
greatly reduces breaking toolchains whenever a change is made to your Python
environment (and is more reliable than any other solution we’ve encountered).
Also, the use of Anaconda/Miniconda is standard practice for working with
Python in general, not simply for using UXarray.

To configure your Python environment:

1. Install either `Anaconda <https://www.anaconda.com/download/>`_ or
`Miniconda <https://conda.io/miniconda.html/>`_.

   * We recommend Miniconda as it does not install unnecessarily many
     packages that will not be needed for UXarray development.

2. Make sure your conda is up to date by running this command from the
terminal::

    conda update conda

At this point you will have a current version of conda available on your
system. Before using your conda environment to work on UXarray development,
you’ll need to perform an additional setup that is going to be described in
the next section.

3. How UXarray Uses Git/GitHub
==============================

UXarray uses the `GitHub Flow
<https://docs.github.com/en/get-started/quickstart/github-flow>`_ model
for its workflow. UXArray also uses an automated formatter on all commits
in local development environment. This changes the normal workflow slightly,
so in order to avoid any confusions, follow these steps:

3.1. Select an issue to work on
-------------------------------

Virtually any work should be addressing an issue. There are a few options to
select an issue to work on:

1. First, check the existing `UXarray issues
<https://github.com/UXARRAY/uxarray/issues>`_. These issues might have been
created from either the

* `UXarray draft API
  <https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
  entries,
* An entry in a `UXarray Discussion <https://github.com/UXARRAY/uxarray/discussions>`_
* A feature request, bug report, or any finding of the developers or users
  as a work to-do.

2. If you are going to work on one of those issues above, self-assign that
issue before you start working.

3. Or else, if you are going to work on something else, create an issue. The
title, description, and label of an issue could be very helpful for the team to
triage the issue and move forward with the work needed.

* Title needs to reflect the purpose of the issue. Some examples:
  - Start with "Add ..." if it is a new functionality needed
  - Start with "Error ...", "Exception ...", or "Bug ..." if that is a bug
* Use issue labels accordingly, e.g. "bug", "support", "enhancement",
  "documentation", etc.
* Clearly describe the issue (e.g. what the work should be, which parts of
  UXarray it would need to change, if known, etc.)

.. note::
    The work that addresses an issue will, most of the time, be eventually
    turned into a "pull request" (or maybe multiple in some cases) that lets you
    tell others about changes you're aiming to make on the project. Once a pull
    request (PR) is opened, it will require others' thorough review and discussion
    before your changes can take place in the project. Hence, a rule of thumb about
    pull requests should be that they should target logically connected, as atomic
    as possible work.

.. note::
    If you need any clarification/discussion about any requirements, or if you
    think the implementation of that issue will require significant changes to
    the code, design, documentation, etc. that issue itself is the right place
    to manage such discussions with the other UXarray'ers. Don't hesitate to
    ask ad-hoc meetings, etc.

    Do not forget, early requirements analysis, specifications, and design
    discussions can avoid redundat code review and modifications later!

3.2. Fork or locally clone the UXarray repository
-------------------------------------------------

Let us first make a decision of whether to fork or locally clone:

3.2.1. Should you fork?
^^^^^^^^^^^^^^^^^^^^^^^

"Fork"ing a repository as described below in this section creates a copy of
the repository under your own account or any GitHub organization of
which you are a member on GitHub. Forking can be advisable for cases such as:

* You want to safely make changes to the forked repository contents without changing
  the actual UXarray repository. This is because any changes you make to your
  fork will only be seen by you until you are ready to share them with others,
  and hopefully “merge” your changes into UXarray.
* You are not a regular contributor to the UXarray repository; thus, from time
  to time, you would sync your fork to UXarray and make one or more contributions.
* You are planning to initiate a new project by altering the forked repo
  significantly from UXarray, but this case is out of this guide's scope.

You can refer to the `Atlassian's Forking workflow
<https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow>`_
tutorial to further learn about forking.

If you decide on forking the UXarray repository, here is how to do that:

**Forking the UXarray repository**

Refer to GitHub's `Forking a repository
<https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_ guide and apply
instructions regarding the UXarray repository.

.. note::
    The above GitHub guide about forking a repository will also walk you through
    cloning your forked repository into your local work environment. Please do not
    forget to follow those instructions as well. Or, refer to `GitHub's Cloning a
    repository <https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`_
    guide, and apply those instructions for the UXarray fork that is listed under
    your GitHub account or organization.

After these steps, you will have two copies of the forked UXarray repo, one remote
and one local.

3.2.2. Should you locally clone instead?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In contrast to above cases that might be better suitable for fork, a regular
UXarray contributor who is comfortable with working on their local clone
of the actual UXarray repository and making their changes immediately viewable by
the other UXarray'ers (i.e. after "push"ing their "commit"s) can choose to locally
clone the UXarray repository.

**Locally cloning the UXarray repository**

Refer to `GitHub's Cloning a repository
<https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`_
guide and apply those instructions for the actual `UXarray repository
<https://github.com/UXARRAY/uxarray>`_.

.. note::
    Regardless of you either fork or clone, there will be a local directory created
    in the name "uxarray" (unless you specified a different name at the step with the
    :code:`git clone` command). You can type the following command in the terminal/shell to
    go into your local UXarray repository::

        $ cd uxarray

3.3. Configure UXarray Conda Environment
----------------------------------------

In your local UXarray directory, creating a UXarray conda environment is needed for
development purposes. USe the following commands for this::

    $ conda env create --file ci/environment.yml
    $ conda activate uxarray_build

THe above commands will use the :code:`environment.yml` conda environment definition
file that is housed in the :code:`ci` folder and create a conda environment with the
name "uxarray_build". Once you activate that environment with the help of the second
command, you will be able to develop UXarray codes in your local configuration.

3.3. Use Feature Branches
-------------------------

In your local clone, make a new feature branch off of the :code:`main` branch.
Naming this branch, whenever applicable, like the following is not required but may be much
helpful for tracking purposes: "issue_XXX" where XXX is the number of the issue or something
that is representative of the changes you’re making.
