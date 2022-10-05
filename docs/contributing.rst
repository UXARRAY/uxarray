.. currentmodule:: uxarray

.. _contributing:

*******************
Contributor's Guide
*******************

Welcome to the Contributor's Guide for UXarray!

Welcome to the team! If you are reading this document, we hope that
you are already or soon-to-be a UXarray contributor, please keep reading!

.. note::
    If you haven't done so yet, you may want to give a quick read through
    our `README <https://github.com/UXARRAY/uxarray/blob/main/README.md>`_
    as it provides a lot of significant information about UXarray.

1. Overview
===========

UXarray is a community-owned, open-development effort that is a result
of the collaboration between NSF's `Project Raijin <https://raijin.ucar.edu/>`_
and DOE’s `SEATS Project <https://seatstandards.org/>`_. Even though the
UXarray team has initiated and been expanding this package, outside
contributions are welcome and vital for its long-term sustainability.
Therefore, we invite other community members to become part of this
collaboration at any level of contribution.

1.1. Many Ways to Contribute
----------------------------

There are many different ways to contribute to UXarray. Anyone can, for
example,

* Write or revise documentation (including this document)
* Implement data analysis operators for unstructured grids from scratch
  or from their workflows
* Develop example notebooks that demonstrate how a particular function
  is used
* Answer a support question
* Request a feature or report a bug

All of these activities are signicant contributions to the on-going
development and maintenance of UXarray.

1.2. About This Guide
---------------------

The UXarray content is hosted on GitHub, and through this document,
we aim to ease the community's experience with contributing to this project.
However, this guide might still be missing case-specific details; please do
not hesitate to reach out to us to consult any such cases.

.. note::
    Much of the information in this guide has been co-opted from the
    `GeoCAT <https://geocat.ucar.edu/pages/contributing.html>`_ project and
    `Project Pythia <https://projectpythia.org/contributing.html>`_.

1.3. Project-specific Resources
-------------------------------

Some important UXarray resources are as follows:

* `UXarray GitHub repository <https://github.com/UXARRAY/uxarray>`_ houses
  the open-source code base along with some significant documentation such
  as the `README <https://github.com/UXARRAY/uxarray/blob/main/README.md>`_ and
  `UXarray draft API <https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_.

* `UXarray documentation <https://uxarray.readthedocs.io/>`_
  houses significant documentation such as :ref:`quickstart`, :ref:`installation`,
  :ref:`contributing` (i.e. this document), :ref:`examples`, :ref:`tutorials`,
  and :ref:`api`.

* `UXarray draft API <https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
  tentatively shows the eventual list of UXarray functionalities. It is
  open to community feedback and inputs!

  - Please create a
    GitHub `Discussion <https://github.com/UXARRAY/uxarray/discussions>`_ or
    `Issue <https://github.com/UXARRAY/uxarray/issues>`_ if you feel any changes
    should be made to this document!

2. Configuring GitHub & Git, and Setting Up Python Environment
==============================================================

In this section, we will detail what is needed to be done before starting to
contribute UXarray.

2.1. Getting Started with GitHub and Git
----------------------------------------

Contributing to UXarray requires using GitHub, and contributing to a GitHub
repository follows almost the same process by any open source Python project
maintained on GitHub. However, it can still seem complex and somewhat varied from
one project to another. As such, we will refer the reader to comprehensive
resources for basic learning and detailed information about GitHub (such as the
`Getting Started with Github <https://docs.github.com/en/get-started>`_ guide).

.. note::
    To emphasize the details of how UXarray uses GitHub and how to contribute to
    it, we will provide significant details in the
    `3. How UXarray Uses Git/GitHub`_ section.

Git is an open source, command line tool for collaborative software version
control, while GitHub is an online, web-accessible service that greatly simplifies
using the powerful, yet often complex, Git. Just like GitHub, we believe that the
basics of Git is out of scope of this guide, so we will refer the reader to
Git-specific guides for that purpose (e.g. GitHub's `Set up Git
<https://docs.github.com/en/get-started/quickstart/set-up-git>`_ guide and `Git's
homepage <https://git-scm.com/>`_).

.. note::
    Git has lots and lots of commands, each with lots and lots of options. Even if we
    can cover some of them throughout this guide, your best friend for figuring out
    to do things with Git may be Google, and in particular
    `StackOverflow <https://stackoverflow.com/>`_.

Configure your environment to authenticate with GitHub from Git. This is a
complicated process, so we suggest that you refer to the details in the
`Authenticating with GitHub from Git
<https://docs.github.com/en/get-started/quickstart/set-up-git#authenticating-with-github-from-git>`_
guide to complete this step.

.. attention::
    The basic steps for GitHub/Git configuration, which can be learned from the
    linked guides here, need to be performed **before** contributing to UXarray!

2.2. Python Environment Setup
-----------------------------

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

.. tip::
    We recommend Miniconda as it does not unnecessarily install many
    packages that will not be needed for UXarray development.

2. Make sure your conda is up to date by running this command from the
terminal::

    $ conda update conda

.. warning::
    Don’t type the $ character. This simply indicates the command line prompt.

At this point, you will have a current version of conda available on your
system. Before using your Python environment to work on UXarray development,
you’ll need to perform additional steps that are going to be described in
the next section.

3. How UXarray Uses Git/GitHub
==============================

UXarray uses the `GitHub Flow
<https://docs.github.com/en/get-started/quickstart/github-flow>`_ model
for its workflow. UXarray also uses an automated formatter, which is
described in `3.4. Install and Setup Pre-commit Hooks`_ in detail, on
all commits in local development environment in order to ensure code
formatting. This changes the normal workflow slightly, so in order to
avoid any confusions, follow these steps:

3.1. Select An Issue to Work on
-------------------------------

Virtually any work should be addressing an issue. There are a few options to
select an issue to work on:

1. First, check the existing `UXarray issues
<https://github.com/UXARRAY/uxarray/issues>`_. These issues might have been
created from either the:

* `UXarray draft API
  <https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_ entries,
* An entry in a `UXarray Discussion <https://github.com/UXARRAY/uxarray/discussions>`_
* A feature request, bug report, or any finding of the developers or users
  as a work to-do.

2. If you are going to work on one of those issues above, self-assign that
issue before you start working.

3. Or else, if you are going to work on something else, create an issue and
assign yourself. The title, description, and label of an issue could be very
helpful for the team to triage the issue and move forward with the work
needed.

* Title needs to reflect the purpose of the issue. Some examples:

  - Start with "Add ..." if it is a new functionality needed
  - Start with "Error ...", "Exception ...", or "Bug ..." if that is a bug

* Use issue labels accordingly, e.g. "bug", "support", "enhancement",
  "documentation", etc.

* Clearly describe the issue (e.g. what the work should be, which parts of
  UXarray it would need to change, if known, etc.)

.. caution::
    The work that addresses an issue will, most of the time, be eventually
    turned into a "pull request" (or maybe multiple in some cases) that lets you
    tell others about changes you're aiming to make on the project. Once a pull
    request (PR) is opened, it will require others' thorough review and discussion
    before your changes can take place in the project. Hence, a rule of thumb about
    pull requests should be that they should target logically connected, as atomic
    as possible, work.

.. note::
    If you need any clarification/discussion about any requirements, or if you
    think the implementation of that issue will require significant changes to
    the code, design, documentation, etc. that issue itself is the right place
    to manage such discussions with the other UXarray-ers. Don't hesitate to
    ask ad-hoc meetings, etc.

    Do not forget, early requirements analysis, specifications, and design
    discussions can avoid redundat code review and modifications later!

3.2. Fork or Locally Clone The UXarray Repository
-------------------------------------------------

Let us first make a decision of whether to fork or locally clone:

3.2.1. Should You Fork?
^^^^^^^^^^^^^^^^^^^^^^^

Forking a repository as described below in this section creates a copy of
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

.. hint::
    The above GitHub guide about forking a repository will also walk you through
    cloning your forked repository into your local work environment. Please do not
    forget to follow those instructions as well. Or, refer to GitHub's `Cloning a
    repository <https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`_
    guide, and apply those instructions for the UXarray fork that is listed under
    your GitHub account or organization.

After these steps, you will have two copies of the forked UXarray repo, one remote
and one local.

3.2.2. Should You Locally Clone Instead?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In contrast to above cases that might be better suitable for fork, a regular
UXarray contributor who is comfortable with working on their local clone
of the actual UXarray repository and making their changes immediately viewable by
the other UXarray-ers (i.e. after pushing their commits) can choose to locally
clone the UXarray repository.

**Locally cloning the UXarray repository**

Refer to GitHub's `Cloning a repository
<https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`_
guide and apply those instructions for the actual `UXarray repository
<https://github.com/UXARRAY/uxarray>`_.

.. hint::
    Regardless of whether you fork or clone, there will be a local directory created
    in the name "uxarray" (unless you specified a different name at the step with the
    :code:`git clone` command). You can type the following command in the terminal/shell
    to go into your local UXarray repository::

        $ cd uxarray

3.3. Configure UXarray Conda Environment
----------------------------------------

In your local UXarray directory, creating a UXarray conda environment is needed for
development purposes. Use the following commands for this::

    $ conda env create --file ci/environment.yml
    $ conda activate uxarray_build

THe above commands will use the ``environment.yml`` conda environment definition
file that is hosted under the ``ci`` folder and create a conda environment with
the name ``uxarray_build``. Once you activate that environment with the help of the
second command, you will be able to develop UXarray codes in your local configuration.

3.4. Install and Setup Pre-commit Hooks
---------------------------------------

Pre-commit hooks are useful for identifying simple issues with the code format before
submission to code review (i.e. in your local before commits are processed). You specify
a list of hooks you want (this list has already been specified for UXarray in the
`.pre-commit-config.yaml <https://github.com/UXARRAY/uxarray/blob/main/.pre-commit-config.yaml>`_
file, so there is no action needed), and install pre-commit as described below. Pre-commit
manages the installation and execution of the hooks specified.

Hooks are run on every commit to automatically point out issues in code formatting such
as the number of characters in each line, missing semicolons, trailing whitespace, debug
statements, etc. By pointing these issues out before code review, this allows a code
reviewer to focus on the architecture of a change while not wasting time with trivial
style nitpicks. Refer to the `Pre-commit documentation <https://pre-commit.com/#intro>`_
for further learning.

.. attention::
    We also use pre-commit GitHub Actions workflow to make sure every code contribution
    to our UXarray's GitHub repository (i.e. pull request) aligns with our code standards.
    Therefore, the code changes that are not being checked through local `pre-commit
    hooks` will eventually be tested against our workflow in the GitHub server as
    described in `3.7.4.2. GitHub Actions checks`_. If there are any issues with the
    code format, it will lead the pull request to fail the pre-commit checks.

3.4.1. Pre-commit Setup
^^^^^^^^^^^^^^^^^^^^^^^

If you configured your Conda environment via the instructions in `3.3. Configure UXarray
Conda Environment`_, you will have the ``pre-commit`` package already installed in your
environment (Otherwise, you will need to run :code:`conda install -c conda-forge pre_commit`
to get it installed into your Conda environment for UXarray development). The only thing you
will have to do additionally in order to set up pre-commit is to run the following command
in your terminal in the UXarray root directory::

    $ pre-commit install

At this step, you are good to go with your pre-commit hooks to check your future commits.
If, at any time, you'd like to run pre-commit hooks on your all files, you can run the
following command::

    $ pre-commit run --all-files

3.5. Use Feature Branches
-------------------------

In your local clone, make a new branch off of the ``main`` branch (this is the way to go
most of the time, but there might be specific cases where, for example, a branch is needed
to be created off of another feature branch). Naming this branch, whenever applicable,
like the following is not required but may be helpful for tracking purposes:
"issue_XXX" where XXX is the number of the issue or something that is representative of
the changes you’re making, e.g. "do-this-work", "add-that-function", etc.

Here are example commands that assume, you are checking out the ``main`` branch first,
pulling  from the remote server to have everything in your local up-to-date, and creating a new
branch off of ``main``::

    $ git checkout main
    $ git pull
    $ git checkout -b <new_branch>

Once you create the new branch, you are good to go with your local changes in the UXarray
directory!

3.6. Local Changes, Commits, and Pushes
---------------------------------------

The local development process can very basically be itemized as follows:

1. Make changes to your local copy of the UXarray repository

   .. attention::
        Please refer to `3.7.3. Common Elements of Pull Requests`_ to make sure
        your local changes have all of the elements they should cover.

2. Add your local changes to the "staged" changes for them to be included in the commit.

   * You can see any uncommitted changes you’ve made to your local copy of the repository by
     running the following command from anywhere (any directory) within the directory where
     you ran :code:`git checkout`::

        $ git status

   * So, add changed file(s) into the staged changes to be included in a single commit::

        $ git add PATH/TO/NEW/FILE

     where ``PATH/TO/NEW/FILE`` is the path name of the newly created file.

3. Now, commit the staged change(s)::

        $ git commit -m "Descriptive comment about what this commit does"

   * Limiting the commit to file(s) changed for an atomic task would be very much
     helpful in cases you need to review and maybe revert commits.

   * Do not forget that ``pre-commit hooks`` will check your code changes at this
     point. If some of those hooks fail, you will need to add those files again
     and attempt to make the same commit again as the hooks will have changed those
     file(s) to make them comply with the code formatting rules.

   * A good practice is to run::

        $ git status

     after your commit to verify everything looks as expected.

4. Push your commit(s) into the remote repository::

    $ git push

.. attention::
    Remember that we are still not proposing to merge our work into the UXarray
    GitHub repository, which will be described in the next subsection, `3.7. Pull
    Requests`_.

3.7. Pull Requests
------------------

Pull requests are GitHub's way for developers to propose and collaborate on changes
to a repository. If you are interested in learning the foundations about pull requests,
please refer to GitHub's `Pull requests <https://docs.github.com/en/pull-requests>`_
documentation.

Once you have completed making changes to your local copy of the UXarray repository,
pushed them all, and are ready to have your changes merged into the repository on GitHub,
you need to submit a PR asking the UXarray maintainers to consider your merge request.

The merge will occur between your personal branch that should have all of your commits
from local in the GitHub repository (either in your fork or in the actual UXarray repo)
and the ``main`` branch, if not other, in the UXarray's GitHub repository. There
might be some exceptions to this generic case, which can always be learned through
reading or discussed with the maintainers and community.

.. note::
    While pull requests are supposed to be based on changes that are ready to be tested,
    reviewed, and eventually be merged to repositories; there is an exception to this:
    Draft Pull Requests. We encourage Draft PRs if you want to just start a
    conversation about your code that isn’t in any state to be judged, and get feedback
    as well as some guidance from others. Please read this GitHub `blog
    <https://github.blog/2019-02-14-introducing-draft-pull-requests/>`_ about draft PRs.

Please refer to Github's `Creating a pull request
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_
guide for the instructions. We are especially avoiding to detail such instructions
here in this guide as there are many references to GitHub's graphical user interface,
which might be changed in the future.

.. attention::
    Below are significant things about pull requests that can be very helpful throughout
    the entire contribution process (i.e. review and merge) process when performed:

3.7.1. Link PR to Issue
^^^^^^^^^^^^^^^^^^^^^^^

Don't forget to link your PR to the issue you have been working on! This will help not
only keep track of issues easier but also get them closed automatically once your PR is
merged. You can refer to GitHub's `Linking a pull request to an issue
<https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue>`_
for this.

3.7.2. Review Your Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you finalize opening the actual PR, it is a good practice to review the changes
that you’ve made. You should be able to review all of the changes that will go into
this PR just before you press the `Create pull request` button.

If there are any changes you want to make in the PR, you can delay creating the PR
and push new commits, revert existing changes, etc. You can then create the PR.

3.7.3. Common Elements of Pull Requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Despite the fact that pull requests can differ regarding their purposes (e.g.
correction of a single typo in only one file could make a PR), most of the PRs may
consist of code changes that should, most of the time, include some other elements
with it. Having all of such elements addressed in a PR could make the review and
merge process a lot easier. Assuming such code changes, these are what might
accompany them:

3.7.3.1. Unit tests
~~~~~~~~~~~~~~~~~~~

Virtually all new UXarray code needs to include unit tests of the functionality
implemented. The UXarray project makes use of diverse technologies for unit testing
as follows:

* All the unit tests of every single Python module (i.e. `.py` file) should be
  implemented as a separate test script under the ``\test`` folder of the
  repository's root directory.

* The `pytest <https://docs.pytest.org/en/stable/contents.html>`_ testing framework
  is used as `runner` for the tests. If you configured your Conda environment via
  the instructions in `3.3. Configure UXarray Conda Environment`_, you will have
  the ``pytest`` package already installed in your environment (Otherwise, you
  will need to run :code:`conda install -c conda-forge pytest` to get it installed
  into your Conda environment for UXarray development).

* Test scripts themselves are not intended to use ``pytest`` through implementation.
Instead, ``pytest`` should be used only for running test scripts as follows::

    $ pytest test/<test_script_name>.py

, or::

    $ python -m pytest test/<test_script_name>.py

, the latter of which will also add the current directory to ``sys.path``.

Not using ``pytest`` for implementation allows the unit tests to be also run
by using (a number of benefits/conveniences coming from using ``pytest`` can be
seen `here <https://docs.pytest.org/en/7.1.x/how-to/unittest.html#how-to-use-unittest-based-tests-with-pytest>`_
though)::

    $ python -m unittest test/<test_script_name>.py

Also, all of the test scripts can be run at once with the following command::

    $ pytest test

* Python's unit testing framework, `unittest
  <https://docs.python.org/3/library/unittest.html>`_ is used for implementation of
  the test scripts.

* Reference results (i.e. expected output or ground truth for not all but the most cases)
should not be magic values (i.e. they need to be justified and/or documented).

* Recommended, but not mandatory, implementation approach is as follows:

  - Common data structures, variables and functions,  as well as
    expected outputs, which could be used by multiple test methods throughout
    the test script, are defined either under a base test class or in the very
    beginning of the test script for being used by multiple unit test cases.

  - Any group of testing functions dedicated to testing a particular
    phenomenon (e.g. a specific edge case, data structure, etc.) is
    implemented by a class, which inherits ``TestCase`` from Python's
    ``unittest`` and likely the base test class implemented for the purpose
    mentioned above.

  - Assertions are used for testing various cases such as array comparison.

  - Please see previously implemented test cases for reference of the
    recommended testing approach

.. attention::
    Our test suite that includes all the unit tests is executed automatically
    for PRs with the help of GitHub Actions workflows to ensure new code passes
    tests. Hence, please check `3.7.4.2. GitHub Actions checks`_ to make sure your
    PR tests are all passing before asking others to review your work.

3.7.3.2. Docstrings
~~~~~~~~~~~~~~~~~~~

All Python functions must contain a `Google Style Python
<https://github.com/google/styleguide/blob/gh-pages/pyguide.md>`_ docstring (i.e.
triple quoted comment blocks). These docstrings are accessed from the Python interpreter
whenever the user types::

    help <FUNCTION_NAME>

They are also automatically converted into web-accessible documentation available from
the `UXarray documentation <https://uxarray.readthedocs.io/>`_.

The docstrings must contain:

1. A brief description of the functionality provided. What does this function do?

2. If available, references to the algorithm or implementation employed

3. A complete description of arguments and return values in Google format

   * Please refer to the existing functions that already have this

4. One or more very short usage examples that demonstrates how to invoke
   the function, and possibly what to expect it to return

   * No need for these examples to actually be executable

   * If a usage example is longer than a handful of lines, a more complete
     example may be created instead, referring to `3.7.3.4. Usage Examples`_.

3.7.3.3. Documentation
~~~~~~~~~~~~~~~~~~~~~~

As we mentioned a few times throughout this guide, UXarray has a static `documentation
<https://uxarray.readthedocs.io/>`_ :ref:`index` page that is being generated automatically from the
repository's code structure. However, there needs to be some manual additions to the
proper documentation index file(s) for the automation to work.

The index files ``docs/user_api/index.rst`` and ``docs/internal_api/index.rst`` (paths
relative from the root directory) are used for UXarray documentation to allow the
`User API <user_api/index.rst>`_ and `Internal API <internal_api/index.rst>`_,
respectively, to be automatically generated.

That being said, the code changes, which might be a new function implementation or some
modifications to existing ones, must be added to the appropriate ``index.rst``
file so that its documentation page is automatically generated.

.. caution::
    Please also ensure that you are changing the `UXarray draft API
    <https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
    accordingly if you are proposing changes to the API (e.g. new functions/attributes,
    modification(s) to existing functions, etc.)

3.7.3.5. Usage examples
~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    This may not be required for every PR and can be handled as separate PRs when
    needed. However, it would be a great practice to provide usage examples in the
    same PR, especially for demonstrating the use of complex UXarray functions.

The UXarray documentation houses ``examples/<example-name>.ipynb`` files (paths
relative from the root directory) to provide `Usage Examples <examples.rst>`_ to be
automatically generated. If you prefer to provide usage examples for the work you
have put together, please be sure to put your notebook(s) under this same directory.

3.7.4. After You Open The Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here are a few more tips that could be helpful throughout getting your PR merged:

3.7.4.1. Addressing reviews
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that you have opened a PR that has the necessary elements with it and got reviewers
assigned, you will hopefully begin to receive reviews soon from them. Please make sure
to address those review comments, change requests, etc to get your PR ready to be
merged.

.. attention::
    When you addressed all of a reviewer's comments/requests, do not forget to
    re-request their next review!


3.7.4.2. GitHub Actions checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UXarray employs a number of GitHub Actions workflows (please refer to the `GitHub
Actions <https://docs.github.com/en/actions>`_ guide for detailed information) to make
sure our PRs, branches, etc. pass certain test scenarios such as the pre-commit hooks,
code test suite, and documentation generation. The pre-commit hooks workflow ensures
the code being proposed to be merged is complying with code standards. The test suite
workflows ensure that the changes are passing for those tests described in `3.7.3.1.
Unit tests`_ for a matrix of various platforms and Python versions. The documentation
generation workflow ensures the changes proposed are still allowing our documentation
to be generated without any issues.

.. note::
    We require PRs to pass all of these checks before getting merged in order to
    always ensure our ``main`` branch stability.

These checks can be extremely helpful for contributors to be sure about they are
changing things in correct directions and their PRs are ready to be reviewed and
merged. For example, the ``docs/readthedocs.org:uxarray`` check can show whether
the UXarray documentation is able to be generated (i.e. pass/fail) and if it passes, it
also shows, just by clicking to its `Details`, the corresponding documentation being
generated for the PR.
