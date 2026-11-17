import numpy as np
import pytest
import xarray as xr


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace):
    # Docstrings copied from xarray (where, fillna) assume np and xr exist.
    doctest_namespace["np"] = np
    doctest_namespace["xr"] = xr
