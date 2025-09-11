"""Pytest configuration and fixtures for uxarray tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def gridpath():
    """Get the path to test grid/mesh file(s).

    Parameters
    ----------
    *args : str
        Path components relative to test/meshfiles/
        The last argument can be a list of filenames to return multiple paths.

    Returns
    -------
    Path or list of Path
        Single path or list of paths to test grid files
    """
    base_path = Path(__file__).parent / "meshfiles"

    def _get_path(*args):
        # If the last argument is a list, handle multiple files
        if args and isinstance(args[-1], list):
            base_parts = args[:-1]
            filenames = args[-1]
            paths = []
            for filename in filenames:
                path = base_path.joinpath(*base_parts, filename)
                if not path.exists():
                    pytest.skip(f"Test grid file not found: {path}")
                paths.append(path)
            return paths
        else:
            # Single file case
            path = base_path.joinpath(*args)
            if not path.exists():
                raise FileNotFoundError(f"Test grid file not found: {path}")
            return path

    return _get_path


@pytest.fixture
def datasetpath():
    """Get the path to test dataset file(s).

    Parameters
    ----------
    *args : str
        Path components relative to test/meshfiles/
        The last argument can be a list of filenames to return multiple paths.

    Returns
    -------
    Path or list of Path
        Single path or list of paths to test dataset files
    """
    base_path = Path(__file__).parent / "meshfiles"

    def _get_path(*args):
        # If the last argument is a list, handle multiple files
        if args and isinstance(args[-1], list):
            base_parts = args[:-1]
            filenames = args[-1]
            paths = []
            for filename in filenames:
                path = base_path.joinpath(*base_parts, filename)
                if not path.exists():
                    pytest.skip(f"Test dataset file not found: {path}")
                paths.append(path)
            return paths
        else:
            # Single file case
            path = base_path.joinpath(*args)
            if not path.exists():
                raise FileNotFoundError(f"Test dataset file not found: {path}")
            return path

    return _get_path


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "meshfiles"


@pytest.fixture
def mesh_constants():
    """Test constants for mesh validation."""
    return {
        'NNODES_ov_RLL10deg_CSne4': 683,
        'NNODES_outCSne8': 386,
        'NNODES_outCSne30': 5402,
        'NNODES_outRLL1deg': 64442,
        'DATAVARS_outCSne30': 4,
        'TRI_AREA': 0.02216612469199045,
        'CORRECTED_TRI_AREA': 0.02244844510268421,
        'MESH30_AREA': 12.566,
        'PSI_INTG': 12.566,
        'VAR2_INTG': 12.566,
        'UNIT_SPHERE_AREA': 4 * np.pi,
        'FACE_VERTS_AREA': 0.515838,
    }
