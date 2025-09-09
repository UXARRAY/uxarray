"""Pytest configuration and fixtures for uxarray tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def datapath():
    """Get the path to a test data file.

    Parameters
    ----------
    *args : str
        Path components relative to test/meshfiles/

    Returns
    -------
    Path
        Full path to the test data file
    """
    base_path = Path(__file__).parent / "meshfiles"

    def _get_path(*args):
        path = base_path.joinpath(*args)
        if not path.exists():
            pytest.skip(f"Test data file not found: {path}")
        return path

    return _get_path


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
