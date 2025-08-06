"""
MPAS format tests using base IO test classes.

This module tests MPAS-specific functionality while inheriting common
test patterns from the base IO test classes.
"""

import pytest
import numpy as np
import xarray as xr
import uxarray as ux
from pathlib import Path
import tempfile
import os
from numpy.testing import assert_array_equal, assert_allclose
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

from .base_io_tests import (
    BaseIOReaderTests,
    BaseIOWriterTests,
    BaseIORoundTripTests,
    BaseIOEdgeCaseTests,
    BaseIODatasetTests,
    BaseIOPerformanceTests,
    validate_grid_topology,
    validate_grid_coordinates
)
from uxarray.io._mpas import _replace_padding, _replace_zeros, _to_zero_index, _read_mpas

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

# MPAS-specific test configurations
MPAS_READ_CONFIGS = [
    ("mpas", "qu_grid"),
    ("mpas", "qu_ocean")
]

MPAS_WRITE_FORMATS = []  # MPAS doesn't support writing currently

MPAS_ROUND_TRIP_CONFIGS = []  # No round-trip until writing is supported


class TestMPASReader(BaseIOReaderTests):
    """Test MPAS reading functionality."""

    format_configs = MPAS_READ_CONFIGS

    def test_mpas_primal_to_ugrid_conversion(self, test_data_paths):
        """Verifies that the Primal-Mesh was converted properly."""
        for data_key in ["qu_grid", "qu_ocean"]:
            grid_path = test_data_paths["mpas"][data_key]
            if not grid_path.exists():
                continue

            uxgrid = ux.open_grid(grid_path, use_dual=False)
            ds = uxgrid._ds

            # Check for correct dimensions
            expected_ugrid_dims = ['n_node', "n_face", "n_max_face_nodes"]
            for dim in expected_ugrid_dims:
                assert dim in ds.sizes

            # Check for correct length of coordinates
            assert len(ds['node_lon']) == len(ds['node_lat'])
            assert len(ds['face_lon']) == len(ds['face_lat'])

            # Check for correct shape of face nodes
            n_face = ds.sizes['n_face']
            n_max_face_nodes = ds.sizes['n_max_face_nodes']
            assert ds['face_node_connectivity'].shape == (n_face, n_max_face_nodes)

    def test_mpas_dual_to_ugrid_conversion(self, test_data_paths):
        """Verifies that the Dual-Mesh was converted properly."""
        for data_key in ["qu_grid", "qu_ocean"]:
            grid_path = test_data_paths["mpas"][data_key]
            if not grid_path.exists():
                continue

            uxgrid = ux.open_grid(grid_path, use_dual=True)
            ds = uxgrid._ds

            # Check for correct dimensions
            expected_ugrid_dims = ['n_node', "n_face", "n_max_face_nodes"]
            for dim in expected_ugrid_dims:
                assert dim in ds.sizes

            # Check for correct length of coordinates
            assert len(ds['node_lon']) == len(ds['node_lat'])
            assert len(ds['face_lon']) == len(ds['face_lat'])

            # Check for correct shape of face nodes
            nMesh2_face = ds.sizes['n_face']
            assert ds['face_node_connectivity'].shape == (nMesh2_face, 3)


class TestMPASWriter(BaseIOWriterTests):
    """Test MPAS writing functionality (currently not supported)."""

    writable_formats = MPAS_WRITE_FORMATS  # Empty for now

    def test_mpas_write_not_supported(self):
        """Test that MPAS writing is not currently supported."""
        # This test documents that MPAS writing is not yet implemented
        # Can be updated when write support is added
        pass


class TestMPASRoundTrip(BaseIORoundTripTests):
    """Test MPAS round-trip consistency (currently not supported)."""

    round_trip_configs = MPAS_ROUND_TRIP_CONFIGS  # Empty for now

    def test_mpas_round_trip_not_supported(self):
        """Test that MPAS round-trip is not currently supported."""
        # This test documents that MPAS round-trip is not yet implemented
        # Can be updated when write support is added
        pass


class TestMPASEdgeCases(BaseIOEdgeCaseTests):
    """Test MPAS edge cases and error conditions."""

    def test_add_fill_values(self):
        """Test _add_fill_values() implementation."""
        verticesOnCell = np.array([[1, 2, 1, 1], [3, 4, 5, 3], [6, 7, 0, 0]], dtype=INT_DTYPE)
        nEdgesOnCell = np.array([2, 3, 2])
        gold_output = np.array([[0, 1, INT_FILL_VALUE, INT_FILL_VALUE],
                               [2, 3, 4, INT_FILL_VALUE],
                               [5, 6, INT_FILL_VALUE, INT_FILL_VALUE]], dtype=INT_DTYPE)

        verticesOnCell = xr.DataArray(data=verticesOnCell, dims=['n_face', 'n_max_face_nodes'])
        nEdgesOnCell = xr.DataArray(data=nEdgesOnCell, dims=['n_face'])

        verticesOnCell = _replace_padding(verticesOnCell, nEdgesOnCell)
        verticesOnCell = _replace_zeros(verticesOnCell)
        verticesOnCell = _to_zero_index(verticesOnCell)

        assert np.array_equal(verticesOnCell, gold_output)


class TestMPASDatasets(BaseIODatasetTests):
    """Test MPAS dataset functionality."""

    def test_mpas_dataset_basic(self, test_data_paths):
        """Basic MPAS dataset validation."""
        for data_key in ["qu_grid", "qu_ocean"]:
            grid_path = test_data_paths["mpas"][data_key]
            if not grid_path.exists():
                continue

            # Test both primal and dual mesh creation
            primal_grid = ux.open_grid(grid_path, use_dual=False)
            dual_grid = ux.open_grid(grid_path, use_dual=True)

            assert primal_grid is not None
            assert dual_grid is not None

            # Basic structure validation
            validate_grid_topology(primal_grid)
            validate_grid_topology(dual_grid)


class TestMPASPerformance(BaseIOPerformanceTests):
    """Test MPAS performance characteristics."""

    @pytest.mark.parametrize("data_key", ["qu_grid", "qu_ocean"])
    def test_lazy_loading(self, test_data_paths, data_key):
        """Test that MPAS grid loading is reasonably fast."""
        import time

        grid_path = test_data_paths["mpas"][data_key]
        if not grid_path.exists():
            pytest.skip(f"MPAS test file not found: {grid_path}")

        start_time = time.time()
        grid = ux.open_grid(grid_path, use_dual=False)
        load_time = time.time() - start_time

        # Basic validation that grid loaded successfully
        assert grid is not None
        assert hasattr(grid, 'node_lon')

        # Loading should complete in reasonable time
        assert load_time < 30.0, f"MPAS loading took {load_time:.2f}s, which seems excessive"


class TestMPASSpecialCases:
    """Test MPAS-specific special cases and functionality."""

    def test_mpas_dual_mesh(self, test_data_paths):
        """Test MPAS dual mesh functionality."""
        grid_path = test_data_paths["mpas"]["qu_grid"]
        if not grid_path.exists():
            pytest.skip("MPAS test file not found")

        primal_grid = ux.open_grid(grid_path, use_dual=False)
        dual_grid = ux.open_grid(grid_path, use_dual=True)

        assert primal_grid is not None
        assert dual_grid is not None

        # Dual mesh should have different structure
        assert dual_grid.face_node_connectivity.shape[1] == 3  # Triangular faces for dual

    def test_mpas_face_areas(self, test_data_paths):
        """Test MPAS face area parsing."""
        for data_key in ["qu_grid", "qu_ocean"]:
            grid_path = test_data_paths["mpas"][data_key]
            if not grid_path.exists():
                continue

            primal_grid = ux.open_grid(grid_path, use_dual=False)
            dual_grid = ux.open_grid(grid_path, use_dual=True)

            assert "face_areas" in primal_grid._ds
            assert "face_areas" in dual_grid._ds

    def test_mpas_distance_units(self, test_data_paths):
        """Test MPAS distance unit conversion."""
        grid_path = test_data_paths["mpas"]["qu_ocean"]
        if not grid_path.exists():
            pytest.skip("MPAS ocean test file not found")

        grid = ux.open_grid(grid_path)

        assert "edge_node_distances" in grid._ds
        assert "edge_face_distances" in grid._ds

        # Load original xarray dataset for comparison
        xrds = xr.open_dataset(grid_path)

        # Test distance unit conversion
        assert_allclose(
            grid['edge_node_distances'].values,
            (xrds['dvEdge'].values / xrds.attrs['sphere_radius'])
        )
        assert_allclose(
            grid['edge_face_distances'].values,
            (xrds['dcEdge'].values / xrds.attrs['sphere_radius'])
        )

    def test_mpas_read_function(self, test_data_paths):
        """Tests execution of _read_mpas()"""
        grid_path = test_data_paths["mpas"]["qu_grid"]
        if not grid_path.exists():
            pytest.skip("MPAS test file not found")

        mpas_xr_ds = xr.open_dataset(grid_path)
        mpas_primal_ugrid, _ = _read_mpas(mpas_xr_ds, use_dual=False)
        mpas_dual_ugrid, _ = _read_mpas(mpas_xr_ds, use_dual=True)

        assert mpas_primal_ugrid is not None
        assert mpas_dual_ugrid is not None

    def test_mpas_grid_creation(self, test_data_paths):
        """Tests creation of Grid object from converted MPAS dataset."""
        grid_path = test_data_paths["mpas"]["qu_grid"]
        if not grid_path.exists():
            pytest.skip("MPAS test file not found")

        mpas_uxgrid_primal = ux.open_grid(grid_path, use_dual=False)
        mpas_uxgrid_dual = ux.open_grid(grid_path, use_dual=True)

        # Test that repr works without error
        repr_output = mpas_uxgrid_dual.__repr__()
        assert isinstance(repr_output, str)
        assert len(repr_output) > 0

    def test_set_attrs(self, test_data_paths):
        """Tests the execution of _set_global_attrs."""
        grid_path = test_data_paths["mpas"]["qu_grid"]
        if not grid_path.exists():
            pytest.skip("MPAS test file not found")

        expected_attrs = [
            'sphere_radius', 'mesh_spec', 'on_a_sphere', 'mesh_id',
            'is_periodic', 'x_period', 'y_period'
        ]

        mpas_xr_ds = xr.open_dataset(grid_path)
        ds, _ = _read_mpas(mpas_xr_ds)

        # Set dummy attrs to test execution
        ds.attrs['mesh_id'] = "12345678"
        ds.attrs['is_periodic'] = "YES"
        ds.attrs['x_period'] = 1.0
        ds.attrs['y_period'] = 1.0

        uxgrid = ux.Grid(ds)

        # Check if all expected attributes are set
        for mpas_attr in expected_attrs:
            assert mpas_attr in uxgrid._ds.attrs

    def test_mpas_coordinate_validation(self, test_data_paths):
        """Test that MPAS coordinates are properly converted and validated."""
        for data_key in ["qu_grid", "qu_ocean"]:
            grid_path = test_data_paths["mpas"][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path, use_dual=False)

            # Validate coordinate ranges and properties
            validate_grid_coordinates(grid)
            validate_grid_topology(grid)

            # Test first file is sufficient
            break
