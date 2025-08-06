# UXArray IO Test Structure Documentation

## Overview

This document describes the centralized and modular testing structure for UXArray's IO functionality. The design follows pytest best practices and promotes code reuse while maintaining format-specific customization.

## Architecture

### Core Components

1. **Base Test Classes** (`base_io_tests.py`)
   - Contains reusable test patterns for all grid formats
   - Provides common validation utilities
   - Defines standard test categories

2. **Format-Specific Test Modules** (e.g., `test_mpas.py`, `test_ugrid.py`)
   - Inherit from base classes
   - Define format-specific configurations
   - Add format-specific tests as needed

3. **Shared Fixtures** (`conftest.py`)
   - Centralized test data paths
   - Temporary directory management
   - Session-scoped fixtures for performance

## Base Test Classes

### `BaseIOReaderTests`
Tests basic grid reading functionality across all formats.

**Key Methods:**
- `test_read_grid_basic()`: Tests grid loading, validation, and structure
- `_validate_grid_structure()`: Common grid validation logic

**Usage:**
```python
class TestFormatReader(BaseIOReaderTests):
    format_configs = [("format_name", "data_key")]
```

### `BaseIOWriterTests`
Tests grid writing functionality for formats that support it.

**Key Methods:**
- `test_write_format()`: Tests writing grids to supported formats
- `_get_extension()`: Maps format names to file extensions

### `BaseIORoundTripTests`
Tests consistency when converting grids between formats.

**Key Methods:**
- `test_round_trip_consistency()`: Validates data preservation
- `_validate_round_trip_consistency()`: Checks topology/coordinate consistency

### `BaseIOEdgeCaseTests`
Tests error conditions and edge cases.

**Key Methods:**
- `test_invalid_file_path()`: File not found handling
- `test_corrupted_file_handling()`: Malformed file handling
- `test_standardized_dtype_and_fill_values()`: Data type consistency

### `BaseIODatasetTests`
Tests dataset (grid + data) functionality.

### `BaseIOPerformanceTests`
Tests performance characteristics like lazy loading.

## Format-Specific Test Modules

Each format has its own test module that inherits from base classes and adds format-specific tests.

### Current Format Modules

1. **`test_ugrid.py`** - CF-UGRID format
2. **`test_mpas.py`** - MPAS format (with dual mesh support)
3. **`test_esmf.py`** - ESMF format
4. **`test_exodus.py`** - Exodus format (supports mixed face types)
5. **`test_scrip.py`** - SCRIP format
6. **`test_icon.py`** - ICON format (icosahedral grids)
7. **`test_fesom.py`** - FESOM format (multiple input types)

### Format Module Structure

```python
"""Format-specific test module template."""

# Format-specific configurations
FORMAT_READ_CONFIGS = [
    ("format_name", "data_key1"),
    ("format_name", "data_key2")
]

FORMAT_WRITE_FORMATS = ["FormatName"]  # If writing is supported

FORMAT_ROUND_TRIP_CONFIGS = [
    ("format_name", "data_key", "output_format")
]

class TestFormatReader(BaseIOReaderTests):
    """Test format reading functionality."""
    format_configs = FORMAT_READ_CONFIGS

    def test_format_specific_feature(self, test_data_paths):
        """Format-specific test example."""
        # Custom test implementation

class TestFormatWriter(BaseIOWriterTests):
    """Test format writing functionality."""
    writable_formats = FORMAT_WRITE_FORMATS

class TestFormatRoundTrip(BaseIORoundTripTests):
    """Test format round-trip consistency."""
    round_trip_configs = FORMAT_ROUND_TRIP_CONFIGS

class TestFormatSpecialCases:
    """Format-specific special cases."""
    
    def test_special_format_behavior(self, test_data_paths):
        """Test unique format characteristics."""
        pass
```

## Test Data Organization

### Directory Structure
```
test/meshfiles/
├── ugrid/
│   ├── outCSne30/
│   ├── outRLL1deg/
│   └── ov_RLL10deg_CSne4/
├── exodus/
│   ├── outCSne8/
│   └── mixed/
├── esmf/
│   └── ne30/
├── scrip/
│   └── outCSne8/
├── mpas/
│   └── QU/
├── icon/
│   └── R02B04/
└── fesom/
    ├── pi/
    └── soufflet-netcdf/
```

### Test Data Paths Fixture
The `test_data_paths` fixture in `conftest.py` provides centralized access to all test files:

```python
@pytest.fixture(scope="session")
def test_data_paths():
    """Centralized test data paths for all formats."""
    return {
        "format_name": {
            "data_key": Path("path/to/test/file")
        }
    }
```

## Utility Functions

### Grid Validation
- `validate_grid_topology()`: Validates basic topology properties
- `validate_grid_coordinates()`: Validates coordinate ranges (handles degrees/radians)
- `compare_grids_topology()`: Compares topology between grids

### Usage Examples

```python
# Basic validation
validate_grid_topology(grid)
validate_grid_coordinates(grid)

# Grid comparison
compare_grids_topology(original_grid, converted_grid)
```

## Running Tests

### Run All Format Tests
```bash
pytest test/test_*.py
```

### Run Specific Format
```bash
pytest test/test_mpas.py
```

### Run Specific Test Category
```bash
pytest test/test_mpas.py::TestMPASReader
```

### Run Specific Test
```bash
pytest test/test_mpas.py::TestMPASReader::test_read_grid_basic
```

## Adding New Formats

To add support for a new format:

1. **Create format-specific test module** (`test_newformat.py`)
2. **Define format configurations**:
   ```python
   NEWFORMAT_READ_CONFIGS = [("newformat", "test_data_key")]
   ```
3. **Add test data paths** to `conftest.py`
4. **Inherit from base classes** and customize as needed
5. **Add format-specific tests** in dedicated classes

## Best Practices

### Test Organization
- Use inheritance for common functionality
- Add format-specific tests in separate classes
- Group related tests in logical classes

### Test Data
- Use descriptive data keys
- Ensure test files are representative
- Document special characteristics of test files

### Error Handling
- Use `pytest.skip()` for missing test files
- Test both success and failure cases
- Validate error messages when appropriate

### Performance
- Use session-scoped fixtures for expensive setup
- Skip tests gracefully when files are missing
- Set reasonable timeouts for performance tests

## Migration from Old Structure

The new structure replaces individual format test files that had significant code duplication. Key improvements:

1. **Reduced Duplication**: Common test patterns are centralized
2. **Consistent Coverage**: All formats get the same base test coverage
3. **Easy Extension**: Adding new formats requires minimal code
4. **Better Maintenance**: Changes to common patterns propagate automatically
5. **Flexible Customization**: Formats can still add specific tests as needed

## Format-Specific Notes

### MPAS
- Supports both primal and dual mesh configurations
- Tests both `use_dual=True` and `use_dual=False`
- Special handling for coordinate unit conversion

### Exodus
- Tests mixed face types (triangles and quadrilaterals)
- Validates face type analysis
- Round-trip testing through multiple formats

### FESOM
- Supports multiple input formats (ASCII, NetCDF, UGRID)
- Tests format comparison between ASCII and UGRID
- Ocean model specific validations

### ICON
- Icosahedral grid structure validation
- Triangular face predominance checks
- Resolution-specific property testing

### SCRIP
- SCRIP-specific variable detection
- Coordinate conversion validation
- Format detection testing

This structure provides a solid foundation for comprehensive IO testing while maintaining flexibility for format-specific requirements.