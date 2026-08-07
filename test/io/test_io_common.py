"""
Common IO tests that apply to all grid formats. These tests make sure the
same basic things work no matter which file format you start with.
"""

import pytest
import numpy as np
import xarray as xr
import uxarray as ux
import tempfile
import warnings
from numpy.testing import assert_array_equal, assert_allclose
from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE, INT_FILL_VALUE





# Define all testable format combinations
# Format: (format_type, subpath, filename)
IO_READ_TEST_FORMATS = [
    ("ugrid", "ugrid/quad-hexagon", "grid.nc"),
    ("ugrid", "ugrid/outCSne30", "outCSne30.ug"),
    ("ugrid", "ugrid/outRLL1deg", "outRLL1deg.ug"),
    ("mpas", "mpas/QU/480", "grid.nc"),
    ("esmf", "esmf/ne30", "ne30pg3.grid.nc"),
    ("exodus", "exodus/outCSne8", "outCSne8.g"),
    ("exodus", "exodus/mixed", "mixed.exo"),
    ("scrip", "scrip/outCSne8", "outCSne8.nc"),
    ("icon", "icon/R02B04", "icon_grid_0010_R02B04_G.nc"),
    ("fesom", "fesom/pi", None),  # Special case - multiple files
    ("healpix", None, None),  # Constructed via classmethod
]

# Formats that support writing
WRITABLE_FORMATS = ["ugrid", "exodus", "scrip", "esmf"]

# SCRIP stores corner coordinates rather than node indices, so its reader
# rebuilds nodes by deduplicating coordinates and renumbers them in the process.
# Its geometry survives a round trip; its connectivity is not restored verbatim.
EXACT_CONNECTIVITY_FORMATS = ["ugrid", "exodus", "esmf"]

# Format conversion test pairs - removed for now as format conversion
# requires more sophisticated handling than simple to_netcdf


@pytest.fixture(params=IO_READ_TEST_FORMATS)
def grid_from_format(request, test_data_dir):
    """Load a Grid from each supported format for parameterized tests.

    Handles special cases (FESOM multi-file, HEALPix) and tags the grid with
    ``_test_format`` for easier debugging.
    """
    format_name, subpath, filename = request.param

    if format_name == "fesom" and filename is None:
        # Special handling for FESOM with multiple input files
        fesom_data_path = test_data_dir / "fesom" / "pi" / "data"
        fesom_mesh_path = test_data_dir / "fesom" / "pi"
        grid = ux.open_grid(fesom_mesh_path, fesom_data_path)
    elif format_name == "healpix":
        # Construct a basic HEALPix grid
        grid = ux.Grid.from_healpix(zoom=1, pixels_only=False)
    else:
        grid_path = test_data_dir / subpath / filename
        if not grid_path.exists():
            pytest.skip(f"Test file not found: {grid_path}")

        # Handle special cases
        if format_name == "mpas":
            grid = ux.open_grid(grid_path, use_dual=False)
        else:
            grid = ux.open_grid(grid_path)

    # Add format info to the grid for test identification
    grid._test_format = format_name
    return grid


# File suffix to write each format under. Exodus is sniffed by extension.
FORMAT_SUFFIX = {"ugrid": ".nc", "exodus": ".exo", "scrip": ".nc", "esmf": ".nc"}

# Formats that write node indices rather than coordinates, as
# {format: (variable name prefix, on-disk fill value or None)}. Exodus splits
# connectivity across one exactly-sized connect<N> per element block, so it has
# no padding to skip; ESMF writes a single padded array.
ENCODED_INDEX_VARS = {"exodus": ("connect", None), "esmf": ("elementConn", -1)}

RAGGED_FACE_NODES = np.array(
    [
        [0, 1, 2, 3],  # quad
        [1, 4, 2, INT_FILL_VALUE],  # triangle
        [0, 3, 4, INT_FILL_VALUE],  # triangle
    ]
)
RAGGED_NODE_LON = np.array([0.0, 10.0, 10.0, 0.0, 20.0])
RAGGED_NODE_LAT = np.array([0.0, 0.0, 10.0, 10.0, 0.0])


@pytest.fixture
def ragged_grid():
    """A grid whose faces are not all the same size.

    Every uniform grid pads nothing, so the fill-value paths in the encoders are
    only reachable with mixed face sizes.
    """
    return ux.Grid.from_topology(
        node_lon=RAGGED_NODE_LON,
        node_lat=RAGGED_NODE_LAT,
        face_node_connectivity=RAGGED_FACE_NODES,
        fill_value=INT_FILL_VALUE,
    )


def _write_and_reload(grid, fmt, directory):
    """Encode ``grid`` as ``fmt``, write it out, and read it back."""
    path = directory / f"round_trip_{fmt}{FORMAT_SUFFIX[fmt]}"
    grid.to_xarray(fmt).to_netcdf(path)
    return ux.open_grid(path)


def _face_geometry(grid):
    """Describe each face by its corner coordinates instead of node indices.

    Lets formats that renumber nodes, or that pad a short face by repeating a
    vertex, be compared against the grid they were written from.
    """
    conn = grid.face_node_connectivity.values
    lon = grid.node_lon.values
    lat = grid.node_lat.values

    faces = []
    for row in conn:
        corners = {
            (round(float(lon[i]), 6), round(float(lat[i]), 6))
            for i in row
            if i != INT_FILL_VALUE
        }
        faces.append(tuple(sorted(corners)))
    return sorted(faces)


class TestIOCommon:
    """Common IO tests across all formats. Helps catch format-specific
    regressions early and keep behavior consistent.
    """

    def test_return_type(self, grid_from_format):
        """Open each format and return a ux.Grid. Checks that the public API
        is consistent across readers.
        """
        grid = grid_from_format

        # Basic validation
        assert isinstance(grid, ux.Grid)

    def test_ugrid_compliance(self, grid_from_format):
        """Check that a loaded grid looks like a UGRID mesh. We look for
        required topology, coordinates, proper fill values, reasonable degree
        ranges, and that ``validate()`` passes.
        """
        grid = grid_from_format

        # Basic topology and coordinate presence
        assert 'face_node_connectivity' in grid.connectivity
        assert 'node_lon' in grid.coordinates
        assert 'node_lat' in grid.coordinates

        # Required dimensions
        assert 'n_node' in grid.dims
        assert 'n_face' in grid.dims

        # Validate grid structure
        assert grid.validate()

        # Check UGRID compliance
        # 1. Connectivity should use proper fill values
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE

        # 3. Check that grid has been properly standardized by uxarray
        # (Not all input files have Conventions attribute, but uxarray should handle them)

    def test_grid_properties_consistency(self, grid_from_format):
        """Make sure core dims and variables are present with the expected
        dtypes across formats. Avoid surprises for downstream code.
        """
        grid = grid_from_format

        # Check that all grids have the essential properties
        assert 'n_node' in grid.dims
        assert 'n_face' in grid.dims
        assert 'face_node_connectivity' in grid.connectivity
        assert 'node_lon' in grid.coordinates
        assert 'node_lat' in grid.coordinates

        # Check data types are consistent
        assert np.issubdtype(grid.face_node_connectivity.dtype, np.integer)
        assert np.issubdtype(grid.node_lon.dtype, np.floating)
        assert np.issubdtype(grid.node_lat.dtype, np.floating)

    def test_standardized_dtype_and_fill(self, grid_from_format):
        """Test that face_node_connectivity uses expected dtype and fill value across all formats."""
        grid = grid_from_format

        # Check that face_node_connectivity uses an integer dtype (may vary by platform/format)
        assert np.issubdtype(grid.face_node_connectivity.dtype, np.integer)

        # Check that face_node_connectivity uses the standardized fill value
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE


class TestIOWriteRoundTrip:
    """Write each format back out and read it in again.

    The encoders historically broke on padded connectivity: a fill value that
    gets offset, narrowed to a smaller dtype, or written out as a coordinate
    comes back as a real vertex. Nothing raises when that happens -- the mesh
    just quietly gains nodes -- so these tests assert on the reloaded topology
    rather than on the write succeeding.
    """

    @pytest.mark.parametrize("fmt", EXACT_CONNECTIVITY_FORMATS)
    def test_uniform_grid_round_trip(self, fmt, gridpath, tmp_path):
        """A grid with uniform face sizes survives a write/read cycle intact."""
        original = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
        reloaded = _write_and_reload(original, fmt, tmp_path)

        assert_array_equal(
            original.face_node_connectivity.values,
            reloaded.face_node_connectivity.values,
            err_msg=f"{fmt}: face connectivity changed across a round trip",
        )
        assert_allclose(
            original.node_lon.values, reloaded.node_lon.values, rtol=ERROR_TOLERANCE
        )
        assert_allclose(
            original.node_lat.values, reloaded.node_lat.values, rtol=ERROR_TOLERANCE
        )

    @pytest.mark.parametrize("fmt", WRITABLE_FORMATS)
    def test_ragged_grid_round_trip_adds_no_nodes(self, fmt, ragged_grid, tmp_path):
        """Padding must not survive a round trip as a usable vertex.

        Covers the whole family at once: an unguarded index offset, a narrowing
        cast that truncates the fill value into a small valid index, and padding
        written out as NaN coordinates that dedupe into a phantom node.
        """
        reloaded = _write_and_reload(ragged_grid, fmt, tmp_path)

        assert reloaded.n_face == ragged_grid.n_face
        assert reloaded.n_node == ragged_grid.n_node, (
            f"{fmt}: round trip changed the node count, "
            "which means padding became a real vertex"
        )
        assert not np.isnan(reloaded.node_lon.values).any(), f"{fmt}: NaN node_lon"
        assert not np.isnan(reloaded.node_lat.values).any(), f"{fmt}: NaN node_lat"

        # Anything that is not the fill value has to be a usable index. A
        # negative leftover is the dangerous case: it is a valid Python index
        # that silently wraps to the end of the coordinate array.
        conn = reloaded.face_node_connectivity.values
        valid = conn[conn != INT_FILL_VALUE]
        assert valid.min() >= 0, f"{fmt}: negative index left in connectivity"
        assert valid.max() < reloaded.n_node, f"{fmt}: out-of-range node index"

        # Node renumbering is allowed; changing the shape of a face is not.
        assert _face_geometry(reloaded) == _face_geometry(ragged_grid), (
            f"{fmt}: face geometry changed across a round trip"
        )

    @pytest.mark.parametrize("fmt", list(ENCODED_INDEX_VARS))
    def test_ragged_grid_encodes_usable_indices(self, fmt, ragged_grid):
        """Every index written out must name a real node or be the fill value.

        A round trip can hide this. Exodus connectivity is int64, so the
        writer's +1 offset and the reader's -1 cancel exactly at
        INT_FILL_VALUE: the grid reloads intact even when the file holds an
        index no other Exodus reader could use. Check the encoded output
        directly rather than trusting the trip back.
        """
        encoded = ragged_grid.to_xarray(fmt)
        prefix, fill = ENCODED_INDEX_VARS[fmt]

        index_vars = [v for v in encoded.data_vars if v.startswith(prefix)]
        assert index_vars, f"{fmt}: no connectivity variable written"

        for name in index_vars:
            values = encoded[name].values
            if fill is not None:
                values = values[values != fill]
            assert values.min() >= 1, f"{fmt}: {name} holds an index below 1"
            assert values.max() <= ragged_grid.n_node, (
                f"{fmt}: {name} indexes a node that does not exist"
            )

    @pytest.mark.parametrize("fmt", EXACT_CONNECTIVITY_FORMATS)
    def test_ragged_grid_round_trip_is_exact(self, fmt, ragged_grid, tmp_path):
        """Index-based formats restore ragged connectivity verbatim.

        Face order matters as much as face content: a reordered mesh silently
        misaligns face-centered data with the faces it describes.
        """
        reloaded = _write_and_reload(ragged_grid, fmt, tmp_path)

        assert_array_equal(
            ragged_grid.face_node_connectivity.values,
            reloaded.face_node_connectivity.values,
            err_msg=f"{fmt}: ragged connectivity not preserved",
        )
        assert_array_equal(
            reloaded.n_nodes_per_face.values,
            np.array([4, 3, 3]),
            err_msg=f"{fmt}: face sizes not preserved",
        )
