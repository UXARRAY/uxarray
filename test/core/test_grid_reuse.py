"""Regression tests for opening ensemble members with a shared Grid."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import dask.array as da
import numpy as np
import xarray as xr

import uxarray as ux


class TestGridReuse(TestCase):
    def setUp(self):
        """Create a synthetic tetrahedral grid and two independent members."""
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.grid_path = Path(self.directory.name) / "grid.nc"
        # Equal node and face counts ensure source names, rather than sizes,
        # distinguish node-centered and face-centered variables after reuse.
        self.grid_ds = xr.Dataset(
            {
                "mesh": xr.DataArray(
                    0,
                    attrs={
                        "cf_role": "mesh_topology",
                        "topology_dimension": 2,
                        "node_coordinates": "lon lat",
                        "face_node_connectivity": "face_nodes",
                        "node_dimension": "vertices",
                        "face_dimension": "cells",
                    },
                ),
                "lon": ("vertices", [0.0, 120.0, -120.0, 0.0]),
                "lat": ("vertices", [0.0, 0.0, 0.0, 90.0]),
                "face_nodes": (
                    ("cells", "corners"),
                    [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]],
                    {"start_index": 0},
                ),
            }
        )
        self.grid_ds.to_netcdf(self.grid_path)
        self.grid = ux.open_grid(self.grid_path)
        self.addCleanup(self.grid._ds.close)
        self.source_dims = self.grid._source_dims_dict.copy()
        self.grid_snapshot = self.grid._ds.copy(deep=True)
        self.members = [
            xr.Dataset(
                {"temperature": (("time", "cells"), np.arange(12).reshape(3, 4) + i)},
                coords={"time": [0, 1, 2]},
            )
            for i in range(2)
        ]
        self.paths = []
        for i, member in enumerate(self.members):
            path = Path(self.directory.name) / f"member_{i}.nc"
            member.to_netcdf(path)
            self.paths.append(path)

    def assert_grid_unchanged(self):
        """Opening data must not alter the shared grid or its source mapping."""
        self.assertEqual(self.grid._source_dims_dict, self.source_dims)
        xr.testing.assert_identical(self.grid._ds, self.grid_snapshot)

    def test_open_dataset_reuses_grid_for_file_and_dataset_members(self):
        """A single Grid is attached by identity to independent member data."""
        for sources in (self.paths, self.members):
            for source, expected in zip(sources, self.members):
                with self.subTest(source=type(source).__name__):
                    result = ux.open_dataset(self.grid, source)
                    self.addCleanup(result.close)
                    self.assertIs(result.uxgrid, self.grid)
                    self.assertEqual(result.temperature.dims, ("time", "n_face"))
                    np.testing.assert_array_equal(
                        result.temperature, expected.temperature
                    )
                    self.assertEqual(
                        result.source_datasets,
                        None if isinstance(source, xr.Dataset) else str(source),
                    )
                    self.assert_grid_unchanged()

    def test_members_with_different_grid_dimensions_preserve_mapping(self):
        """Opening face-only data must not lose the next member's node mapping."""
        node_data = xr.Dataset({"velocity": ("vertices", np.arange(4))})
        for member, name, dimension in (
            (self.members[0], "temperature", "n_face"),
            (node_data, "velocity", "n_node"),
            (self.members[1], "temperature", "n_face"),
        ):
            result = ux.open_dataset(self.grid, member)
            self.assertEqual(result[name].dims[-1], dimension)
            self.assert_grid_unchanged()

    def test_chunks_preserve_grid_and_caller_options(self):
        """Chunk data with source or UGRID names without rechunking the Grid."""
        for source in (self.paths[0], self.members[0]):
            for chunk_grid in (True, False):
                for chunks in (
                    {"n_face": 2, "time": 1},
                    {"cells": 2, "time": 1},
                    {"cells": 4, "n_face": 2, "time": 1},
                ):
                    with self.subTest(
                        source=type(source).__name__,
                        chunk_grid=chunk_grid,
                        chunks=chunks,
                    ):
                        original_chunks = chunks.copy()
                        grid_kwargs = {}
                        result = ux.open_dataset(
                            self.grid,
                            source,
                            chunks=chunks,
                            chunk_grid=chunk_grid,
                            grid_kwargs=grid_kwargs,
                        )
                        self.addCleanup(result.close)
                        self.assertIs(result.uxgrid, self.grid)
                        self.assertIsInstance(result.temperature.data, da.Array)
                        self.assertEqual(result.temperature.chunks, ((1, 1, 1), (2, 2)))
                        np.testing.assert_array_equal(
                            result.temperature.compute(), self.members[0].temperature
                        )
                        self.assertEqual(chunks, original_chunks)
                        self.assertEqual(grid_kwargs, {})
                        self.assert_grid_unchanged()

    def test_scalar_and_auto_chunks(self):
        """All non-dict chunk forms reach the member data unchanged."""
        for source in (self.paths[0], self.members[0]):
            for chunks in (-1, "auto", {}, None):
                with self.subTest(source=type(source).__name__, chunks=chunks):
                    result = ux.open_dataset(self.grid, source, chunks=chunks)
                    self.addCleanup(result.close)
                    self.assertEqual(
                        isinstance(result.temperature.data, da.Array),
                        chunks is not None,
                    )
                    self.assert_grid_unchanged()

    def test_prechunked_grid_is_not_rechunked(self):
        """Each member retains the same already-chunked grid arrays."""
        self.grid.chunk(n_node=2, n_face=2)
        connectivity = self.grid.face_node_connectivity.data
        result = ux.open_dataset(self.grid, self.paths[0], chunks={"n_face": 1})
        self.addCleanup(result.close)
        self.assertIs(result.uxgrid, self.grid)
        self.assertIs(self.grid.face_node_connectivity.data, connectivity)
        self.assertEqual(result.temperature.chunks[-1], (1, 1, 1, 1))
        self.assertEqual(self.grid.face_node_connectivity.chunks[0], (2, 2))

    def test_existing_grid_does_not_open_or_reconstruct_grid(self):
        """An in-memory Grid/Dataset pair must not invoke file backends."""
        with (
            patch("uxarray.core.api.open_grid") as open_grid,
            patch("xarray.open_dataset") as open_dataset,
        ):
            result = ux.open_dataset(self.grid, self.members[0], chunks={"n_face": 2})
        self.assertIs(result.uxgrid, self.grid)
        open_grid.assert_not_called()
        open_dataset.assert_not_called()

    def test_open_mfdataset_reuses_grid(self):
        """The multi-file API shares the same reuse and chunking contract."""
        chunks = {"n_face": 2, "time": 1}
        with patch("uxarray.core.api.open_grid") as open_grid:
            result = ux.open_mfdataset(
                self.grid,
                self.paths,
                chunks=chunks,
                combine="nested",
                concat_dim="member",
            )
        self.addCleanup(result.close)
        open_grid.assert_not_called()
        self.assertIs(result.uxgrid, self.grid)
        self.assertEqual(result.temperature.dims, ("member", "time", "n_face"))
        self.assertEqual(result.temperature.chunks, ((1, 1), (1, 1, 1), (2, 2)))
        np.testing.assert_array_equal(
            result.temperature.compute(),
            np.stack([member.temperature.values for member in self.members]),
        )
        self.assertEqual(chunks, {"n_face": 2, "time": 1})
        self.assert_grid_unchanged()

    def test_grid_construction_options_are_rejected(self):
        """Options requiring grid construction should not be silently ignored."""
        for opener, data in (
            (ux.open_dataset, self.members[0]),
            (ux.open_mfdataset, self.paths),
        ):
            for options, message in (
                ({"use_dual": True}, "use_dual"),
                ({"grid_kwargs": {"drop_variables": "face_nodes"}}, "grid_kwargs"),
            ):
                with self.subTest(opener=opener.__name__, options=options):
                    with self.assertRaisesRegex(ValueError, message):
                        opener(self.grid, data, **options)
                    self.assert_grid_unchanged()

    def test_grid_requires_separate_data(self):
        """A Grid alone does not provide member data."""
        with self.assertRaisesRegex(TypeError, "filename_or_obj"):
            ux.open_dataset(self.grid)

    def test_canonical_dimensions_are_not_inferred_from_equal_sizes(self):
        """Canonical face and node dimensions remain distinct across members."""
        members = [
            member.rename({"cells": "n_face"}).assign(
                velocity=("n_node", np.arange(4) + i)
            )
            for i, member in enumerate(self.members)
        ]
        paths = []
        for i, member in enumerate(members):
            path = Path(self.directory.name) / f"canonical_{i}.nc"
            member.to_netcdf(path)
            paths.append(path)
        for sources in (members, paths):
            for source, expected in zip(sources, members):
                result = ux.open_dataset(
                    self.grid, source, chunks={"n_face": 2, "n_node": 1}
                )
                self.addCleanup(result.close)
                self.assertIs(result.uxgrid, self.grid)
                self.assertEqual(result.temperature.dims, ("time", "n_face"))
                self.assertEqual(result.velocity.dims, ("n_node",))
                self.assertEqual(result.temperature.chunks[-1], (2, 2))
                self.assertEqual(result.velocity.chunks, ((1, 1, 1, 1),))
                np.testing.assert_array_equal(result.temperature, expected.temperature)
                np.testing.assert_array_equal(result.velocity, expected.velocity)
                self.assert_grid_unchanged()
        combined = ux.open_mfdataset(
            self.grid,
            paths,
            combine="nested",
            concat_dim="member",
            chunks={"n_face": 2, "n_node": 1},
        )
        self.addCleanup(combined.close)
        self.assertEqual(combined.temperature.dims, ("member", "time", "n_face"))
        self.assertEqual(combined.velocity.dims, ("member", "n_node"))
        np.testing.assert_array_equal(
            combined.temperature,
            np.stack([member.temperature.values for member in members]),
        )
        self.assert_grid_unchanged()
        # Canonical members must not disrupt later source-dimension members.
        native = ux.open_dataset(self.grid, self.members[0])
        self.assertEqual(native.temperature.dims, ("time", "n_face"))
        self.assert_grid_unchanged()

    def test_topology_grid_without_source_mapping(self):
        """A topology-created Grid can reuse already-standardized data."""
        grid = ux.Grid.from_topology(
            node_lon=self.grid.node_lon.values,
            node_lat=self.grid.node_lat.values,
            face_node_connectivity=self.grid.face_node_connectivity.values,
        )
        member = xr.Dataset({"temperature": ("n_face", np.arange(4))})
        result = ux.open_dataset(grid, member, chunks={"n_face": 2})
        self.assertIs(result.uxgrid, grid)
        self.assertEqual(result.temperature.dims, ("n_face",))
        self.assertEqual(result.temperature.chunks, ((2, 2),))
        np.testing.assert_array_equal(result.temperature, member.temperature)
        self.assertEqual(grid._source_dims_dict, {})

    def test_explicit_source_mapping_takes_precedence(self):
        """An explicit source mapping wins even for a canonical-looking name."""
        self.grid._source_dims_dict = {"n_face": "n_node"}
        result = ux.open_dataset(
            self.grid, xr.Dataset({"velocity": ("n_face", np.arange(4))})
        )
        self.assertEqual(result.velocity.dims, ("n_node",))
        self.assertEqual(self.grid._source_dims_dict, {"n_face": "n_node"})

    def test_structured_grid_chunks_use_source_dimensions(self):
        """Structured grids retain their tuple-valued source mapping on reuse."""
        grid = ux.Grid.from_structured(
            lon=xr.DataArray([0.0, 20.0, 40.0, 60.0], dims="x"),
            lat=xr.DataArray([-30.0, 0.0, 30.0], dims="y"),
        )
        source_dims = grid._source_dims_dict.copy()
        member = xr.Dataset({"temperature": (("y", "x"), np.arange(12).reshape(3, 4))})
        path = Path(self.directory.name) / "structured.nc"
        member.to_netcdf(path)
        for source in (member, path):
            chunks = {"x": 2, "y": 1}
            result = ux.open_dataset(grid, source, chunks=chunks)
            self.addCleanup(result.close)
            self.assertIs(result.uxgrid, grid)
            self.assertEqual(result.temperature.dims, ("n_face",))
            self.assertEqual(result.temperature.chunks, ((2, 2, 2, 2, 2, 2),))
            np.testing.assert_array_equal(result.temperature, np.arange(12))
            self.assertEqual(grid._source_dims_dict, source_dims)
            self.assertEqual(chunks, {"x": 2, "y": 1})

    def test_structured_grid_accepts_already_flattened_members(self):
        """Do not flatten canonical structured member data a second time."""
        grid = ux.Grid.from_structured(
            lon=xr.DataArray([0.0, 20.0, 40.0, 60.0], dims="x"),
            lat=xr.DataArray([-30.0, 0.0, 30.0], dims="y"),
        )
        source_dims = grid._source_dims_dict.copy()
        members = [
            xr.Dataset({"temperature": ("n_face", np.arange(12) + i)}) for i in range(2)
        ]
        paths = []
        for i, member in enumerate(members):
            path = Path(self.directory.name) / f"flattened_{i}.nc"
            member.to_netcdf(path)
            paths.append(path)
        for sources in (members, paths):
            for source, expected in zip(sources, members):
                result = ux.open_dataset(grid, source, chunks={"n_face": 2})
                self.addCleanup(result.close)
                self.assertIs(result.uxgrid, grid)
                self.assertEqual(result.temperature.dims, ("n_face",))
                self.assertEqual(result.temperature.chunks, ((2, 2, 2, 2, 2, 2),))
                np.testing.assert_array_equal(result.temperature, expected.temperature)
                self.assertEqual(grid._source_dims_dict, source_dims)
        combined = ux.open_mfdataset(
            grid,
            paths,
            combine="nested",
            concat_dim="member",
            chunks={"n_face": 2},
        )
        self.addCleanup(combined.close)
        self.assertIs(combined.uxgrid, grid)
        self.assertEqual(combined.temperature.dims, ("member", "n_face"))
        self.assertEqual(combined.temperature.chunks, ((1, 1), (2, 2, 2, 2, 2, 2)))
        np.testing.assert_array_equal(
            combined.temperature,
            np.stack([member.temperature.values for member in members]),
        )
        self.assertEqual(grid._source_dims_dict, source_dims)

    def test_existing_grid_file_and_dataset_inputs(self):
        """Previously supported grid inputs still construct equivalent grids."""
        for grid_source in (self.grid_path, self.grid_ds):
            result = ux.open_dataset(grid_source, self.paths[0], chunks={"n_face": 2})
            self.addCleanup(result.close)
            self.addCleanup(result.uxgrid._ds.close)
            self.assertIsNot(result.uxgrid, self.grid)
            self.assertEqual(result.temperature.dims, ("time", "n_face"))
            self.assertEqual(result.temperature.chunks[-1], (2, 2))
            np.testing.assert_array_equal(
                result.temperature, self.members[0].temperature
            )
            xr.testing.assert_identical(result.uxgrid._ds.compute(), self.grid_snapshot)
