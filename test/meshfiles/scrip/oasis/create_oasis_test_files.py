#!/usr/bin/env python
"""Create small OASIS test files for UXarray tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

BASE_DIR = Path(__file__).parent


def create_oasis_test_files() -> None:
    """Generate deterministic OASIS-style multi-grid SCRIP files."""
    output_dir = BASE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_path = output_dir / "grids.nc"
    mask_path = output_dir / "masks.nc"

    grid_ds = xr.Dataset()

    # Small ocean grid (3 latitude bands x 4 longitude bands = 12 cells)
    n_ocean = 12
    grid_ds.coords["nc_ocn"] = np.arange(n_ocean)
    grid_ds.coords["nv_ocn"] = np.arange(4)

    ocean_lons = np.array([0, 120, 240, 360])
    ocean_lats = np.array([-60, -20, 20, 60])

    ocean_clo = np.zeros((n_ocean, 4))
    ocean_cla = np.zeros((n_ocean, 4))

    idx = 0
    for j in range(3):  # latitude bands
        for i in range(4):  # longitude bands
            ocean_clo[idx, 0] = ocean_lons[i]
            ocean_clo[idx, 1] = ocean_lons[(i + 1) % 4]
            ocean_clo[idx, 2] = ocean_lons[(i + 1) % 4]
            ocean_clo[idx, 3] = ocean_lons[i]

            ocean_cla[idx, 0] = ocean_lats[j]
            ocean_cla[idx, 1] = ocean_lats[j]
            ocean_cla[idx, 2] = ocean_lats[j + 1]
            ocean_cla[idx, 3] = ocean_lats[j + 1]
            idx += 1

    grid_ds["ocn.clo"] = xr.DataArray(
        ocean_clo,
        dims=["nc_ocn", "nv_ocn"],
        attrs={
            "units": "degrees_east",
            "long_name": "ocean grid corner longitude",
        },
    )
    grid_ds["ocn.cla"] = xr.DataArray(
        ocean_cla,
        dims=["nc_ocn", "nv_ocn"],
        attrs={
            "units": "degrees_north",
            "long_name": "ocean grid corner latitude",
        },
    )

    # Small atmosphere grid (4 latitude bands x 5 longitude bands = 20 cells)
    n_atmos = 20
    grid_ds.coords["nc_atm"] = np.arange(n_atmos)
    grid_ds.coords["nv_atm"] = np.arange(4)

    atm_lons = np.array([0, 90, 180, 270, 360])
    atm_lats = np.array([-90, -45, 0, 45, 90])

    atm_clo = np.zeros((n_atmos, 4))
    atm_cla = np.zeros((n_atmos, 4))

    idx = 0
    for j in range(4):
        for i in range(5):
            atm_clo[idx, 0] = atm_lons[i]
            atm_clo[idx, 1] = atm_lons[(i + 1) % 5]
            atm_clo[idx, 2] = atm_lons[(i + 1) % 5]
            atm_clo[idx, 3] = atm_lons[i]

            atm_cla[idx, 0] = atm_lats[j]
            atm_cla[idx, 1] = atm_lats[j]
            atm_cla[idx, 2] = atm_lats[j + 1]
            atm_cla[idx, 3] = atm_lats[j + 1]
            idx += 1

    grid_ds["atm.clo"] = xr.DataArray(
        atm_clo,
        dims=["nc_atm", "nv_atm"],
        attrs={
            "units": "degrees_east",
            "long_name": "atmosphere grid corner longitude",
        },
    )
    grid_ds["atm.cla"] = xr.DataArray(
        atm_cla,
        dims=["nc_atm", "nv_atm"],
        attrs={
            "units": "degrees_north",
            "long_name": "atmosphere grid corner latitude",
        },
    )

    grid_ds.attrs["title"] = "OASIS multi-grid test file"
    grid_ds.attrs["description"] = "Small test grids for UXarray OASIS support"
    grid_ds.attrs["conventions"] = "SCRIP"
    grid_ds.attrs["grid_type"] = "curvilinear"

    grid_ds.to_netcdf(grid_path, engine="scipy")

    mask_ds = xr.Dataset()

    ocean_mask = np.ones(n_ocean, dtype=np.int32)
    ocean_mask[8:] = 0
    mask_ds["ocn.msk"] = xr.DataArray(
        ocean_mask,
        dims=["nc_ocn"],
        attrs={
            "long_name": "ocean domain mask",
            "valid_values": "0: land, 1: ocean",
        },
    )

    atmos_mask = np.ones(n_atmos, dtype=np.int32)
    mask_ds["atm.msk"] = xr.DataArray(
        atmos_mask,
        dims=["nc_atm"],
        attrs={
            "long_name": "atmosphere domain mask",
            "valid_values": "0: inactive, 1: active",
        },
    )

    mask_ds.attrs["title"] = "OASIS mask file"
    mask_ds.attrs["description"] = "Domain masks for ocean and atmosphere grids"

    mask_ds.to_netcdf(mask_path, engine="scipy")

    print(f"Created {grid_path} and {mask_path}")


if __name__ == "__main__":
    create_oasis_test_files()
