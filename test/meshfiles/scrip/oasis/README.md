# OASIS Multi-Grid SCRIP Test Files

This directory contains small test files for OASIS/YAC multi-grid SCRIP format support in UXarray.

## Files

- `grids.nc`: Multi-grid file containing two grids
  - `ocn`: Ocean grid with 12 cells (3x4 regular grid)
  - `atm`: Atmosphere grid with 20 cells (4x5 regular grid)

- `masks.nc`: Domain masks for the grids
  - `ocn.msk`: Ocean mask (8 ocean cells, 4 land cells)
  - `atm.msk`: Atmosphere mask (all 20 cells active)

## OASIS Format

OASIS uses a specific naming convention for multi-grid SCRIP files:
- Grid variables are prefixed with grid name: `<gridname>.<varname>`
- Corner latitudes: `<gridname>.cla`
- Corner longitudes: `<gridname>.clo`
- Dimensions: `nc_<gridname>` (cells), `nv_<gridname>` (corners)

## Usage in Tests

```python
import uxarray as ux

# List available grids
grid_names = ux.list_grid_names("grids.nc")
# ['ocn', 'atm']

# Load all grids
grids = ux.open_multigrid("grids.nc")

# Load with masks
masked_grids = ux.open_multigrid("grids.nc", mask_filename="masks.nc")
# Ocean grid will have 8 cells, atmosphere grid will have 20 cells
```

## File Sizes

These files are intentionally small for fast testing:
- `grids.nc`: ~3 KB
- `masks.nc`: ~1 KB
