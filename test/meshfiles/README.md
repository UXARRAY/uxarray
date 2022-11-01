# Meshfiles for testing
## Filenames for each format are as follows

* Shape:
  * grid_fire.shp
* Exodus:
  * outCSne8.g
* SCRIP:
  * outCSne30_vortex.nc
  * outCSne8.nc
  * outRLL1deg_vortex.nc
  * ov_RLL10deg_CSne4_vortex.nc
* UGRID:
  * outCSne30.ug
  * outRLL1deg.ug
  * ov_RLL10deg_CSne4.ug
  * outCSne30_var2.ug

## FESOM Meshfiles

The subfolder `fesom` contains some example output of the [`FESOM 2.1`](https://github.com/FESOM/fesom2) model.

* `fesom.mesh.diag.nc` grid description with topology information in the UGRID format.
* `<VAR>.fesom.1948.nc` model output for several variables:
   * `a_ice`: Area of sea ice
   * `salt`: 3D salinity
   * `sst`: Surface temperature
   * `temp`: 3D temperature
   * `u`: 3D zonal velocity
   * `uice`: zonal velocity of sea ice
   * `v`: 3D meridional velocity
   * `vice`: meridional velocity of sea ice
