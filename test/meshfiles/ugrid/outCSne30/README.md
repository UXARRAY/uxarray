# outCSne30 UGRID assets

This folder houses the NE30 cubed-sphere mesh (5400 faces) and a few companion datasets used throughout the test suite.

## Files

- `outCSne30.ug` – canonical NE30 UGRID mesh with 5400 faces (nMesh2_face), max 4 vertices per face, and 5402 nodes. Provides `Mesh2_face_nodes`, `Mesh2_node_x`, and `Mesh2_node_y`.
- `outCSne30_var2.nc` – single-variable dataset (`var2(ncol=5400)`) used for basic integration/zonal workflows.
- `outCSne30_vortex.nc` – single-variable dataset (`psi(ncol=5400)`) containing the barotropic vortex test case.
- `outCSne30_sel_timeseries.nc` – synthetic 6-step hourly time series (`psi(time=6, ncol=5400)`) where each time slice ramps with longitude; input for the `.sel()` regression tests. Regenerate via `python generate_sel_timeseries.py`.
- `generate_sel_timeseries.py` – helper script that rewrites the synthetic time series using `outCSne30_var2.nc` as a template for face count.
