"""Generate the synthetic NE30 time series used in sel() regression tests.

Run from repo root: ``python test/meshfiles/ugrid/outCSne30/generate_sel_timeseries.py``
"""

from pathlib import Path

import numpy as np
import xarray as xr

TIME_STEPS = 6

def main() -> None:
    base = Path(__file__).parent

    data_src = base / "outCSne30_var2.nc"
    template = xr.open_dataset(data_src)
    n_face = template.dims["ncol"]
    template.close()

    times = np.arange(TIME_STEPS, dtype=np.int64)
    time_vals = (np.datetime64("2018-04-28T00:00:00") + times * np.timedelta64(1, "h"))
    faces = np.arange(n_face, dtype=np.float32)
    field = (times[:, None].astype(np.float32) + faces[None, :] / 100.0)

    ds = xr.Dataset(
        {"psi": (("time", "ncol"), field)},
        coords={"time": time_vals},
        attrs={"source": "Synthetic field for sel regression"},
    )
    data_path = base / "outCSne30_sel_timeseries.nc"
    ds.to_netcdf(data_path)


if __name__ == "__main__":
    main()
