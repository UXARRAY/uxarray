#!/usr/bin/env bash
# adapted from https://github.com/pydata/xarray/blob/main/ci/install-upstream-wheels.sh
# forcibly remove packages to avoid artifacts
conda remove -y --force \
    cartopy \
    dask \
    datashader \
    distributed \
    holoviews \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    shapely \
    xarray
pip uninstall -y \
    antimeridian \
    spatialpandas
# conda list
conda list
# if available install from scientific-python nightly wheels
python -m pip install \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --no-deps \
    --pre \
    --upgrade \
    numpy \
    scikit-learn \
    scipy \
    xarray
# install dask and distributed first with --no-deps to avoid version conflicts
python -m pip install \
    --no-deps \
    git+https://github.com/dask/dask.git \
    git+https://github.com/dask/distributed.git

# install geopandas from stable release (compatible with pandas nightly)
python -m pip install \
    'pandas>=2.0.0' \
    'geopandas>=1.0.0'

# install rest from source
python -m pip install \
    git+https://github.com/gadomski/antimeridian.git \
    git+https://github.com/SciTools/cartopy.git \
    git+https://github.com/holoviz/datashader.git \
    git+https://github.com/holoviz/holoviews.git \
    git+https://github.com/shapely/shapely.git \
    git+https://github.com/holoviz/spatialpandas.git
