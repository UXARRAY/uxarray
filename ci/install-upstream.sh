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
    pandas \
    scikit-learn \
    scipy \
    xarray

# install all remaining packages with --no-deps to avoid dependency conflicts
# (dask/distributed versions may drift, geopandas incompatible with pandas nightly internals)
python -m pip install \
    --no-deps \
    --upgrade \
    git+https://github.com/gadomski/antimeridian.git \
    git+https://github.com/SciTools/cartopy.git \
    git+https://github.com/holoviz/datashader.git \
    git+https://github.com/dask/dask.git \
    git+https://github.com/dask/distributed.git \
    git+https://github.com/holoviz/holoviews.git \
    git+https://github.com/shapely/shapely.git \
    git+https://github.com/holoviz/spatialpandas.git \
    'geopandas>=1.0.0'
