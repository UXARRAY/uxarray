#!/usr/bin/env bash
# adapted from https://github.com/pydata/xarray/blob/main/ci/install-upstream-wheels.sh

# forcibly remove packages to avoid artifacts
conda remove -y --force \
    cartopy \
    dask \
    datashader \
    distributed \
    gmpy2 \
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
    geopandas \
    xarray

# install rest from source
python -m pip install \
    git+https://github.com/gadomski/antimeridian.git \
    git+https://github.com/SciTools/cartopy.git \
    git+https://github.com/holoviz/datashader.git \
    git+https://github.com/dask/dask.git \
    git+https://github.com/dask/distributed.git \
    git+https://github.com/aleaxit/gmpy.git \
    git+https://github.com/holoviz/holoviews.git \
    git+https://github.com/shapely/shapely.git \
    git+https://github.com/holoviz/spatialpandas.git
