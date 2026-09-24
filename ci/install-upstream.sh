#!/usr/bin/env bash
# adapted from https://github.com/pydata/xarray/blob/main/ci/install-upstream-wheels.sh

# forcibly remove packages to avoid artifacts
conda remove -y --force \
    antimeridian \
    cartopy \
    dask \
    datashader \
    distributed \
    matplotlib \
    holoviews \
    hvplot \
    geoviews \
    pandas \
    pyarrow \
    requests \
    scikit-learn \
    scipy \
    shapely \
    spatialpandas \
    xarray

# any packages whose latest versions are being tested here but
# which do not appear during `conda list` from running upstream-dev-ci.yml:
pip uninstall -y \
    pooch

# conda list
conda list

# if available install from scientific-python nightly wheels
# use stable pandas (not nightly) due to geopandas incompatibility with pandas nightly internals
# (see: https://github.com/UXARRAY/uxarray/issues/1414)
python -m pip install \
    'pandas>=2.0.0'

python -m pip install \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --no-deps \
    --pre \
    --upgrade \
    matplotlib \
    pyarrow \
    scikit-learn \
    scipy \
    xarray

# install all remaining packages with --no-deps to avoid dependency conflicts
# (dask/distributed versions may drift, geopandas uses stable release for pandas nightly compatibility)
python -m pip install \
    --no-deps \
    --upgrade \
    git+https://github.com/gadomski/antimeridian.git \
    git+https://github.com/SciTools/cartopy.git \
    git+https://github.com/holoviz/datashader.git \
    git+https://github.com/dask/dask.git \
    git+https://github.com/dask/distributed.git \
    git+https://github.com/fatiando/pooch.git \
    git+https://github.com/pola-rs/polars.git \
    git+https://github.com/holoviz/holoviews.git \
    git+https://github.com/holoviz/hvplot.git \
    git+https://github.com/holoviz/geoviews.git \
    git+https://github.com/psf/requests.git \
    git+https://github.com/shapely/shapely.git \
    git+https://github.com/holoviz/spatialpandas.git \
    'geopandas>=1.0.0'
