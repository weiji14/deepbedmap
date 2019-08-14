# DeepBedMap [[poster]](https://github.com/weiji14/deepbedmap/issues/133)

Going beyond BEDMAP2 using a super resolution deep neural network.
Also a convenient [flat file](https://en.wikipedia.org/wiki/Flat-file_database) data repository for high resolution bed elevation datasets around Antarctica.

![GitHub top language](https://img.shields.io/github/languages/top/weiji14/deepbedmap.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Comet.ML: experiments](https://img.shields.io/badge/Comet.ml-experiments-orange.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAhpJREFUOI2Nk01IVGEUhp/zzfW3sUgllUIEzVApS2JyIwStylYmlC3EWvWzcNGyhdAqiGpZy4ja2UYMrBgoatnCTCNRJBxBpNE0bcbx3rlviyycScF3d+B93nMO33eMPEkiRLEs9BRgdYDzUSKCDTl4bWY5/pwqlJoNBkllmjZevWVjYgrzA37V13LgwjnYUzwH9JlZPL8xodQlSWsPHynJES3VnFTiYLvelcQ0QptGiEl+Vpvqz5kglBoMplZvXsP/8J5I+X5WffiWFMpECANHai5Bx2ScaGPdX/asmY14kgAG088eE0y9IdJYSSif+ek03uJP/KQjRJR3tG+FAZ5IqrZAaowQTK5c2gv7GjEzFhPfsdPXqbnchzmPTHKJsqNN/60NdHtBSK99jkNlGoumMGUoPXaFqhu3/7kKa6q2gwHOuyJHbbY0pOjiLbyWNojOUtHVuxOQr1oHECy8xJ+5T6S1haLOu7gSt9sAXChmS07dofDEAwoarpJd/YorrdwtP+s2Ap6yNIqLHiKYeYEyK6yP3gPC3QQM2+Yzjq5/HGj1Ko4TqahHmR+4gkKIHgZvx2kWgGpnZqQDuovrO7HiMswV4pbjkJ2H9Bikxzf92fyAvpy7yAR/vrKmB6TUuLQ8LCWfS6kxKTUh+Qvaov78NAAkNUv6IklaiUtrn6RsaiuYkHRmK2PbhADE/ICeAo86wAEJYIhtzvk3y+cYpafNe/QAAAAASUVORK5CYII=)](https://www.comet.ml/weiji14/deepbedmap/)
[![Github Actions Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fweiji14%2Fdeepbedmap%2Fbadge&style=flat)](https://github.com/weiji14/deepbedmap/actions)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=weiji14/deepbedmap)](https://dependabot.com)

![DeepBedMap DEM over entire Antarctic continent, EPSG:3031 projection](https://user-images.githubusercontent.com/23487320/60510321-6e0cb280-9ccf-11e9-8096-d2a32eb28e6c.png)

![DeepBedMap Pipeline](https://yuml.me/diagram/scruffy;dir:LR/class/[Data|Highres/Lowres/Misc]->[Preprocessing|data_prep.ipynb],[Preprocessing]->[Model-Training|srgan_train.ipynb],[Model-Training]->[Inference|deepbedmap.ipynb])

<details>
<summary>Directory structure</summary>

```
  deepbedmap/
    ├── features/ (files describing the high level behaviour of various features)
    │    ├── *.feature... (easily understandable specifications written using the Given-When-Then gherkin language)
    │    └── README.md (markdown information on the feature files)
    ├── highres/ (contains high resolution localized DEMs)
    │    ├── *.txt/csv/grd/xyz... (input vector file containing the point-based bed elevation data)
    │    ├── *.json (the pipeline file used to process the xyz point data)
    │    ├── *.nc (output raster netcdf files)
    │    └── README.md (markdown information on highres data sources)
    ├── lowres/ (contains low resolution whole-continent DEMs)
    │    ├── bedmap2_bed.tif (the low resolution DEM!)
    │    └── README.md (markdown information on lowres data sources)
    ├── misc/ (miscellaneous raster datasets)
    │    ├── *.tif (Surface DEMs, Ice Flow Velocity, etc. See list in Issue #9)
    │    └── README.md (markdown information on miscellaneous data sources)
    ├── model/ (*hidden in git, neural network model related files)
    │    ├── train/ (a place to store the raster tile bounds and model training data)
    │    └── weights/ (contains the neural network model's architecture and weights)
    ├── .env (environment variable config file used by pipenv)
    ├── .<something>ignore (files ignored by a particular piece of software)
    ├── .<something else> (stuff to make the code in this repo look and run nicely e.g. linters, CI/CD config files, etc)
    ├── Dockerfile (set of commands to fully reproduce the software stack here into a docker image, used by binder)
    ├── LICENSE.md (the license covering this repository)
    ├── Pipfile (what you want, the summary list of core python dependencies)
    ├── Pipfile.lock (what you need, all the pinned python dependencies for full reproducibility)
    ├── README.md (the markdown file you're reading now)
    ├── data_list.yml (human and machine readable list of the datasets and their metadata)
    ├── data_prep.ipynb/py (paired jupyter notebook/python script that prepares the data)
    ├── deepbedmap.ipynb/py (paired jupyter notebook/python script that predicts an Antarctic bed digital elevation model)
    ├── environment.yml (conda binary packages to install)
    ├── srgan_train.ipynb/py (paired jupyter notebook/python script that trains the ESRGAN neural network model)
    └── test_ipynb.ipynb/py (paired jupyter notebook/python script that runs doctests in the other jupyter notebooks!)
```
</details>

# Getting started

## Quickstart

Launch in [Pangeo Binder](https://pangeo-binder.readthedocs.io) (Interactive jupyter notebook/lab environment in the cloud).

[![Binder](https://binder.pangeo.io/badge_logo.svg)](https://binder.pangeo.io/v2/gh/weiji14/deepbedmap/master)

## Installation

![Installation steps](https://yuml.me/diagram/scruffy/class/[Git|clone-repo]->[Conda|install-binaries-and-pipenv],[Conda]->[Pipenv|install-python-libs])

Start by cloning this [repo-url](/../../)

    git clone <repo-url>

Then I recommend [using conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install the non-python binaries (e.g. GMT, CUDA, etc).
The conda virtual environment will also be created with Python and [pipenv](https://pipenv.readthedocs.io) installed.

    cd deepbedmap
    conda env create -f environment.yml

Activate the conda environment first.

    conda activate deepbedmap

Then set some environment variables **before** using pipenv to install the necessary python libraries,
otherwise you may encounter some problems (see Common problems below).
You may want to ensure that `which pipenv` returns something similar to ~/.conda/envs/deepbedmap/bin/pipenv.

    export HDF5_DIR=$CONDA_PREFIX/
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/
    pipenv install --python $CONDA_PREFIX/bin/python --dev
    #or just
    HDF5_DIR=$CONDA_PREFIX/ LD_LIBRARY_PATH=$CONDA_PREFIX/lib/ pipenv install --python $CONDA_PREFIX/bin/python --dev

Finally, double-check that the libraries have been installed.

    pipenv graph

### Syncing/Updating to new dependencies

    conda env update -f environment.yml
    pipenv sync --dev

### Common problems

Note that the [.env](https://pipenv.readthedocs.io/en/latest/advanced/#configuration-with-environment-variables) file stores some environment variables.
So if you run `conda activate deepbedmap` followed by some other command and get an `...error while loading shared libraries: libpython3.7m.so.1.0...`,
you may need to run `pipenv shell` or do `pipenv run <cmd>` to have those environment variables registered properly.
Or just run this first:

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/

Also, if you get a problem when using `pipenv` to install [netcdf4](https://github.com/Unidata/netcdf4-python), make sure you have done:

    export HDF5_DIR=$CONDA_PREFIX/

and then you can try using `pipenv install` or `pipenv sync` again.
See also this [issue](https://github.com/pydata/xarray/issues/3185#issuecomment-520693149) for more information.

## Running jupyter lab

    conda activate deepbedmap
    pipenv shell

    python -m ipykernel install --user --name deepbedmap  #to install conda env properly
    jupyter kernelspec list --json                        #see if kernel is installed
    jupyter lab &
