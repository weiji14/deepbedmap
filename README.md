# DeepBedMap

Going beyond BEDMAP2 using a super resolution deep neural network.
Also a convenient [flat file](https://en.wikipedia.org/wiki/Flat-file_database) data repository for high resolution bed elevation datasets around Antarctica.

![GitHub top language](https://img.shields.io/github/languages/top/weiji14/deepbedmap.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Codefresh build status](https://g.codefresh.io/api/badges/pipeline/weiji14_marketplace/weiji14%2Fdeepbedmap%2Fdeepbedmap?type=cf-1)](https://g.codefresh.io/public/accounts/weiji14_marketplace/pipelines/weiji14/deepbedmap/deepbedmap)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=weiji14/deepbedmap)](https://dependabot.com)

![DeepBedMap Pipeline](https://yuml.me/diagram/scruffy;dir:LR/class/[Data|Highres/Lowres/Misc]->[Preprocessing|data_prep.ipynb],[Preprocessing]->[Model-Training|srgan_train.ipynb],[Model-Training]->[Inference|deepbedmap.ipynb])

<details>
<summary>Directory structure</summary>

```
  deepbedmap/
    ├── features/ (files describing the high level behaviour of various features)
    │    ├── *.feature... (easily understandable specifications written using the Given-When-Then gherkin language)
    │    └── README.md (markdown information on the feature files)
    ├── highres/ (contains high resolution localized DEMs)
    │    ├── *.grd/las/txt/csv... (input vector file containing the point-based data)
    │    ├── *.json (the pdal pipeline file)
    │    ├── *.tif (output raster geotiff file)
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
    ├── .env (environment config file used by pipenv, supposedly)
    ├── .<something>ignore (files ignored by a particular piece of software)
    ├── Dockerfile (set of commands to reproduce the software stack here into a docker image)
    ├── LICENSE.md (the license covering this repository)
    ├── Pipfile (what you want, the minimal core dependencies)
    ├── Pipfile.lock (what you need, all the pinned dependencies for full reproducibility)
    ├── README.md (the markdown file you're reading now)
    ├── data_list.yml (human and machine readable list of the datasets and their metadata)
    ├── data_prep.ipynb (jupyter notebook that prepares the data)
    ├── environment.yml (conda packages to install, used by binder)
    ├── srgan_train.ipynb (jupyter notebook that trains the Super Resolution Generative Adversarial Network model)
    └── test_ipynb.ipynb (jupyter notebook that runs doctests in the other jupyter notebooks!)
```
</details>

# Getting started

## Quickstart

Launch Binder (Interactive jupyter notebook/lab environment in the cloud).

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/weiji14/deepbedmap/master?urlpath=lab)

## Installation

![Installation steps](https://yuml.me/diagram/scruffy/class/[Git|clone-repo]->[Conda|install-binaries-and-pipenv],[Conda]->[Pipenv|install-python-libs])

Start by cloning this [repo-url](/../../)

    git clone <repo-url>

Then I recommend [using conda](https://pdal.io/download.html#conda) to install the non-python binaries (e.g. GMT, CUDA, etc).
The conda virtual environment will also be created with Python and [pipenv](https://pipenv.readthedocs.io) installed.

    cd deepbedmap
    conda env create -f environment.yml

Activate the conda environment first, and then use pipenv to install the necessary python libraries.
Note that `pipenv install` won't work directly (see Common problems below).
You may want to check that `which pipenv` returns something similar to ~/.conda/envs/deepbedmap/bin/pipenv.

    conda activate deepbedmap

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/
    pipenv install --python $CONDA_PREFIX/bin/python
    #or just
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib/ pipenv install --python $CONDA_PREFIX/bin/python

Finally, double-check that the libraries have been installed.

    pipenv graph

### Syncing/Updating to new dependencies

    conda env update -f environment.yml
    pipenv sync

### Common problems

Note that the [.env](https://pipenv.readthedocs.io/en/latest/advanced/#configuration-with-environment-variables) file stores some environment variables.
However, it only works when running `pipenv shell` or `pipenv run <cmd>`.
So after running `conda activate deepbedmap`, and you see an `...error while loading shared libraries: libpython3.6m.so.1.0...`, you may need to run this:

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/

## Running jupyter lab

    conda activate deepbedmap
    pipenv shell

    python -m ipykernel install --user --name deepbedmap  #to install conda env properly
    jupyter kernelspec list --json                        #see if kernel is installed
    jupyter lab &
