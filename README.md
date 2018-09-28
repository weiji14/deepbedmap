# DeepBedMap

A flat file data repository for high resolution bed elevation datasets around Antarctica.

## Directory structure

```
  deepbedmap/
    ├── highres/ (contains high resolution localized DEMs)
    │    ├── *.grd/las/txt/... (input vector/raster file containing the data)
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
    │    ├── logs/ (directory for tensorboard log files)
    │    └── train/ (a place to store the model training data)
    │        ├── X_data.npy (highres numpy arrays)
    │        └── Y_data.npy (lowres numpy arrays)
    ├── .env (environment config file used by pipenv, supposedly)
    ├── .<something>ignore (files ignored by a particular piece of software)
    ├── Dockerfile (set of commands to reproduce the software stack here into a docker image)
    ├── LICENSE.md (the license covering this repository)
    ├── Pipfile (what you want, the minimal core dependencies)
    ├── Pipfile.lock (what you need, all the pinned dependencies for full reproducibility)
    ├── README.md (the markdown file you're reading now)
    ├── data_prep.ipynb (jupyter notebook that prepares the data)
    ├── environment.yml (conda packages to install, used by binder)
    ├── postBuild (shell script used by binder after environment.yml packages are installed)
    └── srcnn_train.ipynb (jupyter notebook that trains the Super Resolution ConvNet model)
    
```

# Getting started

## Quickstart

Launch Binder (Interactive jupyter notebook in the cloud).

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/weiji14/deepbedmap/master?urlpath=lab)

## Installation

Start by cloning this [repo-url](/../../)

    git clone <repo-url>

Then I recommend [using conda to install the PDAL binary](https://pdal.io/download.html#conda).
The conda environment will also be created with [pipenv](https://pipenv.readthedocs.io) installed.

    cd deepbedmap
    conda env create -f environment.yml

Once you have the PDAL binary installed, you can install the python libraries using pipenv.
Make sure that `which pipenv` returns something like ~/.conda/envs/deepbedmap/bin/pipenv 

    source activate deepbedmap
    pip install pipenv==2018.7.1
    
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/
    pipenv install --python $CONDA_PREFIX/bin/python
    #or just
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib/ pipenv install --python $CONDA_PREFIX/bin/python

Now you can check to see if all the libraries have been installed

    pipenv graph

### Syncing/Updating to new dependencies

    conda env update -f environment.yml
    pipenv sync

### Common problems

Note that the [.env](https://pipenv.readthedocs.io/en/latest/advanced/#configuration-with-environment-variables) file stores some environment variables.
However, it only works when running `pipenv shell` or `pipenv run <cmd>`.
So after running `source activate deepbedmap`, and you see an `...error while loading shared libraries: libpython3.6m.so.1.0...`, you may need to run this:

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/

## Running jupyter lab

    source activate deepbedmap
    pipenv shell
    
    python -m ipykernel install --user --name deepbedmap  #to install conda env properly
    jupyter kernelspec list --json                        #see if kernel is installed
    jupyter lab &

