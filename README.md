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
    ├── model/ (neural network model related files)
    │    └── train
    │        │── X_data.npy (highres numpy arrays)
    │        └── Y_data.npy (lowres numpy arrays)
    ├── .<something>ignore (files ignored by a particular piece of software)
    ├── environment.yml (conda packages to install, used by binder)
    ├── LICENSE.md (the license covering this repository)
    ├── Pipfile (what you want, the minimal core dependencies)
    ├── Pipfile.lock (what you need, all the pinned dependencies for full reproducibility)
    ├── postBuild (shell script used by binder after environment.yml packages are installed)
    └── README.md (the markdown file you're reading now)
    
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
Also note that the [.env](https://pipenv.readthedocs.io/en/latest/advanced/#configuration-with-environment-variables) file stores some environment variables.

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

## Running jupyter lab

    source activate deepbedmap
    pipenv shell
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/
    
    python -m ipykernel install --user --name deepbedmap  #to install conda env properly
    jupyter kernelspec list --json                        #see if kernel is installed
    jupyter lab &

