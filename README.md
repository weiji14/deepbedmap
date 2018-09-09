# DeepBedMap

A flat file data repository for high resolution bed elevation datasets around Antarctica.

# Getting started

## Quickstart

Launch Binder (Interactive jupyter notebook in the cloud).

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/weiji14/deepbedmap/master)

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

### Updating the dependencies

    conda env update -f environment.yml
    pipenv update

## Running jupyter lab

    source activate deepbedmap
    python -m ipykernel install --user  #to install conda env properly
    jupyter kernelspec list --json      #see if kernel is installed
    jupyter lab

