FROM buildpack-deps:bionic@sha256:59661846ab0c581272f4b4688702617e6cc83ef1a9ae1cf918978126babbc858 AS base
LABEL maintainer "https://github.com/weiji14"
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Initiate docker container with user 'jovyan'
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV NB_USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Setup conda
ENV CONDA_DIR ${HOME}/.conda
ENV NB_PYTHON_PREFIX ${CONDA_DIR}
ENV MINICONDA_VERSION 4.7.10

RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "1c945f2b3335c7b2b15130b1b2dc5cf4 *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
    $CONDA_DIR/bin/conda clean --all --quiet --yes && \
    $CONDA_DIR/bin/conda init --verbose

# Setup $HOME directory with correct permissions
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}

# Change to interactive bash shell, so that `conda activate base` works
SHELL ["/bin/bash", "-ic"]

# Install dependencies in environment.yml file using conda
COPY environment.yml ${HOME}
RUN conda env update -n base -f environment.yml && \
    conda clean --all --yes && \
    conda list -n base

# Install dependencies in Pipfile.lock using pipenv
COPY Pipfile* ${HOME}/
RUN conda activate base && \
    export HDF5_DIR=${CONDA_PREFIX} && \
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib && \
    pipenv install --python ${CONDA_PREFIX}/bin/python --dev --deploy && \
    rm --recursive ${HOME}/.cache/pip* && \
    pipenv graph

# Setup DeepBedMap virtual environment properly
RUN conda activate base && \
    pipenv run python -m ipykernel install --user --name deepbedmap && \
    pipenv run jupyter kernelspec list --json

# Copy remaining files to $HOME
COPY --chown=1000:1000 . ${HOME}


FROM base AS app

# Run Jupyter Lab via pipenv in conda environment
EXPOSE 8888
RUN echo -e '#!/bin/bash -i\nset -e\nconda activate\npipenv run "$@"' > .entrypoint.sh && \
    chmod +x .entrypoint.sh
ENTRYPOINT ["./.entrypoint.sh"]
CMD ["jupyter", "lab", "--ip", "0.0.0.0"]
