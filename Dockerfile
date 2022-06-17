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

# Setup mamba
ENV MAMBA_DIR ${HOME}/.mamba
ENV NB_PYTHON_PREFIX ${MAMBA_DIR}
ENV MAMBAFORGE_VERSION 4.11.0-0

RUN cd /tmp && \
    wget --quiet https://github.com/conda-forge/miniforge/releases/download/${MAMBAFORGE_VERSION}/Mambaforge-${MAMBAFORGE_VERSION}-Linux-x86_64.sh && \
    echo "49268ee30d4418be4de852dda3aa4387f8c95b55a76f43fb1af68dcbf8b205c3 *Mambaforge-${MAMBAFORGE_VERSION}-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash Mambaforge-${MAMBAFORGE_VERSION}-Linux-x86_64.sh -f -b -p $MAMBA_DIR && \
    rm Mambaforge-${MAMBAFORGE_VERSION}-Linux-x86_64.sh && \
    $MAMBA_DIR/bin/mamba clean --all --quiet --yes && \
    $MAMBA_DIR/bin/mamba init --verbose

# Setup $HOME directory with correct permissions
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}

# Change to interactive bash shell, so that `mamba activate base` works
SHELL ["/bin/bash", "-ic"]

# Install dependencies in environment.yml file using mamba
COPY environment.yml ${HOME}
RUN mamba env update -n base -f environment.yml && \
    mamba clean --all --yes && \
    mamba list -n base

# Install dependencies in Pipfile.lock using pipenv
COPY Pipfile* ${HOME}/
RUN mamba activate base && \
    export HDF5_DIR=${CONDA_PREFIX} && \
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib && \
    pipenv install --python ${CONDA_PREFIX}/bin/python --dev --deploy && \
    rm --recursive ${HOME}/.cache/pip* && \
    pipenv graph

# Setup DeepBedMap virtual environment properly
RUN mamba activate base && \
    pipenv run python -m ipykernel install --user --name deepbedmap && \
    pipenv run jupyter kernelspec list --json

# Copy remaining files to $HOME
COPY --chown=1000:1000 . ${HOME}


FROM base AS app

# Run Jupyter Lab via pipenv in mamba environment
EXPOSE 8888
RUN echo -e '#!/bin/bash -i\nset -e\nmamba activate\npipenv run "$@"' > .entrypoint.sh && \
    chmod +x .entrypoint.sh
ENTRYPOINT ["./.entrypoint.sh"]
CMD ["jupyter", "lab", "--ip", "0.0.0.0"]
