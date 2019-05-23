FROM buildpack-deps:bionic@sha256:59661846ab0c581272f4b4688702617e6cc83ef1a9ae1cf918978126babbc858
LABEL maintainer "https://github.com/weiji14"
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Initiate docker container with user 'jovyan'
ENV NB_USER jovyan
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Setup conda
ENV CONDA_DIR /opt/conda
ENV NB_PYTHON_PREFIX ${CONDA_DIR}
ENV MINICONDA_VERSION 4.6.14
ENV PATH ${CONDA_DIR}/bin:$HOME/.local/bin:${PATH}

RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "718259965f234088d785cad1fbd7de03 *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
    conda clean --all --yes && \
    rm -rf /home/${NB_USER}/.cache/yarn && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate" >> ~/.bashrc

# Setup $HOME directory with correct permissions
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}

# Change to bash shell
SHELL ["/bin/bash", "-c"]

# Install dependencies in environment.yml file using conda
COPY environment.yml ${HOME}
RUN conda env create -n deepbedmap -f environment.yml && \
    conda clean --all --yes && \
    conda list -n deepbedmap

# Install Generic Mapping Tools binary from source
ENV GMT_COMMIT_HASH 20c95aff295964a99f1e738ff371bc7323ce4421
ENV INSTALLDIR ${HOME}/gmt
ENV COASTLINEDIR ${INSTALLDIR}/coast

RUN git clone https://github.com/GenericMappingTools/gmt.git && \
    cd gmt && \
    git checkout ${GMT_COMMIT_HASH}
RUN cd gmt && \
    mkdir -p ${INSTALLDIR} && \
    mkdir -p ${COASTLINEDIR} && \
    bash ci/download-coastlines.sh
USER root
RUN apt-get -qq update && \
    apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        libcurl4-gnutls-dev \
        libnetcdf-dev \
        libgdal-dev \
        libfftw3-dev \
        libpcre3-dev \
        liblapack-dev \
        ghostscript \
        curl && \
    cd gmt && \
    TEST=false bash ci/build-gmt.sh && \
    rm -rf /var/lib/apt/lists/*
USER ${NB_USER}

# Install dependencies in Pipfile.lock using pipenv
COPY Pipfile* ${HOME}/
RUN source activate deepbedmap && \
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib && \
    pipenv install --python $CONDA_PREFIX/bin/python --dev --deploy && \
    rm --recursive ~/.cache/pipenv && \
    pipenv graph

# Copy remaining files to $HOME
COPY --chown=1000:1000 . ${HOME}

# Run Jupyter Lab via pipenv in conda environment
EXPOSE 8888
CMD source activate deepbedmap && pipenv run jupyter lab --ip 0.0.0.0
