FROM buildpack-deps:bionic-scm@sha256:f37982278d0dfd71d282ee551a927a44294876d07b98ea9c001087282e482817
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
ENV MINICONDA_VERSION 4.5.11
ENV PATH ${CONDA_DIR}/bin:$HOME/.local/bin:${PATH}

RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "e1045ee415162f944b6aebfe560b8fee *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
    conda clean -tipsy && \
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
    conda clean -tipsy && \
    conda list -n deepbedmap

# Install dependencies in Pipfile.lock using pipenv
COPY Pipfile* ${HOME}/
RUN source activate deepbedmap && \
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib && \
    pipenv install --python $CONDA_PREFIX/bin/python && \
    rm --recursive ~/.cache/pipenv && \
    pipenv graph

# Copy remaining files to $HOME
COPY --chown=1000:1000 . ${HOME}

# Run Jupyter Lab via pipenv in conda environment
EXPOSE 8888
CMD source activate deepbedmap && pipenv run jupyter lab --ip 0.0.0.0
