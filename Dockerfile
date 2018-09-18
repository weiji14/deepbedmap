FROM buildpack-deps:bionic@sha256:b1cede2fe7fc26d4e2f59e93e67741d142fb87446d748f0667b1158175e1054f
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
    rm -rf /home/${NB_USER}/.cache/yarn

# Copy files in repository to $HOME
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# Install dependencies using conda and pipenv
WORKDIR ${HOME}
RUN conda env create -n deepbedmap -f environment.yml && \
    conda clean -tipsy && \
    conda list -n deepbedmap

RUN chmod +x postBuild
RUN ./postBuild

# Run Jupyter Lab via pipenv in conda environment
EXPOSE 8888
SHELL ["/bin/bash", "-c"]
CMD source activate deepbedmap && pipenv run jupyter lab --ip 0.0.0.0
