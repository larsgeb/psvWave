FROM condaforge/miniforge3
WORKDIR /home

RUN apt-get --yes -qq update \
 && apt-get --yes -qq upgrade \
 && DEBIAN_FRONTEND=noninteractive \ 
 apt-get --yes -qq install \
                      build-essential \
                      cmake \
                      curl \
                      g++ \
                      gcc \
                      gfortran \
                      git \
                      libblas-dev \
                      liblapack-dev \
                      libopenmpi-dev \
                      openmpi-bin \
                      wget \
                      htop \
                      nano \
                      zsh \
 && apt-get --yes -qq clean \
 && rm -rf /var/lib/apt/lists/*
SHELL ["/bin/bash", "-c"]

RUN conda init bash
RUN conda init zsh

RUN conda create -n psvWave python==3.9
RUN echo "conda activate psvWave" >> $HOME/.zshrc
RUN echo "conda activate psvWave" >> $HOME/.bashrc
SHELL ["conda", "run", "-n", "psvWave", "/bin/bash", "-c"]

RUN mkdir /home/psvWave
ADD .  /home/psvWave

RUN cd /home/psvWave && \
    pip install -e .


CMD ["/bin/zsh" ]
