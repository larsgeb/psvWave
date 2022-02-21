FROM condaforge/miniforge3

RUN apt-get --yes -qq update && \
    apt-get --yes -qq upgrade && \
    DEBIAN_FRONTEND=noninteractive \ 
    apt-get --yes -qq install \
                         build-essential \
                         cmake \
                         g++ \
                         gcc \
                         git \
                         openmpi-bin \
                         zsh && \
    apt-get --yes -qq clean && \
    rm -rf /var/lib/apt/lists/*
SHELL ["/bin/bash", "-c"]

RUN conda init bash
RUN conda init zsh

RUN mkdir /home/psvWave
ADD .  /home/psvWave

RUN conda create -n psvWave python==3.9 && \
    echo "conda activate psvWave" >> $HOME/.zshrc && \
    echo "conda activate psvWave" >> $HOME/.bashrc
SHELL ["conda", "run", "-n", "psvWave", "/bin/bash", "-c"]


RUN cd /home/psvWave && \
    pip install -e .

WORKDIR /home/psvWave/


CMD ["conda", "run", "--no-capture-output", "-n", "psvWave", "jupyter", \
     "notebook", "--notebook-dir=./notebooks", "--ip=0.0.0.0", \
     "--port=8888", "--allow-root", "--NotebookApp.token=", \
     "--no-browser"]
