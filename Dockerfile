FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# apt
RUN apt-get update \ 
    && apt-get install --yes --no-install-recommends \
    # common
    wget \
    git \
    vim \
    # building things
    build-essential \
    cmake \
    pkg-config \
    # python stuff
    python3-pip \
    python3-setuptools \
    python3-dev \ 
    # Docker language stuff. eg: UTF-8
    locales \
    locales-all \ 
    # needed for openCV python
    libsm6 \
    libxrender-dev \
    libxext6

# pip
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt && rm requirements.txt

# languages
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8     

# user
ARG USER_ID
RUN useradd -ms /bin/bash --no-log-init --password "" --uid ${USER_ID} --user-group user
USER user