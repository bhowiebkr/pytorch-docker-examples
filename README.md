# PyTorch docker examples

Various ML examples using pytorch. 


# Install
1. install [Docker 19.03](https://docs.docker.com/install/linux/docker-ce/ubuntu/) for Ubuntu
1. install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

# How to use:
1. run **`build`** to build the image.
1. run **`run`** to run the the current main python under app. Calls the wrapper.py to launch docker

## Config Settings
- See **`config.ini`**. Make sure to give **DockerTag** a new name.
- For debugging stuff inside the running container turn on **Interactive** to be **true** in the config file for interactive mode.