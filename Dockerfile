# Base Image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Install dependencies
RUN apt update && \
    pip install transformers && \
    pip install datasets && \
    pip install evaluate && \
    pip install numpy && \
    pip install scikit-learn && \
    pip install pytorch-lightning && \
    pip install wandb && \
    pip install omegaconf && \
    pip install hydra-core && \
    pip install hydra-colorlog && \
    pip install rich && \
    pip install numba && \
    pip install prettytable && \
    pip install webcolors && \
    pip install opencv-python

RUN apt-get -y install git

# # Change working directory
# WORKDIR /workspace
# # Copy project files into workspace
# COPY . .


# ENV PYTHONPATH "${PYTHONPATH}:/workspace"
