# Base Image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Install dependencies
RUN apt update && \
    pip install numpy \
    pip install transformers==4.24 \
    pip install datasets \
    pip install evaluate \
    pip install click \
    pip install wandb \
    pip install scikit-learn

# Change working directory
WORKDIR /workspace
# Copy project files into workspace
COPY . .


# ENV PYTHONPATH "${PYTHONPATH}:/workspace"
