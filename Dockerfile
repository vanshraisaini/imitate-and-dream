FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

# The base image's Python is 3.11, so make a separate 3.12 env
RUN conda create -n app python=3.12 -y
ENV PATH=/opt/conda/envs/app/bin:$PATH

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./datasets /workspace/datasets
COPY ./models /workspace/models
COPY ./scripts /workspace/scripts/
COPY .git/ /workspace/.git/