FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

RUN apt-get update 

## Python Set-Up
RUN apt-get install -y \
    python3 python3-venv python3-dev build-essential git 
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

## Onnx Dependencies
RUN pip install onnx
RUN pip install onnxruntime-openvino

## User setup
RUN apt-get install -y sudo \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash customuser
RUN echo "customuser:password" | chpasswd
RUN usermod -aG sudo customuser
WORKDIR /home/customuser

## Clear Cache
RUN rm -rf /var/lib/apt/lists/*

# User set
USER customuser
# USER ubuntu

CMD ["bash"]
