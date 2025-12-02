FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

RUN apt-get update 

## Python Set-Up
RUN apt-get install -y \
    python3 python3-venv python3-dev build-essential git 
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

## Ultralytics Install
RUN pip install ultralytics
ENV PATH="/ultralytics/ultralytics:$PATH"

## Image processing
RUN pip install opencv-python
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN pip install pillow
RUN pip install supervision
 
## Other dependencies
RUN apt-get install -y curl

## Onnx configuration
RUN pip install onnx
RUN pip install onnxruntime
RUN pip install onnxruntime-gpu
RUN pip install onnxslim
RUN pip install onnx-tool

## Customuser setup
RUN apt-get install -y sudo
RUN useradd -m -s /bin/bash customuser
RUN echo "customuser:password" | chpasswd
RUN usermod -aG sudo customuser
WORKDIR /home/customuser

## Clear apt cache
# RUN rm -rf /var/lib/apt/lists/*

## Set user
USER customuser

CMD ["bash"]