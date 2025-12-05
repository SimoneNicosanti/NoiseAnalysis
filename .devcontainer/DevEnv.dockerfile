FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

RUN apt-get update 

## Python Set-Up
RUN apt-get install -y \
    python3 python3-venv python3-dev build-essential git 

## Customuser setup
RUN useradd -m -s /bin/bash customuser
RUN apt-get install -y sudo
RUN echo "customuser:password" | chpasswd
RUN usermod -aG sudo customuser

## Environment Set-Up
## Creating as customuser to use pip in the container
USER customuser
RUN python3 -m venv /home/customuser/venv
ENV PATH="/home/customuser/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel
USER root

## TensorRT Set-Up
RUN pip install tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12
ENV LD_LIBRARY_PATH="/home/customuser/venv/lib/python3.10/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}"

## Ultralytics Set-Up
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

## Other Python Packages
RUN pip install pandas
RUN pip install matplotlib


## Clear apt cache
# RUN rm -rf /var/lib/apt/lists/*

## Set user
USER customuser

CMD ["bash"]