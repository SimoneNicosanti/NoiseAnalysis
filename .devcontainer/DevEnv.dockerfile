#trunk-ignore-all(git-diff-check/error)
#trunk-ignore-all(hadolint)

## Version Table of TensorRT Ready Containers @ https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/index.html
## Version 25.08 is for CUDA 13.x with Ubuntu 24.04 LTS
## If the host driver is different just change the base image accordingly
FROM nvcr.io/nvidia/tensorrt:25.08-py3

## Setting up cuDNN
ENV CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}"

## TensorRT Set-Up
RUN pip install tensorrt

## Model-Optimizer Installer
## Library to apply different optimizations for DL Models (Quantization, Pruning, etc...)
RUN pip install nvidia-modelopt[onnx]

## For Cuda13.x and Python 14.x Compatibility
## Current stable of onnxruntime does not support CUDA 13.x
## Installing nightly build to support CUDA 13.x (Reference @ https://github.com/microsoft/onnxruntime/issues/26547)
## If base driver is different, just install the stable version of onnxruntime and onnxruntime-gpu or just leave the version installed with nvidia-modelopt[onnx]
RUN pip uninstall onnxruntime onnxruntime-gpu -y
RUN pip install flatbuffers numpy packaging protobuf sympy coloredlogs
RUN pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu --no-deps

## Ultralytics Set-Up
RUN pip install ultralytics
ENV PATH="/ultralytics/ultralytics:$PATH"

## Running apt-get update
RUN apt-get update 
 
## Image processing
RUN pip install opencv-python
RUN apt-get install -y libglx-mesa0
RUN apt-get install -y libglib2.0-0
RUN pip install pillow
RUN pip install supervision

## Onnx configuration
RUN pip install onnx
RUN pip install onnxslim
RUN pip install onnx-tool

## Other Python Packages
RUN pip install pandas
RUN pip install matplotlib
RUN pip install seaborn

## importhook for dynamic code patching
RUN pip install importhook

## Argcomplete for command line auto-completion
RUN pip install argcomplete
RUN activate-global-python-argcomplete

## Other dependencies
RUN apt-get install -y curl

# Customuser setup
RUN useradd -m -s /bin/bash customuser
RUN apt-get install -y sudo
RUN echo "customuser:password" | chpasswd
RUN usermod -aG sudo customuser

# ## Clear apt cache
# USER root
# RUN rm -rf /var/lib/apt/lists/*

## Set user
# USER customuser

CMD ["bash"]