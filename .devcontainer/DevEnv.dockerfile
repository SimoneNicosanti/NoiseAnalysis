FROM python:3.12-slim-bookworm

RUN apt update

## Ultralytics Install
# RUN pip install ultralytics
# ENV PATH="/ultralytics/ultralytics:$PATH"
# RUN apt-get install -y libgl1-mesa-glx

## Onnx configuration
RUN pip install onnx
RUN pip install onnxruntime

## Image processing
RUN pip install opencv-python
RUN apt install -y libgl1-mesa-glx
RUN apt install -y libglib2.0-0
RUN pip install pillow
RUN pip install supervision

## Other dependencies
RUN apt install -y curl
RUN apt install -y git

## Customuser setup
RUN apt install -y sudo
RUN useradd -m -s /bin/bash customuser
RUN echo "customuser:password" | chpasswd
RUN usermod -aG sudo customuser
WORKDIR /home/customuser

## Clear apt cache
# RUN rm -rf /var/lib/apt/lists/*

## Set user
USER customuser

CMD ["bash"]