FROM nvidia/cuda:9.0-base-ubuntu16.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        libcudnn7=7.2.1.38-1+cuda9.0 \
        libnccl2=2.2.13-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        # begin install for opencv
        make \
        cmake \
        gcc \
        g++ \
        wget \
        zlib1g-dev \
        libffi-dev \
        libssl-dev \
        nano \
        ca-certificates \
        libgtk2.0-dev \
        libjpeg-dev libpng-dev \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavresample-dev \
        libswscale-dev \
        libv4l-dev \
        libtbb-dev \
        # end install for opencv
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
        apt-get update && \
        apt-get install libnvinfer4=4.1.2-1+cuda9.0

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip \
    ${PYTHON}-tk

RUN ${PIP} install --upgrade \
    pip \
    setuptools

# begin install for opencv
RUN OPENCV_VERSION="4.0.0" && \
        mkdir -p /tmp/opencv && cd /tmp/opencv && \ 
        wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
        unzip ${OPENCV_VERSION}.zip -d . && \
        mkdir /tmp/opencv/opencv-${OPENCV_VERSION}/build && cd /tmp/opencv/opencv-${OPENCV_VERSION}/build/ && \
        cmake -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D WITH_FFMPEG=ON -D WITH_TBB=ON  .. | tee /tmp/opencv_cmake.log && \
        make -j "$(nproc)" | tee /tmp/opencv_build.log && \
        make install | tee /tmp/opencv_install.log
# end install for opencv

WORKDIR /app
ADD . /app

#write the python packages you want to use in gpu_requirements.txt
RUN ${PIP} install --trusted-host pypi.python.org -r gpu_requirements.txt

RUN mkdir -p /root/.torch/models && wget -O /root/.torch/models/vgg19-dcbb9e9d.pth https://download.pytorch.org/models/vgg19-dcbb9e9d.pth 
