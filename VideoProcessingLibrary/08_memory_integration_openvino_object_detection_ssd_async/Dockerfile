# docker build -t oneapivpl_openvino:2020.1.beta05_2020.2.120 .
# To run the docker container: 
# xhost local:root
# docker run --privileged -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix oneapivpl_openvino:2020.1.beta05_2020.2.120
# To launch the sample while in the container:
#
# Sample Videos transcoded from - https://github.com/intel-iot-devkit/sample-videos
# Models downloaded - https://github.com/opencv/open_model_zoo/blob/master/models/intel/face-detection-0100/model.yml
# Models downloaded - https://github.com/opencv/open_model_zoo/blob/master/models/intel/face-detection-retail-0005/model.yml
# 

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

FROM intel/oneapi:os-tools-ubuntu18.04 as VPL
RUN apt update && apt install -y ca-certificates wget
ARG url=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
ADD $url /
RUN file=$(basename "$url") && \
    apt-key add "$file" && \
    rm "$file"

# configure the repository
ARG apt_repo=https://apt.repos.intel.com/oneapi
RUN echo "deb $apt_repo all main" > /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update -y

# install Intel(R) oneAPI Base Toolkit
RUN apt-get install -y --no-install-recommends -o=Dpkg::Use-Pty=0 \
intel-oneapi-dev-utilities \
intel-oneapi-python \
intel-oneapi-common-vars \
intel-oneapi-common-licensing \
intel-oneapi-onevpl-devel \
--

# install Intel(R) Graphics Compute Runtime for OpenCL(TM)
RUN echo 'deb [trusted=yes arch=amd64] https://repositories.intel.com/graphics/ubuntu bionic main' \
> /etc/apt/sources.list.d/intel-graphics.list

RUN apt-get update -y && \
apt-get install -y --no-install-recommends -o=Dpkg::Use-Pty=0 \
intel-opencl
# End of Get oneAPI VPL components only

# Build VPL and OpenVINO demo
FROM docker.io/openvino/ubuntu18_dev:2020.2 as OpenVINOBuild
USER root

# OneAPI requires libgtk2 so pull it
RUN apt update && apt install -y libgtk2.0-0 vim wget


# Download OpenVINO models for this testing with this demo
RUN echo "face-detection-0100" > FILE.LST && echo "face-detection-retail-0005" >> FILE.LST 
RUN /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --list ./FILE.LST -o /opt/intel/openvino_2020.2.120/vpl/build

# Download sample videos to test with OneAPI
RUN cd /opt/intel/openvino/vpl/build && wget https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4 
# && wget https://github.com/intel/BaseKit-code-samples/raw/master/VideoProcessingLibrary/content/cars_1280x720.avi

# Copy oneAPI VPL components only (does not include the other cool stuff like vtune/ccl/etc)
COPY --from=VPL /opt/intel/inteloneapi/*.sh /opt/intel/inteloneapi/
COPY --from=VPL /opt/intel/inteloneapi/vpl /opt/intel/inteloneapi/vpl

# Copy need files including a sample video that works. Could not get OneAPI to work with MP4 files downloaded
# so transcoded it to AVI
COPY installer /opt/intel/openvino/vpl
RUN mv /opt/intel/openvino/vpl/face-demographics-walking-and-pause_h264.avi /opt/intel/openvino/vpl/build/face-demographics-walking-and-pause_h264.avi

# Optional: Build all OpenVINO demos if you want
# RUN cd /opt/intel/openvino_2020.2.120/inference_engine/demos/ && ./build_demos.sh

RUN bash -c "source /opt/intel/openvino/bin/setupvars.sh && source /opt/intel/inteloneapi/setvars.sh && cd /opt/intel/openvino_2020.2.120/vpl/build && cmake -DINFERENCE_ENGINE=/opt/intel/openvino/inference_engine/lib/intel64/libinference_engine.so .. && make"
COPY installer/run-vpl-demo.sh /opt/intel/openvino
RUN chmod +x /opt/intel/openvino/run-vpl-demo.sh