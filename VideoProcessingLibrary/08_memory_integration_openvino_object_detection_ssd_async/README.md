# 08_memory_integration_openvino_object_detection_ssd_async

This sample shows how to use oneVPL memory functions to access output pixel data
for integration into user pipelines such as deep learning inference with the
Intel® OpenVINO™ Toolkit for object detection (e.g. face detection).

This sample was validated using Docker so that all prerequisite software is automatically downloaded. This sample demonstrate how media and analytics can
be easily offloaded from CPU to GPU with no code changes. Both Intel® OpenVINO™ Toolkit and Intel® oneAPI Video Processing Library support workloads on CPU or GPU with a simple flag setting. 

This sample also utilizes C++17 for thread-safety.

| Optimized for   | Description
|---------------- | ----------------------------------------
| OS              | Ubuntu* 18.04 (tested); Windows* 10 (not tested)
| Hardware        | Intel® Processor Graphics GEN9 or newer
| Software        | Intel® oneAPI Video Processing Library
|                 | Docker* image Intel® OpenVINO™ Toolkit openvino/ubuntu18_dev:2020.2

## What You Will Learn
- How to integrate Intel® oneAPI Video Processing Library with Intel® OpenVINO™ Toolkit
- How to create and run a Docker image/container with Intel® oneAPI Video Processing Library with Intel® OpenVINO™ Toolkit
- How to create a VPL workstream
- How to use a standard FFmpeg API to demux the video stream and connect to the
  oneVPL input
- How to configure the VPL workstream to set the color format and resolution
- How to create a decode loop with VPL
- How to use oneVPL memory to process each frame with the Intel® OpenVINO™ Toolkit
- How to output the updated raw video stream to the display


## Time to Complete

  ~30 minutes. Intel® oneAPI Video Processing Library and Intel® OpenVINO™ Toolkit components and models are downloaded from Docker Hub* and GitHub*.


## Sample Details

This sample is a command line application that takes an AVI container with an
H.264 stream as an argument, decodes it with the oneVPL decoder, converts the
output to BGRA format with resolution from the AVI container, takes those frames and
performs custom processing using Intel® OpenVINO™ Toolkit inference results for faces detected, and displays the
decoded raw frames to the screen. The printed frame rate is measured over frame decode, VPP processing, and pixel update loop.

| Config            | Default setting
| ----------------- | ----------------------------------
| Target VPL device | GPU,CPU
| Input format      | AVI container with H.264 stream
| Output format     | BGRA
| Output resolution | Resolution of the video

## Build and Run the Sample

To build the 08_ sample you need Docker* and Linux* (validated on Ubuntu* 18 LTS with Docker*).

To install Docker* refer to the following website: https://docs.docker.com/engine/install/ubuntu/

To build the Docker* image from source: docker build -t oneapivpl_openvino:2020.1.beta05_2020.2.120 . -f Dockerfile

To run the demo :  
xhost local:root
docker run --privileged -it --rm -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix oneapivpl_openvino:2020.1.beta05_2020.2.120 ./run-vpl-demo.sh /opt/intel/openvino/vpl/face-demographics-walking-and-pause_h264.avi CPU GPU /opt/intel/openvino/vpl/build/intel/face-detection-retail-0005/FP32/face-detection-retail-0005.xml

The above command starts the demo using run-vpl-demo.sh script. The run-vpl-demo script takes the following args: <H264_VIDEO_FILE.AVI> <MEDIA_DECODE_DEVICE> <INFERENCE_DEVICE> <MODEL>

MEDIA_DECODE_DEVICE options are : GPU or CPU
INFERENCE_DEVICE options are: GPU or CPU
MODEL : full path and name to an Intel® OpenVINO™ Toolkit compatible model such as https://github.com/opencv/open_model_zoo/tree/master/models/intel. Note that this demo only support object detection/recognition models.

I encourage everyone to test different combinations of media and analytics devices and analytics models. For example:

MEDIA_DECODE_DEVICE on the GPU and INFERENCE_DEVICE on the CPU
docker run --privileged -it --rm -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix oneapivpl_openvino:2020.1.beta05_2020.2.120 ./run-vpl-demo.sh /opt/intel/openvino/vpl/face-demographics-walking-and-pause_h264.avi GPU CPU /opt/intel/openvino/vpl/build/intel/face-detection-retail-0005/FP32/face-detection-retail-0005.xml

### Install Prerequisite Software

If you usng Docker* then only Docker is needed, otherwise to run natively then:
 - Intel® oneAPI Base Toolkit for Windows* or Linux* ar
 - [CMake](https://cmake.org)
 - A C/C++ compiler
 - Intel® OpenVINO™ Toolkit
 For native installs refer to all the steps in the Docker file as this will help aid in all that is needed for native installation on Linux* or Windows*.


### Set Up Your Environment

#### Linux

Run `setvars.sh` every time you open a new terminal window:

The `setvars.sh` script can be found in the root folder of your oneAPI
installation, which is typically `/opt/intel/inteloneapi/` when installed as
root or sudo, and `~/intel/inteloneapi/` when installed as a normal user.  If
you customized the installation folder, the `setvars.sh` is in your custom
location.

To use the tools, whether from the command line or using Eclipse, initialize
your environment. To do it in one step for all tools, use the included
environment variable setup utility: `source <install_dir>/setvars.sh`)

```
source <install_dir>/setvars.sh
```


#### Windows

Run `setvars.bat` every time you open a new command prompt:

The `setvars.bat` script can be found in the root folder of your oneAPI
installation, which is typically `C:\Program Files (x86)\inteloneapi\` when
installed using default options. If you customized the installation folder, the
`setvars.bat` is in your custom location.

To use the tools, whether from the command line or using Visual Studio,
initialize your environment. To do it in one step for all tools, use the
included environment variable setup utility: `<install_dir>\setvars.bat`)

```
<install_dir>\setvars.bat
```


### Build the Sample for Native Linux*/Windows*

From the directory containing this README:

```
mkdir build
cd build
cmake ..
cd ..
```


### Run the Sample for Linux*/Windows*

```
cmake --build build --target run
```

The run target runs the sample executable with the argument
`$VPL_DIR/samples/content/cars_1280x720.avi` on Linux and
`%VPL_DIR%\samples\content\cars_1280x720.avi` on Windows.


