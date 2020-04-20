#!/bin/bash

videoToPlay=$1
mediaDevice=$2
inferenceDevice=$3
inferenceModel=$4

if [ ! -f "$1" ]; then
    videoToPlay=face-demographics-walking-and-pause_h264.avi
    echo "File to decode not specified or found. Launching sample video $videoToPlay instead."
fi

if [ -z "$2" ]; then
    mediaDevice=GPU
fi

if [ -z "$3" ]; then
    inferenceDevice=CPU
fi

if [ -z "$4" ]; then
    inferenceModel=/opt/intel/openvino/vpl/build/intel/face-detection-retail-0005/FP32/face-detection-retail-0005.xml
    echo "Inference model not specified. Using: /opt/intel/openvino/vpl/build/intel/face-detection-retail-0005/FP32/face-detection-retail-0005.xml"
fi
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
source /opt/intel/openvino/bin/setupvars.sh > /dev/null 2>&1
cd /opt/intel/openvino/vpl/build/ && ./memory_integration_openvino_object_detection_ssd_async $videoToPlay $mediaDevice $inferenceDevice $inferenceModel

