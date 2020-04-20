cmake_minimum_required(VERSION 3.5)
project(0memory_integration_openvino_object_detection_ssd_async)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(TARGET memory_integration_openvino_object_detection_ssd_async)
set(SRCS src/memory_integration.cpp)
set(RUNARGS $ENV{VPL_DIR}/samples/content/cars_1280x720.avi)
set(CMAKE_PREFIX_PATH = /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64)
# Set default build type to RelWithDebInfo if not specified
if(NOT CMAKE_BUILD_TYPE)
  message(
    STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE
        STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

message(STATUS "Check for required environment variables")
if(NOT DEFINED ENV{VPLLIB})
  message(FATAL_ERROR "Expected environment variable not defined: VPLLIB")
endif()
if(NOT DEFINED ENV{VPLINCLUDE})
  message(FATAL_ERROR "Expected environment variable not defined: VPLINCLUDE")
endif()

message(STATUS "Check for presence of vpl library")
find_library(VPL_LIB vpl HINTS $ENV{VPLLIB})
if(NOT VPL_LIB)
  message(FATAL_ERROR "vpl library not found. Looked in $ENV{VPLLIB}")
else()
  message(STATUS "Using vpl library: ${VPL_LIB}")
endif()

message(STATUS "Check for presence of vplmemory library")
find_library(VPLMEMORY_LIB vplmemory HINTS $ENV{VPLLIB})
if(NOT VPLMEMORY_LIB)
  message(FATAL_ERROR "vplmemory library not found. Looked in $ENV{VPLLIB}")
else()
  message(STATUS "Using vplmemory library: ${VPLMEMORY_LIB}")
endif()

message(STATUS "Check for presence of FFmpeg libraries")
find_library(AVCODEC_LIB avcodec HINTS $ENV{VPLLIB})
if(NOT AVCODEC_LIB)
  message(FATAL_ERROR "avcodec library not found. Looked in $ENV{VPLLIB}")
else()
  message(STATUS "Using avcodec library: ${AVCODEC_LIB}")
endif()
find_library(AVUTIL_LIB avutil HINTS $ENV{VPLLIB})
if(NOT AVUTIL_LIB)
  message(FATAL_ERROR "avutil library not found. Looked in $ENV{VPLLIB}")
else()
  message(STATUS "Using avutil library: ${AVUTIL_LIB}")
endif()
find_library(AVFORMAT_LIB avformat HINTS $ENV{VPLLIB})
if(NOT AVFORMAT_LIB)
  message(FATAL_ERROR "avformat library not found. Looked in $ENV{VPLLIB}")
else()
  message(STATUS "Using avformat library: ${AVFORMAT_LIB}")
endif()

message(STATUS "Check for presence of OpenCV library")
if(WIN32)
  find_library(VPL_OPENCV_DEBUG_LIB opencv_world opencv_world420d
               HINTS $ENV{VPLLIB})
  find_library(VPL_OPENCV_LIB opencv_world opencv_world420 HINTS $ENV{VPLLIB})
else()
  find_library(VPL_OPENCV_LIB opencv_world opencv_world420 HINTS $ENV{VPLLIB})
  find_library(VPL_OPENCV_DEBUG_LIB opencv_world opencv_world420
               HINTS $ENV{VPLLIB})
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
endif()
if(NOT VPL_OPENCV_LIB OR NOT VPL_OPENCV_DEBUG_LIB)
  message(FATAL_ERROR "opencv_world library not found. Looked in $ENV{VPLLIB}")
else()
  message(STATUS "Using opencv_world library: ${VPL_OPENCV_LIB}")
  message(STATUS "Using opencv_world library(debug): ${VPL_OPENCV_DEBUG_LIB}")
endif()

add_executable(${TARGET} ${SRCS})

include_directories("/opt/intel/openvino/deployment_tools/inference_engine/samples/cpp/common")
include_directories("/opt/intel/openvino/inference_engine/include/")
include_directories("/opt/intel/openvino/deployment_tools/ngraph/include/")

target_include_directories(${TARGET} PRIVATE $ENV{VPLINCLUDE})

link_directories("/opt/intel/openvino/inference_engine/lib/intel64/")
target_link_libraries(${TARGET} ${VPL_LIB} ${VPLMEMORY_LIB} ${AVCODEC_LIB}
                      ${AVUTIL_LIB} ${AVFORMAT_LIB} Threads::Threads ${INFERENCE_ENGINE} /opt/intel/openvino/inference_engine/lib/intel64/libinference_engine_legacy.so
                      /opt/intel/openvino/deployment_tools/ngraph/lib/libngraph.so)
target_link_libraries(${TARGET} debug ${VPL_OPENCV_DEBUG_LIB} Threads::Threads ${INFERENCE_ENGINE} /opt/intel/openvino/inference_engine/lib/intel64/libinference_engine_legacy.so
/opt/intel/openvino/deployment_tools/ngraph/lib/libngraph.so)
target_link_libraries(${TARGET} optimized ${VPL_OPENCV_LIB} Threads::Threads ${INFERENCE_ENGINE} /opt/intel/openvino/inference_engine/lib/intel64/libinference_engine_legacy.so
/opt/intel/openvino/deployment_tools/ngraph/lib/libngraph.so)

add_custom_target(run ${TARGET} ${RUNARGS})
