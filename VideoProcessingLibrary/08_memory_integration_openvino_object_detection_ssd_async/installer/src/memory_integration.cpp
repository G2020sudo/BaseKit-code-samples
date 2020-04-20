/*############################################################################
  # SPDX-License-Identifier: MIT
  #
  # 
  ############################################################################*/
/// Demonstration of Intel® OpenVINO™ Toolkit integration in video pipeline
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <memory>

#include <vplmemory/vplm++.h>
#include <opencv2/opencv.hpp>
#include "vpl/vpl.hpp"

// inference engine related
#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include "samples/ocv_common.hpp"

using namespace InferenceEngine;
// 

extern "C" {
#include <libavformat/avformat.h>
}

#define PROGRAM_NAME "memory_integration"
const int SUCCESS = 0;
const int FAILURE = -1;

void LogTrace(const char* fmt, ...);
void DisplayFrame(cv::Mat& img_bgra);
int DecodeAndRenderFile(const char* mediaDevice, const char* filename, InferRequest::Ptr& async_infer_request, 
  std::string& imageInputName, std::string& outputName,
  const int maxProposalCount, const int objectSize);
void UpdateImageWithInference(vplm_mem* bgraImage, InferRequest::Ptr& async_infer_request, std::string& imageInputName, std::string& outputName,
  const int maxProposalCount, const int objectSize, const size_t frameWidth, const size_t frameHeight,
  cv::Mat& img_bgra);
void PerformInference(cv::Mat& frame, InferRequest::Ptr& async_infer_request, std::string& imageInputName, std::string& outputName,
  const int maxProposalCount, const int objectSize, const size_t frameWidth, 
  const size_t frameHeight);
void FrameToBlob(const cv::Mat& frame, InferRequest::Ptr& inferRequest, const std::string& inputName);
void PrintUsage(FILE* stream);

/// Simple timer that tracks total time elapsed between starts and stops
class Timer {
 public:
  Timer() : elapsed_time_(elapsed_time_.zero()) {}
  void Start() { start_time_ = std::chrono::system_clock::now(); }
  void Stop() {
    stop_time_ = std::chrono::system_clock::now();
    elapsed_time_ += (stop_time_ - start_time_);
  }
  double Elapsed() const { return elapsed_time_.count(); }

 private:
  std::chrono::system_clock::time_point start_time_, stop_time_;
  std::chrono::duration<double> elapsed_time_;
};

/// Simple structure for drawing inference results
struct ObjectDetected {
  float confidence;
  std::string label;
  float xmin;
  float ymin;
  float xmax;
  float ymax;
};

/// Manage infernece state so we can perform in asynchronously
enum InferenceState {
  INFERENCE_UNINITIALIZED,
  INFERENCE_INITIALIZED,
  INFERENCE_PERFORMING,
  INFERENCE_COMPLETED
};

/// C++17 shared_mutex for managing producer/consumer threads of analytics results
std::shared_mutex _mtx;
std::vector<ObjectDetected> _inferenceResults;
InferenceState _inferenceState = INFERENCE_UNINITIALIZED;

/// Program entry point
int main(int argc, char* argv[]) {
  if (argc < 5) {
    fprintf(stderr, "%s: missing file operand\n", PROGRAM_NAME);
    PrintUsage(stderr);
    return FAILURE;
  }

  LogTrace("Inference Engine initializing");
  Core ie;
  
  auto cnnNetwork = ie.ReadNetwork(argv[4]);
  cnnNetwork.setBatchSize(1); /** Set batch size to 1 **/
  std::vector<std::string> labels;
  InputsDataMap inputInfo(cnnNetwork.getInputsInfo());

  std::string imageInputName, imageInfoInputName;
  size_t netInputHeight, netInputWidth;

  /** SSD-based network should have one input and one output **/
  // ---------------------------  Configure input & output ---------------------------------------------
  // --------------------------- Prepare input blobs -----------------------------------------------------
  // Checking that the inputs are as the demo expects
  for (const auto & inputInfoItem : inputInfo) {
      if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // first input contains images
          imageInputName = inputInfoItem.first;
          inputInfoItem.second->setPrecision(Precision::U8);
          inputInfoItem.second->getInputData()->setLayout(Layout::NCHW);
          const TensorDesc& inputDesc = inputInfoItem.second->getTensorDesc();
          netInputHeight = getTensorHeight(inputDesc);
          netInputWidth = getTensorWidth(inputDesc);
      } else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // second input contains image info
          imageInfoInputName = inputInfoItem.first;
          inputInfoItem.second->setPrecision(Precision::FP32);
      } else {
          throw std::logic_error("Unsupported " +
            std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()) + "D "
            "input layer '" + inputInfoItem.first + "'. "
            "Only 2D and 4D input layers are supported");
      }
  }
   
  // --------------------------- Prepare output blobs -----------------------------------------------------
  // Checking that the outputs are as the demo expects
  OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
  if (outputInfo.size() != 1) {
      throw std::logic_error("This demo accepts networks having only one output");
  }
  DataPtr& output = outputInfo.begin()->second;
  auto outputName = outputInfo.begin()->first;

  int num_classes = 0;

  if (auto ngraphFunction = cnnNetwork.getFunction()) {
      for (const auto op : ngraphFunction->get_ops()) {
          if (op->get_friendly_name() == outputName) {
              auto detOutput = std::dynamic_pointer_cast<ngraph::op::DetectionOutput>(op);
              if (!detOutput) {
                  THROW_IE_EXCEPTION << "Object Detection network output layer(" + op->get_friendly_name() +
                      ") should be DetectionOutput, but was " +  op->get_type_info().name;
              }

              num_classes = detOutput->get_attrs().num_classes;
              break;
          }
      }
  } else {
      const CNNLayerPtr outputLayer = cnnNetwork.getLayerByName(outputName.c_str());
      if (outputLayer->type != "DetectionOutput") {
          throw std::logic_error("Object Detection network output layer(" + outputLayer->name +
                                  ") should be DetectionOutput, but was " +  outputLayer->type);
      }

      num_classes = outputLayer->GetParamAsInt("num_classes");
      LogTrace("classes:%d",num_classes);
  }

  if (static_cast<int>(labels.size()) != num_classes) {
      if (static_cast<int>(labels.size()) == (num_classes - 1))  // if network assumes default "background" class, having no label
          labels.insert(labels.begin(), "fake");
      else
          labels.clear();
  }
  const SizeVector outputDims = output->getTensorDesc().getDims();
  const int maxProposalCount = outputDims[2];
  const int objectSize = outputDims[3];
  if (objectSize != 7) {
      throw std::logic_error("Output should have 7 as a last dimension");
  }
  if (outputDims.size() != 4) {
      throw std::logic_error("Incorrect output dimensions for SSD");
  }
  output->setPrecision(Precision::FP32);
  output->setLayout(Layout::NCHW);
  //-----------------------------------------------------------------------------------------------
  ExecutableNetwork network = ie.LoadNetwork(cnnNetwork, argv[3]); // use GPU for inference
  InferRequest::Ptr async_infer_request = network.CreateInferRequestPtr();
  /* it's enough just to set image info input (if used in the model) only once */
  if (!imageInfoInputName.empty()) {
      auto setImgInfoBlob = [&](const InferRequest::Ptr &inferReq) {
          auto blob = inferReq->GetBlob(imageInfoInputName);
          auto data = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
          data[0] = static_cast<float>(netInputHeight);  // height
          data[1] = static_cast<float>(netInputWidth);  // width
          data[2] = 1;
      };
      LogTrace("InfoBlob set");
      setImgInfoBlob(async_infer_request);
  }
  _inferenceState == INFERENCE_INITIALIZED;
  //--------------done with inference setup----------------

  FILE* input_stream = fopen(argv[1], "rb");
  if (!input_stream) {
    fprintf(stderr, "%s: could not open input file '%s'\n", PROGRAM_NAME,
            argv[1]);
    return FAILURE;
  }
  fclose(input_stream);
  
  int status = DecodeAndRenderFile(argv[2], argv[1], async_infer_request, imageInputName, outputName,
                        maxProposalCount, objectSize);
  return status;
}

////////////////////////////////////////////////////////////////////////////////
/// Main decode and render function
////////////////////////////////////////////////////////////////////////////////
int DecodeAndRenderFile(const char* mediaDevice, const char* filename, InferRequest::Ptr& async_infer_request, std::string& imageInputName, std::string& outputName,
                        const int maxProposalCount, const int objectSize) {
  int status = FAILURE;
  int avsts;
  LogTrace("Creating H.264 decoder using  device specified (GPU only if available and specified)");
  
  std::string strMediaDevice(mediaDevice);
  
  vpl::Workstream* decoder;
  if (strMediaDevice == "CPU")
    decoder = new vpl::Workstream(VPL_TARGET_DEVICE_CPU, VPL_WORKSTREAM_DECODEVIDEOPROC); 
  else
    decoder = new vpl::Workstream(VPL_TARGET_DEVICE_DEFAULT, VPL_WORKSTREAM_DECODEVIDEOPROC); 
    
  decoder->SetConfig(VPL_PROP_SRC_BITSTREAM_FORMAT, VPL_FOURCC_H264);

  LogTrace("Setting target format and color-space (CSC).");
  decoder->SetConfig(VPL_PROP_DST_RAW_FORMAT, VPL_FOURCC_BGRA);  // SSD Model Uses BGR  

  LogTrace("Creating and initialize demux context.");
  AVFormatContext* fmt_ctx = NULL;
  avsts = avformat_open_input(&fmt_ctx, filename, NULL, NULL);
  if (0 != avsts) {
    fprintf(stderr, "Could not open input file '%s'\n", filename);
    return FAILURE;
  }

  LogTrace("Selecting video stream from demux outputs.");
  avformat_find_stream_info(fmt_ctx, NULL);
  int stream_index =
      av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0); 
  LogTrace("stream_index %d.", stream_index);

  // get frame width/height if available
  int frameWidth = 352;
  int frameHeight = 288;
  if (fmt_ctx->streams[stream_index]->codec)
  {
    frameWidth = fmt_ctx->streams[stream_index]->codec->width;
    frameHeight = fmt_ctx->streams[stream_index]->codec->height;
  }

  LogTrace("Setting target resolution (scaling): %dx%d", frameWidth, frameHeight);
  VplVideoSurfaceResolution output_size = {frameWidth, frameHeight};
  decoder->SetConfig(VPL_PROP_OUTPUT_RESOLUTION, output_size);

  AVPacket pkt = {0};
  av_init_packet(&pkt);

  size_t frame_count = 0;
  Timer timer;

  bool decode_done = false;
  LogTrace("Entering main decode loop");
  while (!decode_done) {
    vplm_mem* image = nullptr;
    size_t bytes_read = 0;

    switch (decoder->GetState()) {
      case VPL_STATE_READ_INPUT:
        // The decoder can accept more data, read it from file and pass it in.
        timer.Start();        
        avsts = av_read_frame(fmt_ctx, &pkt);
        if (avsts >= 0) {
          if (pkt.stream_index == stream_index) {
            image = decoder->DecodeProcessFrame(pkt.data, pkt.size);            
          }
        } else {
          image = decoder->DecodeFrame(nullptr, 0);
        }
        timer.Stop();
        break;

      case VPL_STATE_INPUT_BUFFER_FULL:
        // The decoder cannot accept more data, call DecodeFrame to drain.
        timer.Start();
        image = decoder->DecodeFrame(nullptr, 0);
        timer.Stop();
        break;

      case VPL_STATE_END_OF_OPERATION:
        // The decoder has completed operation, and has no frames left to give.
        LogTrace("Decode complete");
        decode_done = true;
        status = SUCCESS;
        break;

      case VPL_STATE_ERROR:
        LogTrace("Error during decode. Exiting.");
        decode_done = true;
        status = FAILURE;
        break;
    }

    if (image) {
      cv::Mat img_bgra;
      UpdateImageWithInference(image, async_infer_request, imageInputName, outputName,
                               maxProposalCount, objectSize, frameWidth, frameHeight, img_bgra);
      // DecodeFrame returned a frame, use it.
      frame_count++;
      fprintf(stderr, "Frame: %zu\r", frame_count);
      DisplayFrame(img_bgra);
      // Release the reference to the frame, so the memory can be reclaimed
      vplm_unref(image);
    }
  }
  LogTrace("Close demux context input file.");
  avformat_close_input(&fmt_ctx);

  LogTrace("Frames decoded   : %zu", frame_count);
  LogTrace("Frames per second: %02f", frame_count / timer.Elapsed());

  delete decoder;
  decoder = NULL;
  return status;
}

////////////////////////////////////////////////////////////////////////////////
/// Update pixel data with inference results of 80% confidence if available
////////////////////////////////////////////////////////////////////////////////
void UpdateImageWithInference(vplm_mem* bgraImage, InferRequest::Ptr& async_infer_request, std::string& imageInputName, std::string& outputName,
 const int maxProposalCount, const int objectSize, const size_t frameWidth, const size_t frameHeight,
 cv::Mat& img_bgra) {
  
  vplm_cpu_image handle = {0};
  vplm_image_info desc;
  vplm_get_image_info(bgraImage, &desc);
  vplm_status err = vplm_map_image(bgraImage, VPLM_ACCESS_MODE_READWRITE, &handle);
  unsigned char* data = new unsigned char[desc.height * desc.width * 4];
  size_t pitch0 = handle.planes[0].stride;
  for (size_t y = 0; y < desc.height; y++) {
    memcpy(data + (desc.width * 4 * y), handle.planes[0].data + (pitch0 * y),
           desc.width * 4);
    data[desc.width * 4 *y] = 0;
  }
  
  img_bgra = cv::Mat(desc.height, desc.width, CV_8UC4, data);  

  // OpenVINO doesn't need alpha channel
  cv::cvtColor(img_bgra, img_bgra, cv::COLOR_BGRA2BGR);
 
  if (_inferenceState == INFERENCE_COMPLETED)
  {
      // Processing results inference request      
      const float *detections = async_infer_request->GetBlob(outputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
      
      for (int i = 0; i < maxProposalCount; i++) 
      {
          float image_id = detections[i * objectSize + 0];
          if (image_id < 0) {
              break;
          }

          float confidence = detections[i * objectSize + 2];

          if (confidence > 0.8) { // use 80% confidence
              // Update pixels with inference data if available. Will block when new inference data
              // is being saved  
              ObjectDetected inferenceResult;
              inferenceResult.confidence = detections[i * objectSize + 2];
              inferenceResult.label = static_cast<int>(detections[i * objectSize + 1]);
              inferenceResult.xmin = detections[i * objectSize + 3] * frameWidth;
              inferenceResult.ymin = detections[i * objectSize + 4] * frameHeight;
              inferenceResult.xmax = detections[i * objectSize + 5] * frameWidth;
              inferenceResult.ymax = detections[i * objectSize + 6] * frameHeight;
              
              cv::rectangle(img_bgra, cv::Point2f(inferenceResult.xmin, inferenceResult.ymin), 
                            cv::Point2f(inferenceResult.xmax, inferenceResult.ymax), 
                            cv::Scalar(0, 0, 255));              
          }
          _inferenceState = INFERENCE_INITIALIZED;
      } // endfor
  }
  
  
  PerformInference(img_bgra, async_infer_request, imageInputName, outputName,
                   maxProposalCount, objectSize, frameWidth, frameHeight);
  
  delete data;  
  data = NULL;
  vplm_unmap_image(&handle);
}

////////////////////////////////////////////////////////////////////////////////
/// Start inference on another thread/background
////////////////////////////////////////////////////////////////////////////////
void PerformInference(cv::Mat& frame, InferRequest::Ptr& async_infer_request, std::string& imageInputName, std::string& outputName,
                      const int maxProposalCount, const int objectSize, const size_t frameWidth, 
                      const size_t frameHeight)
{
  {
    std::shared_lock lock(_mtx);
    if (_inferenceState == INFERENCE_PERFORMING)
    {
      return; // let last inference call finish
    }
  }
  
  FrameToBlob(frame, async_infer_request, imageInputName);

  async_infer_request->SetCompletionCallback([&]
  {    
      std::unique_lock lock(_mtx);      
      _inferenceState = INFERENCE_COMPLETED;
  });

  std::unique_lock lock(_mtx);     
  if (_inferenceState != INFERENCE_COMPLETED) 
  {
    async_infer_request->StartAsync();
    _inferenceState = INFERENCE_PERFORMING;  
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Prepare input for inference engine
////////////////////////////////////////////////////////////////////////////////
void FrameToBlob(const cv::Mat& frame, InferRequest::Ptr& inferRequest, const std::string& inputName) {
    /* Resize and copy data from the image to the input blob */
    Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);    
    matU8ToBlob<uint8_t>(frame, frameBlob);
}
/// Print command line usage
void PrintUsage(FILE* stream) {
  fprintf(stream, "Usage: %s FILE MEDIA_DECODE_DEVICE INFERENCE_DEVICE MODEL\n\n", PROGRAM_NAME);
  fprintf(stream,
          "Demux and decode FILE using Intel(R) oneAPI Video Processing "
          "Library.\n\n"
          "Then directly manipulate the decoded frame in code"
          "Demux is done using 3rd party library.\n\n"
          "FILE must be in H264 format\n\n"
          "Example:\n"
          "  %s %s\n",
          PROGRAM_NAME, "content/cars_1280x720.avi");
}

/// Render frame to display
void DisplayFrame(cv::Mat& img_bgra) {
  //cv::Mat img_nv12, img_bgra;
  bool have_display = true;
  static bool first_call = true;
#ifdef __linux__
  const char* display = getenv("DISPLAY");
  if (!display) {
    if (first_call) LogTrace("Display unavailable, continuing without...");
    have_display = false;
  }
#endif

  if (have_display) cv::imshow("OneAPI VPL Decoding With OpenVINO Analytics", img_bgra);
  cv::waitKey(24); // let window paint
}

/// Print message to stderr
void LogTrace(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  va_end(args);
}
