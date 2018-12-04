// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/servables/tensorrt/plan_bundle.h"

#include <NvInfer.h>
#include <NvOnnxParserRuntime.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/server_status.h"
#include "src/core/utils.h"
#include "src/servables/tensorrt/logging.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/io/path.h"

namespace nvidia { namespace inferenceserver {

namespace {

uint64_t
GetSize(const int max_batch_size, const DataType& dtype, const DimsList& dims)
{
  size_t dt_size = nvidia::inferenceserver::GetSize(dtype, dims);
  return std::max(1, max_batch_size) * dt_size;
}

DataType
ConvertDatatype(nvinfer1::DataType trt_type)
{
  switch (trt_type) {
    case nvinfer1::DataType::kFLOAT:
      return TYPE_FP32;
    case nvinfer1::DataType::kHALF:
      return TYPE_FP16;
    case nvinfer1::DataType::kINT8:
      return TYPE_INT8;
    case nvinfer1::DataType::kINT32:
      return TYPE_INT32;
  }

  return TYPE_INVALID;
}

bool
CompareDims(const nvinfer1::Dims& model_dims, const DimsList& dims)
{
  if (model_dims.nbDims != dims.size()) {
    return false;
  }

  for (int i = 0; i < model_dims.nbDims; ++i) {
    if (model_dims.d[i] != dims[i]) {
      return false;
    }
  }

  return true;
}

const std::string
DimsDebugString(const DimsList& dims)
{
  bool first = true;
  std::string str;
  str.append("[");
  for (int i = 0; i < dims.size(); ++i) {
    if (!first) {
      str.append(",");
    }
    str.append(std::to_string(dims[i]));
    first = false;
  }
  str.append("]");
  return str;
}

const std::string
DimsDebugString(const nvinfer1::Dims& dims)
{
  bool first = true;
  std::string str;
  str.append("[");
  for (int i = 0; i < dims.nbDims; ++i) {
    if (!first) {
      str.append(",");
    }
    str.append(std::to_string(dims.d[i]));
    first = false;
  }
  str.append("]");
  return str;
}

}  // namespace


PlanBundle::Context::Context(
  const std::string& name, const int gpu_device, const int max_batch_size)
    : name_(name), gpu_device_(gpu_device), max_batch_size_(max_batch_size),
      runtime_(nullptr), engine_(nullptr), context_(nullptr), num_inputs_(0),
      byte_sizes_(nullptr), buffers_(nullptr), stream_(nullptr)
{
}

PlanBundle::Context::Context(Context&& o)
    : name_(std::move(o.name_)), gpu_device_(o.gpu_device_),
      max_batch_size_(o.max_batch_size_), runtime_(o.runtime_),
      engine_(o.engine_), context_(o.context_), num_inputs_(o.num_inputs_),
      byte_sizes_(o.byte_sizes_), buffers_(o.buffers_), stream_(o.stream_)
{
  o.runtime_ = nullptr;
  o.engine_ = nullptr;
  o.context_ = nullptr;
  o.num_inputs_ = 0;
  o.byte_sizes_ = nullptr;
  o.buffers_ = nullptr;
  o.stream_ = nullptr;
}

PlanBundle::Context::~Context()
{
  LOG_VERBOSE(1) << "~PlanBundle::Context ";

  if (byte_sizes_ != nullptr) {
    delete[] byte_sizes_;
    byte_sizes_ = nullptr;
  }
  if (buffers_ != nullptr) {
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
      if (buffers_[i] != nullptr) {
        cudaError_t err = cudaFree(buffers_[i]);
        if (err != cudaSuccess) {
          LOG_ERROR << "Failed to free cuda memory for '" << name_
                    << "': " << cudaGetErrorString(err);
        }
      }
    }

    delete[] buffers_;
    buffers_ = nullptr;
  }

  if (stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda stream: " << cudaGetErrorString(err);
    }
    stream_ = nullptr;
  }

  if (context_ != nullptr) {
    context_->destroy();
    context_ = nullptr;
  }
  if (engine_ != nullptr) {
    engine_->destroy();
    engine_ = nullptr;
  }
  if (runtime_ != nullptr) {
    runtime_->destroy();
    runtime_ = nullptr;
  }
}

tensorflow::Status
PlanBundle::Init(const tensorflow::StringPiece& path, const ModelConfig& config)
{
  TF_RETURN_IF_ERROR(ValidateModelConfig(config, kTensorRTPlanPlatform));
  TF_RETURN_IF_ERROR(SetModelConfig(path, config));

  // Initialize the datatype map and label provider for each output
  const auto model_dir = tensorflow::io::Dirname(path);
  for (const auto& io : config.output()) {
    output_dtype_map_.insert(std::make_pair(io.name(), io.data_type()));

    if (!io.label_filename().empty()) {
      const auto label_path =
        tensorflow::io::JoinPath(model_dir, io.label_filename());
      TF_RETURN_IF_ERROR(label_provider_.AddLabels(io.name(), label_path));
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
PlanBundle::CreateExecutionContexts(
  const std::unordered_map<std::string, std::vector<char>>& models)
{
  // TensorRT engine creation is not thread-safe, so multiple creations
  // are serialized with a global lock.
  static std::mutex global_context_mu;
  std::lock_guard<std::mutex> glock(global_context_mu);

  uint32_t total_context_cnt = 0;

  // Create a runtime/engine/context trifecta for each instance.
  //
  // TODO [DLIS-14] This can be optimized by sharing a runtime (across
  // all instances?), and sharing an engine across instances that have
  // access to the same GPU.
  for (const auto& group : Config().instance_group()) {
    // TensorRT requires that every context have a GPU.
    if (
      (group.kind() != ModelInstanceGroup::KIND_GPU) ||
      (group.gpus().size() == 0)) {
      return tensorflow::errors::InvalidArgument(
        "instance group ", group.name(), " of model ", Name(),
        " must be KIND_GPU and must specify at least on GPU id");
    }

    for (int c = 0; c < group.count(); c++) {
      for (int gpu_device : group.gpus()) {
        const std::string instance_name = group.name() + "_" +
                                          std::to_string(c) + "_gpu" +
                                          std::to_string(gpu_device);
        TF_RETURN_IF_ERROR(
          CreateExecutionContext(instance_name, gpu_device, models));
        total_context_cnt++;
      }
    }
  }

  // Create one runner for each context available for this model. Each
  // runner is exclusively tied to the context.
  TF_RETURN_IF_ERROR(SetRunnerCount(total_context_cnt));

  LOG_VERBOSE(1) << "plan bundle for " << Name() << std::endl << *this;

  return tensorflow::Status::OK();
}

tensorflow::Status
PlanBundle::CreateExecutionContext(
  const std::string& instance_name, const int gpu_device,
  const std::unordered_map<std::string, std::vector<char>>& models)
{
  cudaError_t cuerr;

  // Determine the model file to use for device compute capability
  cudaDeviceProp cuprops;
  cuerr = cudaGetDeviceProperties(&cuprops, gpu_device);
  if (cuerr != cudaSuccess) {
    return tensorflow::errors::Internal(
      "unable to get CUDA device properties for ", Name(), ": ",
      cudaGetErrorString(cuerr));
  }

  const std::string cc =
    std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
  const auto& cc_itr = Config().cc_model_filenames().find(cc);
  const std::string& cc_model_filename =
    (cc_itr == Config().cc_model_filenames().end())
      ? Config().default_model_filename()
      : cc_itr->second;

  const auto& mn_itr = models.find(cc_model_filename);
  if (mn_itr == models.end()) {
    return tensorflow::errors::Internal(
      "unable to find PLAN model '", cc_model_filename, "' for ", Name());
  }

  LOG_INFO << "Creating instance " << instance_name << " on GPU " << gpu_device
           << " (" << cc << ") using " << cc_model_filename;

  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  const int mbs = (Config().max_batch_size() <= 0) ? Context::NO_BATCHING
                                                   : Config().max_batch_size();

  contexts_.emplace_back(instance_name, gpu_device, mbs);
  Context& context = contexts_.back();

  // Set the device before generating engine and context.
  cuerr = cudaSetDevice(gpu_device);
  if (cuerr != cudaSuccess) {
    return tensorflow::errors::Internal(
      "unable to set device for ", Name(), ": ", cudaGetErrorString(cuerr));
  }

  // Create plugin factory to provide onnx plugins. This should be
  // generalized based on what the model requires [DLIS-54]
  nvonnxparser::IPluginFactory* onnx_plugin_factory =
    nvonnxparser::createPluginFactory(tensorrt_logger);

  // Runtime and engine...
  context.runtime_ = nvinfer1::createInferRuntime(tensorrt_logger);
  if (context.runtime_ == nullptr) {
    return tensorflow::errors::Internal("unable to create TensorRT runtime");
  }

  context.engine_ = context.runtime_->deserializeCudaEngine(
    &((mn_itr->second)[0]), mn_itr->second.size(), onnx_plugin_factory);
  if (context.engine_ == nullptr) {
    return tensorflow::errors::Internal("unable to create TensorRT engine");
  }

  if (context.max_batch_size_ > context.engine_->getMaxBatchSize()) {
    return tensorflow::errors::InvalidArgument(
      "unexpected configuration maximum batch size ", Config().max_batch_size(),
      " for '", Name(), "', model maximum is ",
      context.engine_->getMaxBatchSize());
  }

  // Initialize the inputs and outputs. Make sure the model matches
  // what is in the configuration. Allocate memory for the maximum
  // possible batch size: min(engine maximum, config maximum)
  const int num_expected_bindings = context.engine_->getNbBindings();
  context.byte_sizes_ = new uint64_t[num_expected_bindings];
  context.buffers_ = new void*[num_expected_bindings]();  // init to nullptr

  TF_RETURN_IF_ERROR(context.InitializeInputBindings(Config().input()));
  TF_RETURN_IF_ERROR(context.InitializeOutputBindings(Config().output()));

  // Make sure every index is initialized.
  for (int i = 0; i < num_expected_bindings; ++i) {
    if (context.buffers_[i] == nullptr) {
      return tensorflow::errors::InvalidArgument(
        "expected configuration for ",
        (context.engine_->bindingIsInput(i) ? "input" : "output"), " '",
        context.engine_->getBindingName(i), "' for ", Name());
    }
  }

  // Now the TRT execution context
  context.context_ = context.engine_->createExecutionContext();
  if (context.context_ == nullptr) {
    return tensorflow::errors::Internal("unable to create TensorRT context");
  }

  // Create CUDA objects needed for inference run
  cuerr = cudaStreamCreate(&context.stream_);
  if (cuerr != cudaSuccess) {
    return tensorflow::errors::Internal(
      "unable to create stream for ", Name(), ": ", cudaGetErrorString(cuerr));
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
PlanBundle::Context::InitializeInputBindings(
  const ::google::protobuf::RepeatedPtrField<ModelInput>& ios)
{
  for (const auto& io : ios) {
    TF_RETURN_IF_ERROR(ValidateModelInput(io));

    int index = engine_->getBindingIndex(io.name().c_str());
    if (index < 0) {
      return tensorflow::errors::NotFound(
        "input '", io.name(), "' not found for ", name_);
    }

    if (buffers_[index] != nullptr) {
      return tensorflow::errors::InvalidArgument(
        "input '", io.name(), "' has already appeared as an ",
        "input or output for ", name_);
    }

    if (!engine_->bindingIsInput(index)) {
      return tensorflow::errors::InvalidArgument(
        "input '", io.name(), "' is expected to be an output in model for ",
        name_);
    }

    DataType dt = ConvertDatatype(engine_->getBindingDataType(index));
    if (dt != io.data_type()) {
      return tensorflow::errors::InvalidArgument(
        "input '", io.name(), "' datatype is ", DataType_Name(io.data_type()),
        ", model specifies ", DataType_Name(dt), " for ", name_);
    }

    nvinfer1::Dims dims = engine_->getBindingDimensions(index);
    if (!CompareDims(dims, io.dims())) {
      return tensorflow::errors::InvalidArgument(
        "input '", io.name(), "' dims ", DimsDebugString(dims),
        " don't match configuration dims ", DimsDebugString(io.dims()), " for ",
        name_);
    }

    const uint64_t byte_size = GetSize(max_batch_size_, dt, io.dims());
    if (byte_size == 0) {
      return tensorflow::errors::Internal(
        "unable to calculate size for input '", io.name(), " for ", name_);
    }

    // Allocate CUDA memory
    void* buffer;
    cudaError_t err = cudaMalloc(&buffer, byte_size);
    if (err != cudaSuccess) {
      return tensorflow::errors::Internal(
        "unable to allocate memory for input '", io.name(), " for ", name_,
        ": ", cudaGetErrorString(err));
    }

    byte_sizes_[index] = byte_size;
    buffers_[index] = buffer;
    num_inputs_++;
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
PlanBundle::Context::InitializeOutputBindings(
  const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
{
  for (const auto& io : ios) {
    TF_RETURN_IF_ERROR(ValidateModelOutput(io));

    int index = engine_->getBindingIndex(io.name().c_str());
    if (index < 0) {
      return tensorflow::errors::NotFound(
        "output '", io.name(), "' not found for ", name_);
    }

    if (buffers_[index] != nullptr) {
      return tensorflow::errors::InvalidArgument(
        "output '", io.name(), "' has already appeared as an ",
        "input or output for ", name_);
    }

    if (engine_->bindingIsInput(index)) {
      return tensorflow::errors::InvalidArgument(
        "output '", io.name(), "' is expected to be an input in model for ",
        name_);
    }

    DataType dt = ConvertDatatype(engine_->getBindingDataType(index));
    if (dt != io.data_type()) {
      return tensorflow::errors::InvalidArgument(
        "output '", io.name(), "' datatype is ", DataType_Name(io.data_type()),
        ", model specifies ", DataType_Name(dt), " for ", name_);
    }

    nvinfer1::Dims dims = engine_->getBindingDimensions(index);
    if (!CompareDims(dims, io.dims())) {
      return tensorflow::errors::InvalidArgument(
        "output '", io.name(), "' dims ", DimsDebugString(dims),
        " don't match configuration dims ", DimsDebugString(io.dims()), " for ",
        name_);
    }

    const uint64_t byte_size = GetSize(max_batch_size_, dt, io.dims());
    if (byte_size == 0) {
      return tensorflow::errors::Internal(
        "unable to calculate size for output '", io.name(), " for ", name_);
    }

    // Allocate CUDA memory
    void* buffer;
    cudaError_t err = cudaMalloc(&buffer, byte_size);
    if (err != cudaSuccess) {
      return tensorflow::errors::Internal(
        "unable to allocate memory for input '", io.name(), " for ", name_,
        ": ", cudaGetErrorString(err));
    }

    byte_sizes_[index] = byte_size;
    buffers_[index] = buffer;
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
PlanBundle::GetOutputDataType(const std::string& name, DataType* dtype) const
{
  const auto itr = output_dtype_map_.find(name);
  if (itr == output_dtype_map_.end()) {
    return tensorflow::errors::Internal(
      "unable to find datatype for output '", name, "'");
  }

  *dtype = itr->second;
  return tensorflow::Status::OK();
}

tensorflow::Status
PlanBundle::Run(uint32_t runner_idx, std::vector<RunnerPayload>* payloads)
{
  // Each runner executes using the corresponding context...
  if (runner_idx >= contexts_.size()) {
    return tensorflow::errors::Internal(
      "unexpected runner index", runner_idx, ", max allowed ",
      contexts_.size());
  }

  std::vector<ModelInferStats::ScopedTimer> compute_timers;
  for (auto& payload : *payloads) {
    compute_timers.emplace_back();
    payload.stats_->StartComputeTimer(&compute_timers.back());
    payload.stats_->SetGPUDevice(contexts_[runner_idx].gpu_device_);
  }

  return contexts_[runner_idx].Run(payloads);
}

tensorflow::Status
PlanBundle::Context::Run(std::vector<RunnerPayload>* payloads)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";

  cudaSetDevice(gpu_device_);

  // For each request in 'payloads' make sure the inputs are correct
  // and collect up the total batch size for this inference execution.
  size_t total_batch_size = 0;
  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
      payload.request_provider_->RequestHeader();

    if ((size_t)request_header.input().size() != num_inputs_) {
      payload.status_ = tensorflow::errors::InvalidArgument(
        "expected ", num_inputs_, " inputs but got ",
        request_header.input().size());
      continue;
    }

    // For models that don't support batching (i.e. max_batch_size_ ==
    // 0) the request batch-size will still be 1.
    const size_t batch_size = request_header.batch_size();
    if ((batch_size != 1) && ((int)batch_size > max_batch_size_)) {
      payload.status_ = tensorflow::errors::InvalidArgument(
        "unexpected batch size ", batch_size, " for '", name_,
        "', max allowed is ", max_batch_size_);
      continue;
    }

    // Validate that all inputs are expected and of the correct size.
    for (const auto& input : request_header.input()) {
      const std::string& name = input.name();
      const int bindex = engine_->getBindingIndex(name.c_str());
      if ((bindex < 0) || !engine_->bindingIsInput(bindex)) {
        payload.status_ = tensorflow::errors::InvalidArgument(
          "unexpected inference input '", name, "'");
        break;
      }

      const size_t expected_byte_size =
        (byte_sizes_[bindex] / std::max(1, max_batch_size_));
      if (input.byte_size() != expected_byte_size) {
        //payload.status_ = tensorflow::errors::InvalidArgument(
        //  "unexpected size ", input.byte_size(), " for inference input '", name,
        //  "', expecting ", expected_byte_size);
        //break;
      }
    }

    if (!payload.status_.ok()) {
      continue;
    }

    total_batch_size += batch_size;
  }

  // If there are no valid payloads then no need to run the
  // inference. The payloads will have their error status set so can
  // just return.
  if (total_batch_size == 0) {
    return tensorflow::Status::OK();
  }

  // total_batch_size can be 1 for models that don't support batching
  // (i.e. max_batch_size_ == 0).
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size_)) {
    return tensorflow::errors::Internal(
      "dynamic batch size ", total_batch_size, " for '", name_,
      "', max allowed is ", max_batch_size_);
  }

  // For each input, concatenate input values from each payload into
  // the corresponding binding.
  for (int bindex = 0; bindex < engine_->getNbBindings(); ++bindex) {
    if (!engine_->bindingIsInput(bindex)) {
      continue;
    }

    const std::string& name = engine_->getBindingName(bindex);
    const size_t batch1_byte_size =
      byte_sizes_[bindex] / std::max(1, max_batch_size_);
    size_t binding_copy_offset = 0;

    // Visit the payloads in order and copy the input tensors to
    // GPU. Skip payloads that had errors since they are not included in
    // the dynamic batch.
    for (auto& payload : *payloads) {
      if (!payload.status_.ok()) {
        continue;
      }

      const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
      InferRequestProvider* request_provider = payload.request_provider_;
      const size_t expected_byte_size =
        request_header.batch_size() * batch1_byte_size;

      int input_idx = 0;
      for (const auto& input : request_header.input()) {
        if (input.name() == name) {
          size_t copied_byte_size = 0;
          while (payload.compute_status_.ok()) {
            const void* content;
            size_t content_byte_size;
            payload.compute_status_ = request_provider->GetNextInputContent(
              input_idx, &content, &content_byte_size, false);
            if (!payload.compute_status_.ok()) {
              break;
            }

            // No more input content available then done with copying...
            if (content == nullptr) {
              break;
            }

            if (
              (binding_copy_offset + copied_byte_size + content_byte_size) >
              byte_sizes_[bindex]) {
              payload.compute_status_ = tensorflow::errors::InvalidArgument(
                "unexpected size ",
                binding_copy_offset + copied_byte_size + content_byte_size,
                " for inference input '", name, "', expecting ",
                byte_sizes_[bindex]);
              break;
            }

            cudaError_t err = cudaMemcpyAsync(
              static_cast<char*>(buffers_[bindex]) + binding_copy_offset +
                copied_byte_size,
              content, content_byte_size, cudaMemcpyHostToDevice, stream_);
            if (err != cudaSuccess) {
              payload.compute_status_ = tensorflow::errors::Internal(
                "failed to copy input values to GPU for input '", name,
                "': ", cudaGetErrorString(err));
              break;
            }

            copied_byte_size += content_byte_size;
          }

          if (
            payload.compute_status_.ok() &&
            (copied_byte_size != expected_byte_size)) {
            payload.compute_status_ = tensorflow::errors::Internal(
              "expected ", expected_byte_size, " of data for inference input '",
              name, "', got ", copied_byte_size);
          }

          break;
        }

        input_idx++;
      }

      binding_copy_offset += expected_byte_size;
    }
  }

  // Async execute the inference.
  if (!context_->enqueue(total_batch_size, buffers_, stream_, nullptr)) {
    cudaStreamSynchronize(stream_);
    return tensorflow::errors::Internal(
      "unable to enqueue for inference ", name_);
  }

  // For each requested output verify that the output can accept the
  // actual model output and then copy that output from the GPU
  for (int bindex = 0; bindex < engine_->getNbBindings(); ++bindex) {
    if (engine_->bindingIsInput(bindex)) {
      continue;
    }

    const std::string& name = engine_->getBindingName(bindex);
    const size_t batch1_byte_size =
      (byte_sizes_[bindex] / std::max(1, max_batch_size_));
    size_t binding_copy_offset = 0;

    for (auto& payload : *payloads) {
      if (!payload.status_.ok()) {
        continue;
      }

      // If 'payload' requested this output then copy it from the
      // GPU. If it did not request this output then just skip it in
      // the output buffer.
      const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
      const size_t expected_byte_size =
        request_header.batch_size() * batch1_byte_size;

      int output_idx = 0;
      for (const auto& output : request_header.output()) {
        if (output.name() == name) {
          void* content;
          tensorflow::Status status =
            payload.response_provider_->GetOutputBuffer(
              output_idx, &content, expected_byte_size);
          if (!status.ok()) {
            payload.compute_status_ = status;
          } else if (content == nullptr) {
            payload.compute_status_ = tensorflow::errors::Internal(
              "no buffer to accept output values for output '", name, "'");
          } else {
            if (
              (binding_copy_offset + expected_byte_size) >
              byte_sizes_[bindex]) {
              payload.compute_status_ = tensorflow::errors::InvalidArgument(
                "unexpected size ", binding_copy_offset + expected_byte_size,
                " for inference output '", name, "', expecting maximum",
                byte_sizes_[bindex]);
            } else {
              cudaError_t err = cudaMemcpyAsync(
                content,
                static_cast<char*>(buffers_[bindex]) + binding_copy_offset,
                expected_byte_size, cudaMemcpyDeviceToHost, stream_);
              if (err != cudaSuccess) {
                payload.compute_status_ = tensorflow::errors::Internal(
                  "failed to copy output values from GPU for output '", name,
                  "': ", cudaGetErrorString(err));
              }
            }
          }

          break;
        }

        output_idx++;
      }

      binding_copy_offset += expected_byte_size;
    }
  }

  // Wait for the copy-out to complete
  cudaStreamSynchronize(stream_);
  return tensorflow::Status::OK();
}

std::ostream&
operator<<(std::ostream& out, const PlanBundle& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context.name_ << ", gpu="
        << ((context.gpu_device_ == PlanBundle::Context::NO_GPU_DEVICE)
              ? "<none>"
              : std::to_string(context.gpu_device_))
        << ", max_batch_size="
        << ((context.max_batch_size_ == PlanBundle::Context::NO_BATCHING)
              ? "<none>"
              : std::to_string(context.max_batch_size_))
        << std::endl
        << "  bindings:" << std::endl;

    for (int i = 0; i < context.engine_->getNbBindings(); ++i) {
      out << "    " << i << ": byte_size=" << context.byte_sizes_[i]
          << ", buffer=" << context.buffers_[i] << " ]" << std::endl;
    }
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
