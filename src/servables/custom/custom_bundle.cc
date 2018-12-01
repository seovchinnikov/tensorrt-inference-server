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

#include "src/servables/custom/custom_bundle.h"

#include <cuda_runtime_api.h>
#include <stdint.h>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/server_status.h"
#include "src/core/utils.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/io/path.h"

namespace nvidia { namespace inferenceserver {

CustomBundle::Context::Context(
  const std::string& name, const int gpu_device, const int max_batch_size)
    : name_(name), gpu_device_(gpu_device), max_batch_size_(max_batch_size)
{
}

CustomBundle::Context::Context(Context&& o)
    : name_(std::move(o.name_)), gpu_device_(o.gpu_device_),
      max_batch_size_(o.max_batch_size_)
{
  o.gpu_device_ = NO_GPU_DEVICE;
  o.max_batch_size_ = NO_BATCHING;
}

CustomBundle::Context::~Context()
{
  LOG_VERBOSE(1) << "~CustomBundle::Context ";
}

tensorflow::Status
CustomBundle::Init(
  const tensorflow::StringPiece& path, const ModelConfig& config)
{
  TF_RETURN_IF_ERROR(ValidateModelConfig(config, kCustomPlatform));
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
CustomBundle::CreateExecutionContexts(
  const std::unordered_map<std::string, std::vector<char>>& models)
{
  uint32_t total_context_cnt = 0;

  // Create the context for each instance.
  for (const auto& group : Config().instance_group()) {
    for (int c = 0; c < group.count(); c++) {
      if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        const std::string instance_name =
          group.name() + "_" + std::to_string(c) + "_cpu";
        TF_RETURN_IF_ERROR(CreateExecutionContext(
          instance_name, Context::NO_GPU_DEVICE, models));
      } else {
        for (int gpu_device : group.gpus()) {
          const std::string instance_name = group.name() + "_" +
                                            std::to_string(c) + "_gpu" +
                                            std::to_string(gpu_device);
          TF_RETURN_IF_ERROR(
            CreateExecutionContext(instance_name, gpu_device, models));
        }
      }

      total_context_cnt++;
    }
  }

  // Create one runner for each context available for this
  // model. Each runner is exclusively tied to the context.
  TF_RETURN_IF_ERROR(SetRunnerCount(total_context_cnt));

  LOG_VERBOSE(1) << "custom bundle for " << Name() << std::endl << *this;
  return tensorflow::Status::OK();
}

tensorflow::Status
CustomBundle::CreateExecutionContext(
  const std::string& instance_name, const int gpu_device,
  const std::unordered_map<std::string, std::vector<char>>& models)
{
  cudaError_t cuerr;

  // For a GPU context, determine the model file to use for device
  // compute capability. CPU always uses the default model file.
  std::string cc;
  std::string cc_model_filename;
  if (gpu_device == Context::NO_GPU_DEVICE) {
    cc_model_filename = Config().default_model_filename();
  } else {
    cudaDeviceProp cuprops;
    cuerr = cudaGetDeviceProperties(&cuprops, gpu_device);
    if (cuerr != cudaSuccess) {
      return tensorflow::errors::Internal(
        "unable to get CUDA device properties for ", Name(), ": ",
        cudaGetErrorString(cuerr));
    }

    cc = std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
    const auto& cc_itr = Config().cc_model_filenames().find(cc);
    cc_model_filename = (cc_itr == Config().cc_model_filenames().end())
                          ? Config().default_model_filename()
                          : cc_itr->second;
  }


  const auto& mn_itr = models.find(cc_model_filename);
  if (mn_itr == models.end()) {
    return tensorflow::errors::Internal(
      "unable to find Custom model '", cc_model_filename, "' for ", Name());
  }

  if (gpu_device == Context::NO_GPU_DEVICE) {
    LOG_INFO << "Creating instance " << instance_name << " on CPU using "
             << cc_model_filename;
  } else {
    LOG_INFO << "Creating instance " << instance_name << " on GPU "
             << gpu_device << " (" << cc << ") using " << cc_model_filename;
  }

  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  const int mbs = (Config().max_batch_size() <= 0) ? Context::NO_BATCHING
                                                   : Config().max_batch_size();

  contexts_.emplace_back(instance_name, gpu_device, mbs);

  // HERE: is where you can initialize 'context' for a specific
  // 'gpu_device'. 'mn_itr->second' is the path to the model file to
  // use for that context (e.g. model_name/1/model.custom).
  // Context& context = contexts_.back();

  return tensorflow::Status::OK();
}

tensorflow::Status
CustomBundle::GetOutputDataType(const std::string& name, DataType* dtype) const
{
  const auto itr = output_dtype_map_.find(name);
  if (itr == output_dtype_map_.end()) {
    return tensorflow::errors::Internal(
      "unable to find datatype for output '", name, "'");
  }

  *dtype = itr->second;
  return tensorflow::Status::OK();
}

void
CustomBundle::Run(
  uint32_t runner_idx, std::vector<RunnerPayload>* payloads,
  std::function<void(tensorflow::Status)> OnCompleteQueuedPayloads)
{
  // Each runner executes using the corresponding context...
  if (runner_idx >= contexts_.size()) {
    OnCompleteQueuedPayloads(tensorflow::errors::Internal(
      "unexpected runner index", runner_idx, ", max allowed ",
      contexts_.size()));
    return;
  }

  std::vector<ModelInferStats::ScopedTimer> compute_timers;
  for (auto& payload : *payloads) {
    compute_timers.emplace_back();
    payload.stats_->StartComputeTimer(&compute_timers.back());
    payload.stats_->SetGPUDevice(contexts_[runner_idx].gpu_device_);
  }

  OnCompleteQueuedPayloads(contexts_[runner_idx].Run(payloads));
}

tensorflow::Status
CustomBundle::Context::Run(std::vector<RunnerPayload>* payloads)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";

  // HERE each RunnerPayload in 'payloads' has information about one
  // inference. There can be multiple payloads due to batching. The
  // expectation is that you execute the entire batch at once.

  // For each request in 'payloads' make sure the inputs are correct
  // and collect up the total batch size for this inference execution.
  size_t total_batch_size = 0;
  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
      payload.request_provider_->RequestHeader();

    // For models that don't support batching (i.e. max_batch_size_ ==
    // 0) the request batch-size will still be 1.
    const size_t batch_size = request_header.batch_size();
    if ((batch_size != 1) && ((int)batch_size > max_batch_size_)) {
      payload.status_ = tensorflow::errors::InvalidArgument(
        "unexpected batch size ", batch_size, " for '", name_,
        "', max allowed is ", max_batch_size_);
      continue;
    }

    // HERE make sure you have the right number and size of inputs.
    // Validate that all inputs are expected and of the correct size.
    // If something goes wrong you set payload.status_ like this:
    //    payload.status_ = tensorflow::errors::InvalidArgument(
    //      "unexpected inference input '", name, "'");

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

  // HERE each payload provides its input via a request_provider_ that
  // reads bytes directly from the HTTP or GRPC layer into... whatever
  // you want. Likely you want to do something like in plan_bundle.cc
  // where you asyncmemcpy them to the GPU. This part can be tricky so
  // we can discuss more once you get familiar.

  // HERE Run "inference"...

  // HERE check outputs and copy into the response_provider_ which
  // sends you result bytes directly back to the GRPC/HTTP layer. Like
  // inputs this can be tricky so we can discuss.

  return tensorflow::Status::OK();
}

std::ostream&
operator<<(std::ostream& out, const CustomBundle& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context.name_ << ", gpu="
        << ((context.gpu_device_ == CustomBundle::Context::NO_GPU_DEVICE)
              ? "<none>"
              : std::to_string(context.gpu_device_));
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
