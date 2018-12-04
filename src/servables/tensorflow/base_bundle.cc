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

#include "src/servables/tensorflow/base_bundle.h"

#include <cuda_runtime_api.h>
#include <set>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/server_status.h"
#include "src/core/utils.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/io/path.h"

namespace nvidia { namespace inferenceserver {

namespace {

tensorflow::DataType
ConvertDatatype(DataType dtype)
{
  switch (dtype) {
    case DataType::TYPE_INVALID:
      return tensorflow::DT_INVALID;
    case DataType::TYPE_BOOL:
      return tensorflow::DT_BOOL;
    case DataType::TYPE_UINT8:
      return tensorflow::DT_UINT8;
    case DataType::TYPE_UINT16:
      return tensorflow::DT_UINT16;
    case DataType::TYPE_UINT32:
      return tensorflow::DT_UINT32;
    case DataType::TYPE_UINT64:
      return tensorflow::DT_UINT64;
    case DataType::TYPE_INT8:
      return tensorflow::DT_INT8;
    case DataType::TYPE_INT16:
      return tensorflow::DT_INT16;
    case DataType::TYPE_INT32:
      return tensorflow::DT_INT32;
    case DataType::TYPE_INT64:
      return tensorflow::DT_INT64;
    case DataType::TYPE_FP16:
      return tensorflow::DT_HALF;
    case DataType::TYPE_FP32:
      return tensorflow::DT_FLOAT;
    case DataType::TYPE_FP64:
      return tensorflow::DT_DOUBLE;
    default:
      break;
  }

  return tensorflow::DT_INVALID;
}

}  // namespace

BaseBundle::Context::Context(
  const std::string& name, const int gpu_device, const int max_batch_size)
    : name_(name), gpu_device_(gpu_device), max_batch_size_(max_batch_size),
      session_(nullptr)
{
}

BaseBundle::Context::Context(Context&& o)
    : name_(std::move(o.name_)), gpu_device_(o.gpu_device_),
      max_batch_size_(o.max_batch_size_),
      input_name_map_(std::move(o.input_name_map_)),
      output_name_map_(std::move(o.output_name_map_)),
      inputs_(std::move(o.inputs_)), outputs_(std::move(o.outputs_)),
      session_(o.session_)
{
  o.gpu_device_ = NO_GPU_DEVICE;
  o.max_batch_size_ = NO_BATCHING;
  o.session_ = nullptr;
}

BaseBundle::Context::~Context()
{
  LOG_VERBOSE(1) << "~BaseBundle::Context ";

  if (session_ != nullptr) {
    session_->Close().IgnoreError();
    session_ = nullptr;
  }
}

tensorflow::Status
BaseBundle::Init(const tensorflow::StringPiece& path, const ModelConfig& config)
{
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
BaseBundle::CreateExecutionContexts(
  const tensorflow::ConfigProto& session_config,
  const std::unordered_map<std::string, std::string>& paths)
{
  uint32_t total_context_cnt = 0;

  for (const auto& group : Config().instance_group()) {
    for (int c = 0; c < group.count(); c++) {
      if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        const std::string instance_name =
          group.name() + "_" + std::to_string(c) + "_cpu";
        TF_RETURN_IF_ERROR(CreateExecutionContext(
          instance_name, Context::NO_GPU_DEVICE, session_config, paths));
      } else {
        for (int gpu_device : group.gpus()) {
          const std::string instance_name = group.name() + "_" +
                                            std::to_string(c) + "_gpu" +
                                            std::to_string(gpu_device);
          TF_RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, gpu_device, session_config, paths));
        }
      }

      total_context_cnt++;
    }
  }

  // Create one runner for each context available for this
  // model. Each runner is exclusively tied to the context.
  TF_RETURN_IF_ERROR(SetRunnerCount(total_context_cnt));

  LOG_VERBOSE(1) << "bundle for " << Name() << std::endl << *this;

  return tensorflow::Status::OK();
}

tensorflow::Status
BaseBundle::CreateExecutionContext(
  const std::string& instance_name, const int gpu_device,
  const tensorflow::ConfigProto& session_config,
  const std::unordered_map<std::string, std::string>& paths)
{
  // For a GPU context, determine the model file to use for device
  // compute capability. CPU always uses the default model file.
  std::string cc_model_filename;
  if (gpu_device == Context::NO_GPU_DEVICE) {
    cc_model_filename = Config().default_model_filename();

    LOG_INFO << "Creating instance " << instance_name << " on CPU using "
             << cc_model_filename;
  } else {
    cudaDeviceProp cuprops;
    cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, gpu_device);
    if (cuerr != cudaSuccess) {
      return tensorflow::errors::Internal(
        "unable to get CUDA device properties for ", Name(), ": ",
        cudaGetErrorString(cuerr));
    }

    const std::string cc =
      std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
    const auto& cc_itr = Config().cc_model_filenames().find(cc);
    cc_model_filename = (cc_itr == Config().cc_model_filenames().end())
                          ? Config().default_model_filename()
                          : cc_itr->second;

    LOG_INFO << "Creating instance " << instance_name << " on GPU "
             << gpu_device << " (" << cc << ") using " << cc_model_filename;
  }

  const auto& gdp_itr = paths.find(cc_model_filename);
  if (gdp_itr == paths.end()) {
    return tensorflow::errors::Internal(
      "unable to find model '", cc_model_filename, "' for ", Name());
  }

  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  const int mbs = (Config().max_batch_size() <= 0) ? Context::NO_BATCHING
                                                   : Config().max_batch_size();

  contexts_.emplace_back(instance_name, gpu_device, mbs);
  Context& context = contexts_.back();

  // Session GPU option visible_device_list does not work (see
  // https://github.com/tensorflow/tensorflow/issues/8136 and many
  // related issues), so we can't use it here to set the GPU (see
  // CreateSession implementations for SetDefaultDevice). [DLIS-43]
  tensorflow::SessionOptions options;
  options.config = session_config;

  // Enable/disable XLA based on the model config optimization
  // setting.
  tensorflow::OptimizerOptions::GlobalJitLevel xla =
    tensorflow::OptimizerOptions::DEFAULT;
  if (Config().optimization().has_graph()) {
    if (Config().optimization().graph().level() == -1) {
      xla = tensorflow::OptimizerOptions::OFF;
    } else if (Config().optimization().graph().level() == 1) {
      xla = tensorflow::OptimizerOptions::ON_1;
    } else if (Config().optimization().graph().level() > 1) {
      xla = tensorflow::OptimizerOptions::ON_2;
    }
  }

  options.config.mutable_graph_options()
    ->mutable_optimizer_options()
    ->set_global_jit_level(xla);

  TF_RETURN_IF_ERROR(CreateSession(
    options, gpu_device, gdp_itr->second, &context.session_,
    &context.input_name_map_, &context.output_name_map_));

  // Initialize an appropriately sized Tensor for each input and
  // output.
  TF_RETURN_IF_ERROR(context.InitializeInputs(Config().input()));
  TF_RETURN_IF_ERROR(context.InitializeOutputs(Config().output()));

  return tensorflow::Status::OK();
}

tensorflow::Status
BaseBundle::Context::InitializeInputs(
  const ::google::protobuf::RepeatedPtrField<ModelInput>& ios)
{
  for (const auto& io : ios) {
    tensorflow::TensorShape shape;
    for (int d = 0; d < io.dims_size(); ++d) {
      shape.AddDim(io.dims(d));
    }

    tensorflow::DataType dtype = ConvertDatatype(io.data_type());
    inputs_.insert({io.name(), tensorflow::Tensor(dtype, shape)});
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
BaseBundle::Context::InitializeOutputs(
  const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
{
  for (const auto& io : ios) {
    tensorflow::TensorShape shape;
    for (int d = 0; d < io.dims_size(); ++d) {
      shape.AddDim(io.dims(d));
    }

    tensorflow::DataType dtype = ConvertDatatype(io.data_type());
    outputs_.insert({io.name(), tensorflow::Tensor(dtype, shape)});
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
BaseBundle::GetOutputDataType(const std::string& name, DataType* dtype) const
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
BaseBundle::Run(uint32_t runner_idx, std::vector<RunnerPayload>* payloads)
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
BaseBundle::Context::Run(std::vector<RunnerPayload>* payloads)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";

  // For each request in 'payloads' make sure the inputs are correct
  // and collect up the total batch size for this inference execution.
  size_t total_batch_size = 0;
  for (auto& payload : *payloads) {
    const InferRequestHeader& request_header =
      payload.request_provider_->RequestHeader();

    if ((size_t)request_header.input().size() != inputs_.size()) {
      payload.status_ = tensorflow::errors::InvalidArgument(
        "expected ", inputs_.size(), " inputs but got ",
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

      const auto& ii_iter = inputs_.find(name);
      if (ii_iter == inputs_.end()) {
        payload.status_ = tensorflow::errors::InvalidArgument(
          "unexpected inference input '", name, "' for '", name_, "'");
        break;
      }

      const tensorflow::Tensor& tensor = ii_iter->second;
      const size_t expected_byte_size =
        tensor.NumElements() * tensorflow::DataTypeSize(tensor.dtype());
      if (input.byte_size() != expected_byte_size) {
        //payload.status_ = tensorflow::errors::InvalidArgument(
        //  "unexpected size ", input.byte_size(), " for inference input '", name,
        //  "', expecting ", expected_byte_size);
        break;
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

  // Create a tensor for each input, sized correctly for the total
  // payload batch size. Concatenate input values from each payload
  // into the corresponding tensor.
  using TensorVec = std::vector<std::pair<std::string, tensorflow::Tensor>>;
  TensorVec input_tensors;
  for (const auto& ipair : inputs_) {
    const std::string& name = ipair.first;
    const tensorflow::Tensor& batch1_tensor = ipair.second;
    tensorflow::TensorShape shape = ipair.second.shape();
    const size_t batch1_byte_size =
      batch1_tensor.NumElements() *
      tensorflow::DataTypeSize(batch1_tensor.dtype());

    // If model supports batching then prepend the batch dimension
    // onto the input shape.
    if (max_batch_size_ != NO_BATCHING) {
      shape.InsertDim(0, total_batch_size);
    }

    const std::string* input_tensor_name = &name;
    const auto& tn_itr = input_name_map_.find(name);
    if (tn_itr != input_name_map_.end()) {
      input_tensor_name = &tn_itr->second;
    }

    input_tensors.emplace_back(std::make_pair(
      *input_tensor_name, tensorflow::Tensor(ipair.second.dtype(), shape)));
    tensorflow::Tensor& tensor = input_tensors.back().second;
    auto flat = tensor.bit_casted_shaped<char, 1>(
      {tensor.NumElements() * tensorflow::DataTypeSize(tensor.dtype())});
    size_t tensor_copy_offset = 0;

    // Visit the payloads in order and copy the input tensors to
    // GPU. Skip payloads that had errors since they are not included
    // in the dynamic batch.
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
              (tensor_copy_offset + copied_byte_size + content_byte_size) >
              ((size_t)flat.size())) {
              payload.compute_status_ = tensorflow::errors::InvalidArgument(
                "unexpected size ",
                tensor_copy_offset + copied_byte_size + content_byte_size,
                " for inference input '", name, "', expecting ", flat.size());
              break;
            }

            memcpy(
              static_cast<char*>(flat.data()) + tensor_copy_offset +
                copied_byte_size,
              content, content_byte_size);
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

      tensor_copy_offset += expected_byte_size;
    }
  }

  // Collect the names of outputs requested by any request
  // payload. Skip payloads that have an error.
  std::unordered_map<std::string, uint64_t> required_outputs;
  for (auto& payload : *payloads) {
    if (!payload.status_.ok()) {
      continue;
    }

    const InferRequestHeader& request_header =
      payload.request_provider_->RequestHeader();
    for (const auto& output : request_header.output()) {
      required_outputs.insert(
        std::make_pair(output.name(), output.byte_size()));
    }
  }

  // Create the vector of required output names.
  std::vector<std::string> output_names;
  for (const auto& opair : outputs_) {
    const std::string& name = opair.first;
    const auto& ritr = required_outputs.find(name);
    if (ritr == required_outputs.end()) {
      continue;
    }

    const auto& tn_itr = output_name_map_.find(name);
    if (tn_itr == output_name_map_.end()) {
      output_names.push_back(name);
    } else {
      output_names.push_back(tn_itr->second);
    }
  }

  // Run. Session will update the 'outputs'.
  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session_->Run(input_tensors, output_names, {}, &outputs));

  // Make sure each output is of the expected size and copy it into
  // the appropriate response providers.
  int output_idx = 0;
  for (const auto& opair : outputs_) {
    const std::string& name = opair.first;
    const auto& ritr = required_outputs.find(name);
    if (ritr == required_outputs.end()) {
      continue;
    }

    const tensorflow::Tensor& expected_template = opair.second;
    const size_t batch1_byte_size =
      expected_template.NumElements() *
      tensorflow::DataTypeSize(expected_template.dtype());

    // Use the output template and fix the shape based on the batch
    // size of the request.
    tensorflow::TensorShape shape = expected_template.shape();
    if (max_batch_size_ != NO_BATCHING) {
      shape.InsertDim(0, total_batch_size);
    }
    tensorflow::Tensor expected(expected_template.dtype(), shape);

    if (expected.dtype() != outputs[output_idx].dtype()) {
      return tensorflow::errors::InvalidArgument(
        "unexpected datatype ", outputs[output_idx].dtype(),
        " for inference output '", name, "', expecting ", expected.dtype());
    }

    if (expected.shape() != outputs[output_idx].shape()) {
      return tensorflow::errors::InvalidArgument(
        "unexpected shape ", outputs[output_idx].shape().DebugString(),
        " for inference output '", name, "', expecting ",
        expected.shape().DebugString());
    }

    const auto& flat = outputs[output_idx].bit_casted_shaped<char, 1>(
      {outputs[output_idx].NumElements() *
       tensorflow::DataTypeSize(outputs[output_idx].dtype())});
    size_t tensor_copy_offset = 0;

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

      int req_output_idx = 0;
      for (const auto& output : request_header.output()) {
        if (output.name() == name) {
          void* content;
          tensorflow::Status status =
            payload.response_provider_->GetOutputBuffer(
              req_output_idx, &content, expected_byte_size);
          if (!status.ok()) {
            payload.compute_status_ = status;
          } else if (content == nullptr) {
            payload.compute_status_ = tensorflow::errors::Internal(
              "no buffer to accept output values for output '", name, "'");
          } else {
            if (
              (tensor_copy_offset + expected_byte_size) >
              ((size_t)flat.size())) {
              payload.compute_status_ = tensorflow::errors::InvalidArgument(
                "unexpected size ", tensor_copy_offset + expected_byte_size,
                " for inference output '", name, "', expecting ", flat.size());
            } else {
              memcpy(
                content, static_cast<char*>(flat.data()) + tensor_copy_offset,
                expected_byte_size);
            }
          }

          break;
        }

        req_output_idx++;
      }

      tensor_copy_offset += expected_byte_size;
    }

    output_idx++;
  }

  return tensorflow::Status::OK();
}

std::ostream&
operator<<(std::ostream& out, const BaseBundle& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context.name_ << ", gpu="
        << ((context.gpu_device_ == BaseBundle::Context::NO_GPU_DEVICE)
              ? "<none>"
              : std::to_string(context.gpu_device_))
        << ", max_batch_size="
        << ((context.max_batch_size_ == BaseBundle::Context::NO_BATCHING)
              ? "<none>"
              : std::to_string(context.max_batch_size_))
        << std::endl
        << "  inputs:" << std::endl;
    for (const auto& inp : context.inputs_) {
      out << "    name=" << inp.first << " " << inp.second.DebugString()
          << std::endl;
    }
    out << "  inputs:" << std::endl;
    for (const auto& outp : context.outputs_) {
      out << "    name=" << outp.first << " " << outp.second.DebugString()
          << std::endl;
    }
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
