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
#pragma once

#include <condition_variable>
#include <vector>
#include <mutex>
#include "libevent/include/event2/buffer.h"
#include "src/core/api.pb.h"
#include "src/core/grpc_service.pb.h"
#include "src/core/label_provider.h"
#include "src/core/metrics.h"
#include "src/core/model_config.pb.h"
#include "src/core/server_status.h"
#include "tensorflow/core/lib/core/errors.h"

struct evbuffer;

namespace nvidia { namespace inferenceserver {

class InferenceServable;

// Provide inference request inputs and meta-data
class InferRequestProvider {
 public:
  explicit InferRequestProvider(
    const std::string& model_name, const int version)
      : model_name_(model_name), version_(version)
  {
  }

  // Return the requested model name.
  const std::string& ModelName() const { return model_name_; }

  // Return the requested model version, or -1 if no specific version
  // was requested.
  int ModelVersion() const { return version_; }

  // Get the request header for this inference request.
  virtual const InferRequestHeader& RequestHeader() const = 0;

  // Get the next contiguous chunk of bytes for the 'idx'
  // input. Return a pointer to the chunk in 'content' and the length
  // of the chunk in 'content_byte_size'. If there are no more bytes
  // for the input return 'content' = nullptr. If 'force_contiguous'
  // is true then the entire (remaining) input will be returned as a
  // single chunk. In some cases this will require copying the data.
  virtual tensorflow::Status GetNextInputContent(
    int idx, const void** content, size_t* content_byte_size,
    bool force_contiguous) = 0;

 private:
  const std::string& model_name_;
  const int version_;
};

// Inference input provider for a gRPC inference request
class GRPCInferRequestProvider : public InferRequestProvider {
 public:
  // Initialize based on gRPC request
  static tensorflow::Status Create(
    const InferRequest& request,
    std::unique_ptr<GRPCInferRequestProvider>* infer_provider);

  const InferRequestHeader& RequestHeader() const override
  {
    return request_.meta_data();
  }

  tensorflow::Status GetNextInputContent(
    int idx, const void** content, size_t* content_byte_size,
    bool force_contiguous) override;

 private:
  GRPCInferRequestProvider(const InferRequest& request, const int version);

  const InferRequest& request_;
  std::vector<bool> content_delivered_;
  std::vector<std::string> contents;
};

// Inference input provider for an HTTP inference request
class HTTPInferRequestProvider : public InferRequestProvider {
 public:
  // Initialize based on HTTP request
  static tensorflow::Status Create(
    evbuffer* input_buffer, const std::string& model_name,
    const std::string& model_version_str, const std::string& request_header_str,
    std::unique_ptr<HTTPInferRequestProvider>* infer_provider);

  const InferRequestHeader& RequestHeader() const override
  {
    return request_header_;
  }

  tensorflow::Status GetNextInputContent(
    int idx, const void** content, size_t* content_byte_size,
    bool force_contiguous) override;

 private:
  HTTPInferRequestProvider(const std::string& model_name, const int version)
      : InferRequestProvider(model_name, version)
  {
  }

  InferRequestHeader request_header_;
  using Block = std::pair<const char*, size_t>;
  std::vector<std::vector<Block>> contents_;
  std::vector<size_t> contents_idx_;
  std::vector<std::vector<char>> contiguous_buffers_;
};


// Provide inference request outputs
class InferResponseProvider {
 public:
  explicit InferResponseProvider(const InferRequestHeader& request_header)
      : request_header_(request_header)
  {
  }

  // Get the response header for this inference request.
  virtual const InferResponseHeader& ResponseHeader() const = 0;

  // Get the response header for this inference request.
  virtual InferResponseHeader* MutableResponseHeader() = 0;

  // Finialize response based on a servable.
  virtual tensorflow::Status FinalizeResponse(const InferenceServable& is)
  {
    return FinalizeResponseHeader(is);
  }

  // Create a buffer for the next output of the specified
  // 'byte_size'.
  void CreateOutputBuffer(size_t byte_size);

  // Set the buffer for the next output to by 'buffer' which is size
  // of 'byte_size'.
  void AddOutputBuffer(void* buffer, size_t byte_size);

  // Get a pointer to the buffer into which output 'idx' should be
  // written. The size of the buffer must be exactly
  // 'buffer_byte_size'.
  tensorflow::Status GetOutputBuffer(
    int idx, void** buffer, size_t buffer_byte_size);

  // Finialize response header values based on a servable.
  tensorflow::Status FinalizeResponseHeader(const InferenceServable& is);

 private:
  const InferRequestHeader& request_header_;

  using Buffer = std::pair<void*, size_t>;
  std::vector<Buffer> buffers_;

  std::vector<std::unique_ptr<char[]>> created_buffers_;
};

// Inference response provider for a gRPC inference request
class GRPCInferResponseProvider : public InferResponseProvider {
 public:
  // Initialize based on gRPC request
  static tensorflow::Status Create(
    const InferRequestHeader& request_header, InferResponse* response,
    std::unique_ptr<GRPCInferResponseProvider>* infer_provider);

  const InferResponseHeader& ResponseHeader() const override
  {
    return response_->meta_data();
  }

  InferResponseHeader* MutableResponseHeader() override
  {
    return response_->mutable_meta_data();
  }

 private:
  GRPCInferResponseProvider(
    const InferRequestHeader& request_header, InferResponse* response)
      : InferResponseProvider(request_header), response_(response)
  {
  }

  InferResponse* response_;
};

// Inference response provider for an HTTP inference request
class HTTPInferResponseProvider : public InferResponseProvider {
 public:
  static tensorflow::Status Create(
    evbuffer* output_buffer, const InferRequestHeader& request_header,
    std::unique_ptr<HTTPInferResponseProvider>* infer_provider);

  const InferResponseHeader& ResponseHeader() const override
  {
    return response_header_;
  }

  InferResponseHeader* MutableResponseHeader() override
  {
    return &response_header_;
  }

  tensorflow::Status FinalizeResponse(const InferenceServable& is);

 private:
  HTTPInferResponseProvider(
    evbuffer* output_buffer, const InferRequestHeader& request_header);

  InferResponseHeader response_header_;
  evbuffer* output_buffer_;
  struct evbuffer_iovec output_iovec_;
  size_t total_raw_byte_size_;
};


// Interface for servables that handle generic inference requests.
class InferenceServable {
 public:
  InferenceServable();
  virtual ~InferenceServable();

  // Get the name of model being served.
  const std::string& Name() const { return config_.name(); }

  // Get the version of model being served.
  uint32_t Version() const { return version_; }

  // Get the tags of model being served.
  const std::map<std::string, std::string>& Tags() const { return tags_; }

  // Get the configuration of model being served.
  const ModelConfig& Config() const { return config_; }

  // Get the datatype for a named output.
  virtual tensorflow::Status GetOutputDataType(
    const std::string& name, DataType* dtype) const = 0;

  // Get a label provider for the servable.
  virtual const LabelProvider& GetLabelProvider() const = 0;

  // Run inference using the provided request to produce outputs in
  // the provide response.
  tensorflow::Status Run(
    ModelInferStats* stats, InferRequestProvider* request_provider,
    InferResponseProvider* response_provider);

  // Get a metric for the servable specialized for the given GPU index
  // (if -1 then return non-specialized version of the metric).
  prometheus::Counter& MetricInferenceSuccess(int gpu_device) const;
  prometheus::Counter& MetricInferenceFailure(int gpu_device) const;
  prometheus::Counter& MetricInferenceCount(int gpu_device) const;
  prometheus::Counter& MetricInferenceExecutionCount(int gpu_device) const;
  prometheus::Counter& MetricInferenceRequestDuration(int gpu_device) const;
  prometheus::Counter& MetricInferenceComputeDuration(int gpu_device) const;
  prometheus::Counter& MetricInferenceQueueDuration(int gpu_device) const;
  prometheus::Histogram& MetricInferenceLoadRatio(int gpu_device) const;

 protected:
  // Set the configuration of the model being served.
  tensorflow::Status SetModelConfig(
    const tensorflow::StringPiece& path, const ModelConfig& config);

  // Set the number of runners to use for executing requests to this
  // servable. Currently this method may be called only once but in
  // the future could allow it to be called multiple times to
  // dynamically adjust the number of runners.
  tensorflow::Status SetRunnerCount(uint32_t cnt);

  // Called by runer thread when a request has been completed with the
  // result status for the request. If successful the ResponseProvider
  // will have been updated with the response.
  using CompleteFunc = std::function<void(tensorflow::Status)>;

  struct RunnerPayload {
    RunnerPayload() = default;
    RunnerPayload(const RunnerPayload& payload) = default;
    RunnerPayload(
      struct timespec queued_timestamp, ModelInferStats* stats,
      InferRequestProvider* request_provider,
      InferResponseProvider* response_provider, CompleteFunc complete_function)
        : queued_timestamp_(queued_timestamp), stats_(stats),
          request_provider_(request_provider),
          response_provider_(response_provider),
          complete_function_(complete_function),
          status_(tensorflow::Status::OK()),
          compute_status_(tensorflow::Status::OK())
    {
    }

    struct timespec queued_timestamp_;
    ModelInferStats* stats_;
    InferRequestProvider* request_provider_;
    InferResponseProvider* response_provider_;
    CompleteFunc complete_function_;
    tensorflow::Status status_;
    tensorflow::Status compute_status_;
  };

  // Run inference as the runner specified by 'runner_idx' using the
  // provided request payloads to produce outputs in the provided
  // response. A non-OK return status indicates an internal error that
  // prevents any of the of requests from completing. If an error is
  // isolate to a single request payload it will be reported in that
  // payload.
  virtual tensorflow::Status Run(
    uint32_t runner_idx, std::vector<RunnerPayload>* payloads)
  {
    return tensorflow::errors::Unavailable("unable to serve model");
  }

 private:
  // Configuration of the model that this servable represents.
  ModelConfig config_;

  // Version of the model that this servable represents.
  uint32_t version_;

  // Tags of the model that this servable represents.
  std::map<std::string, std::string> tags_;

  // The number of runner threads for this servable.
  uint32_t runner_cnt_;

  // The number of runner threads currently idle.
  uint32_t idle_runner_cnt_;

  // Mutex and condvar protecting the scheduling queue.
  std::mutex mu_;
  std::condition_variable cv_;

  // Queue holding inference requests for the model represented by
  // this servable.
  std::deque<RunnerPayload> queue_;

  std::vector<std::unique_ptr<std::thread>> runner_threads_;
  std::atomic<bool> runner_threads_exit_;

  void RunnerThread(const uint32_t runner_id);
  uint64_t GetDynamicBatch(const ModelDynamicBatching& batching_config);

  size_t max_preferred_batch_size_;
  std::set<int32_t> preferred_batch_sizes_;
  uint64_t pending_batch_delay_ns_;
  size_t pending_batch_size_;
  size_t pending_batch_queue_cnt_;

  void GetMetricLabels(
    std::map<std::string, std::string>* labels, const int gpu_device) const;
  prometheus::Counter& GetCounterMetric(
    std::map<int, prometheus::Counter*>& metrics,
    prometheus::Family<prometheus::Counter>& family,
    const int gpu_device) const;

  mutable std::map<int, prometheus::Counter*> metric_inf_success_;
  mutable std::map<int, prometheus::Counter*> metric_inf_failure_;
  mutable std::map<int, prometheus::Counter*> metric_inf_count_;
  mutable std::map<int, prometheus::Counter*> metric_inf_exec_count_;
  mutable std::map<int, prometheus::Counter*> metric_inf_request_duration_us_;
  mutable std::map<int, prometheus::Counter*> metric_inf_compute_duration_us_;
  mutable std::map<int, prometheus::Counter*> metric_inf_queue_duration_us_;
  mutable std::map<int, prometheus::Histogram*> metric_inf_load_ratio_;
};

}}  // namespace nvidia::inferenceserver
