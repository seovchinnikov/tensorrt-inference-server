// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <unistd.h>
#include <iostream>
#include <string>
#include "src/clients/c++/examples/shm_utils.h"
#include "src/clients/c++/library/request_grpc.h"
#include "src/clients/c++/library/request_http.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    nic::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
            << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "For -i, available protocols are 'grpc' and 'http'."
            << "Default is 'http." << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8000");
  std::string protocol = "http";
  std::map<std::string, std::string> http_headers;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vi:u:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'i':
        protocol = optarg;
        break;
      case 'u':
        url = optarg;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  nic::Error err;

  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "simple";

  // Create a health context and get the ready and live state of the
  // server.
  std::unique_ptr<nic::ServerHealthContext> health_ctx;
  if (protocol == "http") {
    err = nic::ServerHealthHttpContext::Create(
        &health_ctx, url, http_headers, verbose);
  } else if (protocol == "grpc") {
    err = nic::ServerHealthGrpcContext::Create(&health_ctx, url, verbose);
  } else {
    Usage(argv, "unknown protocol '" + protocol + "'");
  }

  if (!err.IsOk()) {
    std::cerr << "error: unable to create health context: " << err << std::endl;
    exit(1);
  }

  bool live, ready;
  err = health_ctx->GetLive(&live);
  if (!err.IsOk()) {
    std::cerr << "error: unable to get server liveness: " << err << std::endl;
    exit(1);
  }

  err = health_ctx->GetReady(&ready);
  if (!err.IsOk()) {
    std::cerr << "error: unable to get server readiness: " << err << std::endl;
    exit(1);
  }

  std::cout << "Health for model " << model_name << ":" << std::endl;
  std::cout << "Live: " << live << std::endl;
  std::cout << "Ready: " << ready << std::endl;

  // Create a status context and get the status of the model.
  std::unique_ptr<nic::ServerStatusContext> status_ctx;
  if (protocol == "http") {
    err = nic::ServerStatusHttpContext::Create(
        &status_ctx, url, http_headers, model_name, verbose);
  } else if (protocol == "grpc") {
    err = nic::ServerStatusGrpcContext::Create(
        &status_ctx, url, model_name, verbose);
  } else {
    Usage(argv, "unknown protocol '" + protocol + "'");
  }

  if (!err.IsOk()) {
    std::cerr << "error: unable to create status context: " << err << std::endl;
    exit(1);
  }

  ni::ServerStatus server_status;
  err = status_ctx->GetServerStatus(&server_status);
  if (!err.IsOk()) {
    std::cerr << "error: unable to get status: " << err << std::endl;
    exit(1);
  }

  std::cout << "Status for model " << model_name << ":" << std::endl;
  std::cout << server_status.DebugString() << std::endl;

  // Create the inference context for the model.
  std::unique_ptr<nic::InferContext> infer_ctx;
  if (protocol == "http") {
    err = nic::InferHttpContext::Create(
        &infer_ctx, url, http_headers, model_name, -1 /* model_version */,
        verbose);
  } else if (protocol == "grpc") {
    err = nic::InferGrpcContext::Create(
        &infer_ctx, url, model_name, -1 /* model_version */, verbose);
  } else {
    Usage(argv, "unknown protocol '" + protocol + "'");
  }

  if (!err.IsOk()) {
    std::cerr << "error: unable to create inference context: " << err
              << std::endl;
    exit(1);
  }

  // Create the shared memory control context
  std::unique_ptr<nic::SharedMemoryControlContext> shared_memory_ctx;
  if (protocol == "http") {
    err = nic::SharedMemoryControlHttpContext::Create(
        &shared_memory_ctx, url, http_headers, verbose);
  } else {
    err = nic::SharedMemoryControlGrpcContext::Create(
        &shared_memory_ctx, url, verbose);
  }
  if (!err.IsOk()) {
    std::cerr << "error: unable to create shared memory control context: "
              << err << std::endl;
    exit(1);
  }

  std::shared_ptr<nic::InferContext::Input> input0, input1;
  std::shared_ptr<nic::InferContext::Output> output0, output1;
  FAIL_IF_ERR(infer_ctx->GetInput("INPUT0", &input0), "unable to get INPUT0");
  FAIL_IF_ERR(infer_ctx->GetInput("INPUT1", &input1), "unable to get INPUT1");
  FAIL_IF_ERR(
      infer_ctx->GetOutput("OUTPUT0", &output0), "unable to get OUTPUT0");
  FAIL_IF_ERR(
      infer_ctx->GetOutput("OUTPUT1", &output1), "unable to get OUTPUT1");

  FAIL_IF_ERR(input0->Reset(), "unable to reset INPUT0");
  FAIL_IF_ERR(input1->Reset(), "unable to reset INPUT1");

  // Get the size of the inputs and outputs from the Shape and DataType
  int input_byte_size =
      infer_ctx->ByteSize(input0->Dims(), ni::DataType::TYPE_INT32);
  int output_byte_size =
      infer_ctx->ByteSize(output0->Dims(), ni::DataType::TYPE_INT32);

  // Create Output0 and Output1 in Shared Memory
  std::string shm_key = "/output_simple";
  int shm_fd_op;
  int* output0_shm;
  FAIL_IF_ERR(
      nic::CreateSharedMemoryRegion(shm_key, output_byte_size * 2, &shm_fd_op),
      "");
  FAIL_IF_ERR(
      nic::MapSharedMemory(
          shm_fd_op, 0, output_byte_size * 2, (void**)&output0_shm),
      "");
  int* output1_shm = (int*)(output0_shm + 16);

  // Register Output shared memory with TRTIS
  err = shared_memory_ctx->RegisterSharedMemory(
      "output_data", "/output_simple", 0, output_byte_size * 2);
  if (!err.IsOk()) {
    std::cerr << "error: unable to register shared memory output region: "
              << err << std::endl;
    exit(1);
  }

  // Set the context options to do batch-size 1 requests. Also request that
  // all output tensors be returned using shared memory.
  std::unique_ptr<nic::InferContext::Options> options;
  FAIL_IF_ERR(
      nic::InferContext::Options::Create(&options),
      "unable to create inference options");

  options->SetBatchSize(1);
  options->AddSharedMemoryResult(output0, "output_data", 0, output_byte_size);
  options->AddSharedMemoryResult(
      output1, "output_data", output_byte_size, output_byte_size);

  FAIL_IF_ERR(
      infer_ctx->SetRunOptions(*options), "unable to set inference options");

  // Create Input0 and Input1 in Shared Memory. Initialize Input0 to unique
  // integers and Input1 to all ones.
  shm_key = "/input_simple";
  int shm_fd_ip, *input0_shm;
  FAIL_IF_ERR(
      nic::CreateSharedMemoryRegion(shm_key, input_byte_size * 2, &shm_fd_ip),
      "");
  FAIL_IF_ERR(
      nic::MapSharedMemory(
          shm_fd_ip, 0, input_byte_size * 2, (void**)&input0_shm),
      "");
  int* input1_shm = (int*)(input0_shm + 16);
  for (size_t i = 0; i < 16; ++i) {
    *(input0_shm + i) = i;
    *(input1_shm + i) = 1;
  }
  // Register Input shared memory with TRTIS
  FAIL_IF_ERR(
      shared_memory_ctx->RegisterSharedMemory(
          "input_data", "/input_simple", 0, input_byte_size * 2),
      "unable to register shared memory output region");

  // Set the shared memory region for Inputs
  FAIL_IF_ERR(
      input0->SetSharedMemory("input_data", 0, input_byte_size),
      "failed to set shared memory input");
  FAIL_IF_ERR(
      input1->SetSharedMemory("input_data", input_byte_size, input_byte_size),
      "failed to set shared memory input");

  // Send inference request to the inference server.
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  FAIL_IF_ERR(infer_ctx->Run(&results), "unable to run model");

  // We expect there to be 2 results. Walk over all 16 result elements
  // and print the sum and difference calculated by the model.
  if (results.size() != 2) {
    std::cerr << "error: expected 2 results, got " << results.size()
              << std::endl;
  }

  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_shm[i] << " + " << input1_shm[i] << " = "
              << output0_shm[i] << std::endl;
    std::cout << input0_shm[i] << " - " << input1_shm[i] << " = "
              << output1_shm[i] << std::endl;

    if ((input0_shm[i] + input1_shm[i]) != output0_shm[i]) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if ((input0_shm[i] - input1_shm[i]) != output1_shm[i]) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }

  // Get shared memory regions all active/registered within TRTIS
  ni::SharedMemoryStatus status;
  FAIL_IF_ERR(
      shared_memory_ctx->GetSharedMemoryStatus(&status),
      "failed to get shared memory status");
  std::cout << "Shared Memory Status:\n" << status.DebugString() << "\n";

  // Unregister shared memory (One by one or all at a time) from TRTIS
  // err = shared_memory_ctx->UnregisterAllSharedMemory();
  FAIL_IF_ERR(
      shared_memory_ctx->UnregisterSharedMemory("input_data"),
      "unable to unregister shared memory input region");
  FAIL_IF_ERR(
      shared_memory_ctx->UnregisterSharedMemory("output_data"),
      "unable to unregister shared memory output region");

  // Cleanup shared memory
  FAIL_IF_ERR(nic::UnmapSharedMemory(input0_shm, input_byte_size * 2), "");
  FAIL_IF_ERR(nic::UnlinkSharedMemoryRegion("/input_simple"), "");
  FAIL_IF_ERR(nic::UnmapSharedMemory(output0_shm, output_byte_size * 2), "");
  FAIL_IF_ERR(nic::UnlinkSharedMemoryRegion("/output_simple"), "");

  return 0;
}
