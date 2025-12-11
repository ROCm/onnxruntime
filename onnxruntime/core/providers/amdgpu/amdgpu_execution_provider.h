// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/span>

#include "amdgpu_provider_utils.h"

class AmdGpuEpFactory;

/// <summary>
/// AMD GPU EP skeleton implementation.
/// This is a minimal implementation that provides the required interface.
/// Future implementations will add ROCm/HIP kernel support.
/// </summary>
class AmdGpuEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool enable_ep_context = false;
    // Other EP configs (typically extracted from OrtSessionOptions or OrtHardwareDevice(s))
  };

  AmdGpuEp(AmdGpuEpFactory& factory, const std::string& name, const Config& config, const OrtLogger& logger);

  ~AmdGpuEp();

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                     _In_ const OrtMemoryInfo* memory_info,
                                                     _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(_In_ OrtEp* this_ptr,
                                                               _In_ const OrtMemoryDevice* memory_device,
                                                               _Outptr_ OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
                                             _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                             _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                             _Out_writes_(count) OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept;

  OrtStatus* CreateEpContextNodes(gsl::span<const OrtNode*> fused_nodes,
                                  /*out*/ gsl::span<OrtNode*> ep_context_nodes);

  AmdGpuEpFactory& factory_;
  std::string name_;
  Config config_{};
  const OrtLogger& logger_;
};
