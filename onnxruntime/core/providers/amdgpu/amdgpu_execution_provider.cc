// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "amdgpu_execution_provider.h"
#include "amdgpu_factory.h"
#include "amdgpu_stream_support.h"

struct AmdGpuNodeComputeInfo : OrtNodeComputeInfo {
  explicit AmdGpuNodeComputeInfo(AmdGpuEp& ep);

  AmdGpuEp& ep;
};

AmdGpuEp::AmdGpuEp(AmdGpuEpFactory& factory, const std::string& name, const Config& config, const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values are default initialized
      ApiPtrs(factory),
      factory_(factory),
      name_(name),
      config_(config),
      logger_(logger) {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  CreateAllocator = CreateAllocatorImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(&logger_,
                                             OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                             ("AmdGpuEp has been created with name " + name_).c_str(),
                                             ORT_FILE, __LINE__, __FUNCTION__));
}

AmdGpuEp::~AmdGpuEp() = default;

/*static*/
const char* ORT_API_CALL AmdGpuEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const AmdGpuEp*>(this_ptr);
  return ep->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                                    OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    AmdGpuEp* ep = static_cast<AmdGpuEp*>(this_ptr);

    // Skeleton implementation: Don't claim to support any nodes yet
    // In a real implementation, this would check which operations can be accelerated on AMD GPU
    // and mark those nodes as supported using ep_api_.GraphSupportInfo_SetNodeSupported()

    IGNORE_ORTSTATUS(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                   OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                   "AmdGpuEp::GetCapability - skeleton implementation, no nodes supported yet",
                                                   ORT_FILE, __LINE__, __FUNCTION__));

    // Example of how to mark a node as supported (commented out for skeleton):
    // Ort::ConstGraph graph{ort_graph};
    // for (size_t i = 0; i < graph.NumberOfNodes(); ++i) {
    //   Ort::ConstNode node = graph.GetNode(i);
    //   if (node.OpType() == "SomeOp") {
    //     RETURN_IF_ERROR(ep->ep_api_.GraphSupportInfo_SetNodeSupported(graph_support_info, i, true));
    //   }
    // }

  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    AmdGpuEp* ep = static_cast<AmdGpuEp*>(this_ptr);
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, ex.what());
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuEp::CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** ort_graphs,
                                              _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                              _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                              _Out_writes_(count) OrtNode** ep_context_nodes) noexcept {
  try {
    AmdGpuEp* ep = static_cast<AmdGpuEp*>(this_ptr);

    IGNORE_ORTSTATUS(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                   OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                   "AmdGpuEp::Compile - skeleton implementation",
                                                   ORT_FILE, __LINE__, __FUNCTION__));

    // Skeleton implementation: Create minimal compute info structures
    // In a real implementation, this would compile the graphs for AMD GPU execution
    for (size_t i = 0; i < count; ++i) {
      node_compute_infos[i] = new AmdGpuNodeComputeInfo(*ep);
      ep_context_nodes[i] = nullptr;  // No EP context nodes in skeleton
    }

    // If EP context is enabled, create EP context nodes
    if (ep->config_.enable_ep_context) {
      RETURN_IF_ERROR(ep->CreateEpContextNodes(gsl::make_span(fused_nodes, count),
                                               gsl::make_span(ep_context_nodes, count)));
    }

  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    AmdGpuEp* ep = static_cast<AmdGpuEp*>(this_ptr);
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, ex.what());
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL AmdGpuEp::ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                        OrtNodeComputeInfo** node_compute_infos,
                                                        size_t num_node_compute_infos) noexcept {
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    delete static_cast<AmdGpuNodeComputeInfo*>(node_compute_infos[i]);
  }
}

OrtStatus* AmdGpuEp::CreateEpContextNodes(gsl::span<const OrtNode*> fused_nodes,
                                          /*out*/ gsl::span<OrtNode*> ep_context_nodes) {
  // Skeleton implementation: EP context not yet supported
  IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(&logger_,
                                             OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                             "AmdGpuEp::CreateEpContextNodes - not yet implemented in skeleton",
                                             ORT_FILE, __LINE__, __FUNCTION__));
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuEp::CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                      _In_ const OrtMemoryInfo* memory_info,
                                                      _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept {
  AmdGpuEp* ep = static_cast<AmdGpuEp*>(this_ptr);

  try {
    // Delegate to factory's CreateAllocator
    return ep->factory_.CreateAllocator(&ep->factory_, memory_info, nullptr, allocator);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuEp::CreateSyncStreamForDeviceImpl(_In_ OrtEp* this_ptr,
                                                                _In_ const OrtMemoryDevice* memory_device,
                                                                _Outptr_ OrtSyncStreamImpl** stream) noexcept {
  AmdGpuEp* ep = static_cast<AmdGpuEp*>(this_ptr);

  try {
    auto stream_impl = std::make_unique<AmdGpuStreamImpl>(ep->factory_, this_ptr, nullptr);
    *stream = stream_impl.release();
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, ex.what());
  }

  return nullptr;
}

//
// AmdGpuNodeComputeInfo implementation
//

AmdGpuNodeComputeInfo::AmdGpuNodeComputeInfo(AmdGpuEp& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = nullptr;  // Skeleton: No state creation needed
  Compute = nullptr;      // Skeleton: No actual compute function yet
  ReleaseState = nullptr; // Skeleton: No state to release
}
