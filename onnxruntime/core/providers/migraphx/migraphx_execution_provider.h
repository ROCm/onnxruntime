#pragma once

#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"
#include <map>
#include "migraphx_inc.h"

namespace onnxruntime {

// Information needed to construct amdmigraphx execution providers.
struct MiGraphXExecutionProviderInfo {
  std::string target_device;
  int device_id {0};
};

// Information to construct kernel function state.
struct MiGraphXFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocate_handle = nullptr;
  migraphx::program prog{};
  migraphx::target t{};
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  OrtMutex* mgx_mu_ptr = nullptr;
};

// Logical device representation.
class MiGraphXExecutionProvider : public IExecutionProvider {
 public:
  explicit MiGraphXExecutionProvider(const MiGraphXExecutionProviderInfo& info);
  ~MiGraphXExecutionProvider() = default;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

private:
  int device_id_;
  migraphx::target t_; 
  OrtMutex mgx_mu_;

  std::unordered_map<std::string, migraphx::program> map_progs_;
  std::unordered_map<std::string, std::vector<std::string>> map_input_names_;
  std::unordered_map<std::string, std::vector<std::string>> map_output_names_;

  AllocatorPtr allocator_;
};

}
