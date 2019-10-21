#pragma once

#include "core/framework/execution_provider.h"
#include <map>

namespace amdmigraphx {
  struct program;
}

namespace onnxruntime {

// Information needed to construct amdmigraphx execution providers.
struct MiGraphExecutionProviderInfo {
 const std::string target_device;
 const int device_id {0};
};

// Information to construct kernel function state.
struct MiGraphXFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocate_handle = nullptr;
  migprahx::program prog{};
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  OrtMutex* mgx_mu_ptr = nullptr;
};

// Logical device representation.
class MiGraphXExecutionProvider : public IExecutionProvider {
 public:
  explicit MiGraphXExecutionProvider(MiGraphXExecutionProvider& info);
  ~MiGraphXExecutionProvider() = default;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

private:
  int device_id_;
  migraphx::target t; 
  OrtMutex mgx_mu_;

  std::unordered_map<std::string, migraphx::program> map_progs_;
  std::unordered_map<std::string, std::vector<std::string>> map_input_names_;
  std::unordered_map<std::string, std::vector<std::string>> map_output_names_;

};

}
