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
 const int device_id;
};

// Logical device representation.
class MiGraphExecutionProvider : public IExecutionProvider {
 public:
  explicit MiGraphExecutionProvider(MiGraphExecutionProviderInfo& info);
  ~MiGraphExecutionProvider() = default;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

private:
  int device_id_;
  OrtMutex tensorrt_mu_;

//  std::shared_ptr<amdmigraphx::program> prog_;
};

}
