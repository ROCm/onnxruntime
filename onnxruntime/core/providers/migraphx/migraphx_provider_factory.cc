#include "core/providers/migraphx/migraphx_provider_factory.h"
#include <atomic>
#include "migraphx_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {
struct MiGraphProviderFactory : IExecutionProviderFactory {
  MiGraphProviderFactory() {}
  ~MiGraphProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<MiGraphExecutionProvider>();
  }
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MiGraph() {
  return std::make_shared<onnxruntime::MiGraphProviderFactory>();
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_MiGraph, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_MiGraph());
  return nullptr;
}
