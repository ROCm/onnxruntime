// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/conv_activation_fusion.h"

#include <string_view>
#include <string>
#include "core/common/inlined_containers.h"
#include "core/framework/tensorprotoutils.h"
#include "core/mlas/inc/mlas.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/selectors_actions/actions.h"

namespace onnxruntime {

namespace {

#if !defined(ORT_MINIMAL_BUILD)
namespace selectors {

const Node* GetLoneConsumerNode(const GraphViewer& graph_viewer, const Node& node) {
  if (!optimizer_utils::CheckOutputEdges(graph_viewer.GetGraph(), node, 1)) {
    return nullptr;
  }
  return &*node.OutputNodesBegin();
}

bool HasElementDataType(const NodeArg& node_arg, int32_t data_type) {
  if (!node_arg.Exists()) {
    return false;
  }

  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto) {
    return false;
  }

  int32_t actual_data_type;
  if (!utils::TryGetElementDataType(*type_proto, actual_data_type)) {
    return false;
  }

  return data_type == actual_data_type;
}

bool ConvFusionDataTypeCheck(const Node& conv_node) {
  // TODO(hasesh): The CPU EP only supports float type for the Conv+Activation
  // and the Conv+Add+Relu fusions.
  // Assess the support level for the other compatible EPs and if they also
  // only support float, remove the EP check altogether.
  const std::string_view node_ep = conv_node.GetExecutionProviderType();
  if (node_ep == kCpuExecutionProvider) {
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
    if (!HasElementDataType(*conv_node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
        !HasElementDataType(*conv_node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)) {
      return false;
    }
#else
    if (!HasElementDataType(*conv_node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
      return false;
    }
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
  }

  return true;
}

class ConvActivationSelector : public NodeSelector {
 public:
  ConvActivationSelector() = default;

  std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const override {
    const std::string_view node_ep = node.GetExecutionProviderType();
    const auto* next_node = GetLoneConsumerNode(graph_viewer, node);
    if (!next_node ||
        next_node->GetExecutionProviderType() != node_ep) {
      return std::nullopt;
    }

    auto is_supported_non_cuda_rocm_ep_activation = [&graph_viewer](const Node& activation_node) {
      if (graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Relu", {6, 13, 14}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Sigmoid", {6, 13}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Tanh", {6, 13}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "LeakyRelu", {6, 16})) {
        return true;
      }

      if (graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Clip", {6, 11, 12, 13})) {
        float min, max;
        if (!optimizer_utils::GetClipConstantMinMax(graph_viewer.GetGraph(), activation_node, min, max)) {
          return false;
        }
        return true;
      }

      return false;
    };

    if (!ConvFusionDataTypeCheck(node)) {
      return std::nullopt;
    }

    // check EP type and activation
    if (node_ep == kCudaExecutionProvider) {
      return std::nullopt;
    } else if (node_ep == kRocmExecutionProvider) {
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(*next_node, "Relu", {6, 13, 14})) {
        return std::nullopt;
      }
    } else if (node_ep.empty() || node_ep == kCpuExecutionProvider || node_ep == kJsExecutionProvider || node_ep == kWebGpuExecutionProvider) {
      if (!is_supported_non_cuda_rocm_ep_activation(*next_node) &&
          !graph_utils::IsSupportedOptypeVersionAndDomain(*next_node, "HardSigmoid", {6})) {
        return std::nullopt;
      }
    } else {
      if (!is_supported_non_cuda_rocm_ep_activation(*next_node)) {
        return std::nullopt;
      }
    }

    NodesToOptimizeIndicesBuilder builder{};
    builder.target_node = node.Index();
    builder.output_nodes.push_back(next_node->Index());
    return builder.Build();
  }
};

}  // namespace selectors
#endif  // !defined(ORT_MINIMAL_BUILD)

namespace actions {

using NTO = NodesToOptimize;

class FuseConvActivationAction : public ReplaceWithNew {
 private:
  std::string OpType(const RuntimeState& runtime_state) const override {
    const auto& domain = runtime_state.selected_nodes.Target().Domain();
    const auto& op_type = runtime_state.selected_nodes.Target().OpType();
    if (domain == kOnnxDomain) {
      if (op_type == "Conv") {
        return "FusedConv";
      }
    } else if (domain == kMSDomain) {
      if (op_type == "NhwcConv") {
        return "NhwcFusedConv";
      }
    } else if (domain == kMSInternalNHWCDomain) {
      if (op_type == "Conv") {
        return "Conv";
      }
    }
    ORT_THROW("Unsupported operator: ", op_type, " and domain: ", domain);
  }

  std::string Domain(const RuntimeState& runtime_state) const override {
    auto domain = runtime_state.selected_nodes.Target().Domain();
    return domain == kOnnxDomain ? kMSDomain : domain;
  }

  NodeAttributes ExtraAttributes(const RuntimeState& state) const override {
    NodeAttributes extra_fused_conv_attributes;

    const auto* activation = state.selected_nodes.Output(0);
    ORT_ENFORCE(activation != nullptr, "Expected activation node.");

    const auto& activation_op_type = activation->OpType();
    utils::SetNodeAttribute(utils::MakeAttribute("activation", activation_op_type), extra_fused_conv_attributes);

    InlinedVector<float> activation_params;
    if (activation_op_type == "LeakyRelu") {
      activation_params.push_back(graph_utils::GetNodeAttribute(*activation, "alpha")->f());
    } else if (activation_op_type == "Clip") {
      float min, max;
      ORT_ENFORCE(optimizer_utils::GetClipConstantMinMax(state.graph, *activation, min, max),
                  "Failed to get Clip min/max constants.");
      activation_params.push_back(min);
      activation_params.push_back(max);
    } else if (activation_op_type == "HardSigmoid") {
      auto* alpha_attr = graph_utils::GetNodeAttribute(*activation, "alpha");
      auto* beta_attr = graph_utils::GetNodeAttribute(*activation, "beta");
      float alpha = (alpha_attr == nullptr ? 0.2f : alpha_attr->f());
      float beta = (beta_attr == nullptr ? 0.5f : beta_attr->f());
      activation_params.push_back(alpha);
      activation_params.push_back(beta);
    }

    if (!activation_params.empty()) {
      utils::SetNodeAttribute(utils::MakeAttribute("activation_params", activation_params),
                              extra_fused_conv_attributes);
    }

    return extra_fused_conv_attributes;
  }

  std::vector<NodeAndMoveInfo> ValueMoves(const RuntimeState&) const override {
    const NTO::NodeLocation conv{NTO::NodeType::kTarget, 0};
    const NTO::NodeLocation activation{NTO::NodeType::kOutput, 0};

    return {
        MoveAll(conv, ArgType::kInput),         // move all inputs from conv
        MoveAll(activation, ArgType::kOutput),  // move all outputs from activation
    };
  }
};

}  // namespace actions

void RegisterConvActivationFusionRules(SelectorActionRegistry& registry) {
  const auto name = "ConvAct";
  auto action = std::make_unique<actions::FuseConvActivationAction>();
#if !defined(ORT_MINIMAL_BUILD)
  const std::string msInternalNHWCDomainConv = SelectorActionRegistry::OpVersionsMapKey("Conv", kMSInternalNHWCDomain);
  const std::string msDomainConv = SelectorActionRegistry::OpVersionsMapKey("NhwcConv", kMSDomain);
  auto selector = std::make_unique<selectors::ConvActivationSelector>();

  registry.RegisterSelectorAndAction(name, {{"Conv", {1, 11}}, {msInternalNHWCDomainConv, {1, 11}}, {msDomainConv, {1}}},
                                     std::move(selector), std::move(action));
#else
  registry.RegisterAction(name, std::move(action));
#endif
}

SelectorActionRegistry CreateSelectorActionRegistry() {
  SelectorActionRegistry registry{};
  RegisterConvActivationFusionRules(registry);
  return registry;
}

}  // namespace

ConvActivationFusion::ConvActivationFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                           const SatApplyContextVariant& apply_context)
    : SelectorActionTransformer{
          "ConvActivationFusion", CreateSelectorActionRegistry(), apply_context, compatible_execution_providers} {
}

}  // namespace onnxruntime
