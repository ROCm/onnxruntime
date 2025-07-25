// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <set>
#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/model/host_utils.h"
#include "core/providers/shared/utils/utils.h"

using namespace CoreML::Specification;

namespace onnxruntime {
namespace coreml {

Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                                        const logging::Logger& logger) const {
  Status status = AddToModelBuilderImpl(model_builder, node, logger);

  if (status.IsOK()) {
    LOGS(logger, VERBOSE) << "Operator name: [" << node.Name() << "] type: [" << node.OpType() << "] was added";
  }

  return status;
}

bool BaseOpBuilder::IsOpSupported(const Node& node, const OpBuilderInputParams& input_params,
                                  const logging::Logger& logger) const {
  if (input_params.create_mlprogram && !SupportsMLProgram()) {
    LOGS(logger, VERBOSE) << "Operator [" << node.OpType() << "] does not support MLProgram";
    return false;
  }

  if (!HasSupportedOpSet(node, logger)) {
    LOGS(logger, VERBOSE) << "Operator [" << node.OpType() << "] does not support this opset";
    return false;
  }

  if (!HasSupportedInputs(node, input_params, logger)) {
    LOGS(logger, VERBOSE) << "Operator [" << node.OpType() << "] has unsupported inputs";
    return false;
  }

  if (!IsOpSupportedImpl(node, input_params, logger)) {
    LOGS(logger, VERBOSE) << "Operator [" << node.OpType() << "] is not supported by the impl";
    return false;
  }
  return true;
}

bool BaseOpBuilder::HasSupportedInputs(const Node& node, const OpBuilderInputParams& input_params,
                                       const logging::Logger& logger) const {
  for (const auto* input : node.InputDefs()) {
    if (!IsInputSupported(node, *input, input_params, logger, allow_empty_tensor_as_input_)) {
      return false;
    }
  }

  return HasSupportedInputsImpl(node, input_params, logger);
}

/* static */
bool BaseOpBuilder::IsInputDtypeSupport(const Node& node, size_t idx,
                                        [[maybe_unused]] const OpBuilderInputParams& input_params,
                                        const logging::Logger& logger) {
  if (idx >= node.InputDefs().size()) {
    LOGS(logger, VERBOSE) << "Input index [" << idx << "] is out of range";
    return false;
  }

  const auto& input = *node.InputDefs()[idx];

  int32_t input_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;

  if (!GetType(input, input_type, logger)) {
    LOGS(logger, VERBOSE) << "[" << node.OpType() << "] Get Input type failed";
    return false;
  }

  // float is supported
  if (input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return true;
  }

#if CAN_BUILD_COREML6_OR_LATER
  // only MLProgram support FP16 and INT64
  if (input_params.create_mlprogram && (input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
                                        input_type == ONNX_NAMESPACE::TensorProto_DataType_INT64)) {
    return true;
  }
#endif

  LOGS(logger, VERBOSE) << "[" << node.OpType() << "] Input type: [" << input_type << "] is not currently supported";
  return false;
}

bool BaseOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                                           const logging::Logger& logger) const {
  // We only check the type of input 0 by default
  // specific op builder can override this
  return IsInputDtypeSupport(node, 0, input_params, logger);
}

bool BaseOpBuilder::HasSupportedOpSet(const Node& node, const logging::Logger& logger) const {
  auto since_version = node.SinceVersion();
  if (since_version < GetMinSupportedOpSet(node) || since_version > GetMaxSupportedOpSet(node)) {
    LOGS(logger, VERBOSE) << node.OpType() << "is only supported for opset ["
                          << GetMinSupportedOpSet(node) << ", "
                          << GetMaxSupportedOpSet(node) << "]";
    return false;
  }

  return true;
}

}  // namespace coreml
}  // namespace onnxruntime
