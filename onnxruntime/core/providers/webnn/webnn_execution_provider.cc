// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "webnn_execution_provider.h"

#include "core/framework/compute_capability.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/memcpy.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph_viewer.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/webnn/allocator.h"
#include "core/providers/webnn/data_transfer.h"
#include "core/providers/partitioning_utils.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"

#include "builders/model.h"
#include "builders/helper.h"
#include "builders/model_builder.h"

namespace onnxruntime {

constexpr const char* WEBNN = "WEBNN";

WebNNExecutionProvider::WebNNExecutionProvider(const std::string& webnn_device_flags)
    : IExecutionProvider{
          onnxruntime::kWebNNExecutionProvider,
          // If MLTensor is supported, we force all the tensors to be allocated as MLTensor.
          OrtDevice(
              webnn::IsMLTensorSupported() ? OrtDevice::GPU : OrtDevice::CPU,
              OrtDevice::MemType::DEFAULT,
              OrtDevice::VendorIds::NONE,
              0)},
      wnn_device_type_(webnn::DeviceTypeFromString(webnn_device_flags)) {
  wnn_context_ = emscripten::val::module_property("currentContext");
  if (!wnn_context_.as<bool>()) {
    ORT_THROW("Failed to create WebNN context.");
  }

  // Retrieve the level of support for different WebNN operators.
  // This varies across implementations and is obtained via the WebNN's opSupportLimits() function.
  // https://www.w3.org/TR/webnn/#api-mlcontext-opsupportlimits
  wnn_limits_ = wnn_context_.call<emscripten::val>("opSupportLimits");

  if (wnn_limits_["preferredInputLayout"].as<std::string>().compare("nhwc") == 0) {
    preferred_layout_ = DataLayout::NHWC;
  } else {
    preferred_layout_ = DataLayout::NCHW;
  }
}

WebNNExecutionProvider::~WebNNExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
WebNNExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                      const IKernelLookup& /*kernel_registries*/,
                                      const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                      IResourceAccountant* /* resource_accountant */) const {
  // For subgraph which is the attribute of the control flow nodes, part of its initializers are stored in its
  // ancestor graphs as common initializers shared for other subgraphs. We need to collect all of them used for
  // identifying the required initializer names and storing into 'meta_def->constant_initializers'.
  // Thus we are able to get the required initialized tensors for this subgraph via the GetInitializerTensors()
  // method defined in the model_builder.h file.
  InitializedTensorSet all_initializers;
  const bool is_subgraph = graph_viewer.IsSubgraph();
  if (is_subgraph) {
    all_initializers = webnn::CollectAllInitializedTensors(graph_viewer);
  }

  const auto& logger = *GetLogger();

  emscripten::val wnn_builder = emscripten::val::global("MLGraphBuilder").new_(wnn_context_);
  if (!wnn_builder.as<bool>()) {
    ORT_THROW("Failed to create WebNN builder.");
  }

  // Get all the NodeUnits in the graph_viewer
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(graph_viewer, logger);

  const auto supported_nodes = webnn::GetSupportedNodes(graph_viewer, wnn_builder, wnn_device_type_, wnn_limits_, logger);

  const auto gen_metadef_name = [&]() {
    HashValue model_hash;
    int metadef_id = metadef_id_generator_.GenerateId(graph_viewer, model_hash);
    return MakeString(WEBNN, "_", model_hash, "_", metadef_id);
  };

  auto result = utils::CreateSupportedPartitions(graph_viewer, supported_nodes, {},
                                                 gen_metadef_name, WEBNN, kWebNNExecutionProvider,
                                                 &node_unit_map, /*drop_constant_initializers*/ true);

  // Release wnn_builder
  wnn_builder = emscripten::val::undefined();

  const auto& graph_output_list = graph_viewer.GetOutputs();
  InlinedHashSet<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());

  for (auto& capability : result) {
    auto& sub_graph = capability->sub_graph;
    if (sub_graph->nodes.empty())
      continue;

    std::vector<std::string> subgraph_initializers;
    for (const auto& index : sub_graph->nodes) {
      const auto* node = graph_viewer.GetNode(index);

      for (const auto* input : node->InputDefs()) {
        if (!input->Exists()) {
          // skip the placeholder inputs.
          continue;
        }
        // If it is a subgraph of a control flow node, collect the constant initializer.
        if (is_subgraph && Contains(all_initializers, input->Name())) {
          subgraph_initializers.push_back(input->Name());
        }
      }
    }

    // Assign inputs and outputs to subgraph's meta_def.
    uint64_t model_hash;
    int metadef_id = metadef_id_generator_.GenerateId(graph_viewer, model_hash);
    const auto meta_def_old = sub_graph->GetMetaDef();
    auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
    meta_def->name = "WEBNN_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
    meta_def->domain = kMSDomain;
    meta_def->since_version = 1;
    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;

    if (is_subgraph) {
      for (const auto& initializer : subgraph_initializers) {
        meta_def->constant_initializers.push_back(initializer);
      }
    }

    for (const auto& input : meta_def_old->inputs) {
      meta_def->inputs.push_back(input);
    }

    for (const auto& output : meta_def_old->outputs) {
      meta_def->outputs.push_back(output);
    }

    sub_graph->SetMetaDef(std::move(meta_def));
  }

  const auto num_of_partitions = result.size();
  const auto num_of_supported_nodes = std::accumulate(
      result.begin(), result.end(), size_t{0},
      [](const auto& acc, const auto& partition) -> size_t {
        return acc + (partition && partition->sub_graph ? partition->sub_graph->nodes.size() : 0);
      });

  const auto summary_msg = MakeString(
      "WebNNExecutionProvider::GetCapability,",
      " number of partitions supported by WebNN: ", num_of_partitions,
      " number of nodes in the graph: ", graph_viewer.NumberOfNodes(),
      " number of nodes supported by WebNN: ", num_of_supported_nodes);

  // If the graph is partitioned in multiple subgraphs, and this may impact performance,
  // we want to give users a summary message at warning level.
  if (num_of_partitions > 1) {
    LOGS(logger, WARNING) << summary_msg;
  } else {
    LOGS(logger, INFO) << summary_msg;
  }

  return result;
}

common::Status WebNNExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                               std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_and_graph.fused_node;
    const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);

    webnn::ModelBuilder builder(graph_viewer, *GetLogger(), wnn_context_,
                                preferred_layout_, wnn_device_type_, wnn_limits_);
    std::unique_ptr<webnn::Model> model;
    ORT_RETURN_IF_ERROR(builder.Compile(model));

    // Build map from input name to its index in input definitions.
    {
      InlinedHashMap<std::string, size_t> input_map;
      const auto& input_defs = fused_node.InputDefs();
      input_map.reserve(input_defs.size());
      for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
        input_map[input_defs[i]->Name()] = i;
      }
      model->SetInputMap(std::move(input_map));
    }
    // Build map from output name to its index in output definitions.
    {
      InlinedHashMap<std::string, size_t> output_map;
      const auto& output_defs = fused_node.OutputDefs();
      output_map.reserve(output_defs.size());
      for (size_t i = 0, end = output_defs.size(); i < end; ++i) {
        output_map[output_defs[i]->Name()] = i;
      }
      model->SetOutputMap(std::move(output_map));
    }
    models_.emplace(fused_node.Name(), std::move(model));

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
      *state = models_[context->node_name].get();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      // The `state` is a webnn::model managed by unique_ptr.
      ORT_UNUSED_PARAMETER(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      Ort::KernelContext ctx(context);

      const size_t num_inputs = ctx.GetInputCount();
      const size_t num_outputs = ctx.GetOutputCount();

      webnn::Model* model = reinterpret_cast<webnn::Model*>(state);

      const auto& model_inputs = model->GetInputs();
      const auto& model_outputs = model->GetOutputs();

      ORT_RETURN_IF_NOT(model_inputs.size() <= num_inputs, "Inconsistent input sizes");
      ORT_RETURN_IF_NOT(model_outputs.size() == num_outputs, "Inconsistent output sizes");

      InlinedHashMap<std::string, webnn::OnnxTensorData> inputs;
      inputs.reserve(model_inputs.size());
      for (size_t i = 0; i < model_inputs.size(); i++) {
        const auto& input_name = model_inputs[i];
        auto input_idx = model->GetMappedInputIdx(input_name);
        auto input_tensor = ctx.GetInput(input_idx);
        auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        const void* inputBuffer = const_cast<void*>(input_tensor.GetTensorRawData());
        inputs.emplace(
            input_name,
            webnn::OnnxTensorData{
                webnn::OnnxTensorInfo{tensor_info.GetElementType(), shape},
                const_cast<void*>(inputBuffer),
            });
      }

      // From this point we will need to take the exclusive lock on the model until the Predict is
      // performed, to block other threads to perform Predict on the same model.
      // TODO, investigate concurrent runs for different executions from the same model.
      {
        std::unique_lock<std::mutex> lock(model->GetMutex());
        InlinedHashMap<std::string, webnn::OnnxTensorData> outputs;
        outputs.reserve(model_outputs.size());
        for (size_t i = 0; i < model_outputs.size(); i++) {
          const auto& output_name = model_outputs[i];
          const auto& output_info = model->GetInputOutputInfo(output_name);
          auto output_shape = output_info.shape;
          auto output_type = output_info.data_type;
          auto output_tensor =
              ctx.GetOutput(i, output_shape.data(), output_shape.size());
          void* output_buffer = output_tensor.GetTensorMutableRawData();
          outputs.emplace(output_name,
                          webnn::OnnxTensorData{
                              webnn::OnnxTensorInfo{output_type, output_shape},
                              output_buffer,
                          });
        }

        return model->Predict(inputs, outputs);
      }
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

class WebNNMemcpy : public OpKernel {
 public:
  explicit WebNNMemcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    auto webnnEnsureTensor = emscripten::val::module_property("webnnEnsureTensor");
    const auto* X = context->Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr, "Memcpy: input tensor is null");
    auto* Y = context->Output(0, X->Shape());
    ORT_ENFORCE(X != nullptr, "Memcpy: output tensor is null");
    emscripten::val shape = emscripten::val::array();
    for (auto dim : X->Shape().GetDims()) {
      shape.call<void>("push", SafeInt<uint32_t>(dim).Ref());
    }

    webnnEnsureTensor(emscripten::val::undefined(),
                      reinterpret_cast<intptr_t>(Y->MutableDataRaw()),
                      Y->GetElementType(),
                      shape, false)
        .await();

    const auto* data_transfer = Info().GetDataTransferManager().GetDataTransfer(X->Location().device, Y->Location().device);

    return data_transfer->CopyTensor(*X, *Y);
  }
};

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kWebNNExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    WebNNMemcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kWebNNExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kWebNNExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kWebNNExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static void RegisterWebNNKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebNNExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebNNExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    ORT_ENFORCE(kernel_registry.Register(function_table_entry()).IsOK());
  }
}

std::shared_ptr<KernelRegistry> GetWebNNKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry =
      std::make_shared<KernelRegistry>();
  RegisterWebNNKernels(*kernel_registry);

  return kernel_registry;
}

std::shared_ptr<KernelRegistry>
WebNNExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry =
      onnxruntime::GetWebNNKernelRegistry();
  return kernel_registry;
}

std::unique_ptr<onnxruntime::IDataTransfer> WebNNExecutionProvider::GetDataTransfer() const {
  if (!webnn::IsMLTensorSupported()) {
    return nullptr;
  }
  return std::make_unique<webnn::DataTransfer>();
}

std::vector<AllocatorPtr> WebNNExecutionProvider::CreatePreferredAllocators() {
  if (!webnn::IsMLTensorSupported()) {
    return {};
  }
  AllocatorCreationInfo customAllocatorCreationInfo([&](OrtDevice::DeviceId) {
    return std::make_unique<webnn::WebNNTensorAllocator>();
  },
                                                    0, false);
  return {CreateAllocator(customAllocatorCreationInfo)};
}

}  // namespace onnxruntime
