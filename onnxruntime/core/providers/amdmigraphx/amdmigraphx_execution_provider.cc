// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "migraphx_inc.h"
#include "amdmigraphx_execution_provider.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, std::min(destsz, srcsz))

namespace onnxruntime {

constexpr const char* MIGRAPHX = "MiGraphX";

MiGraphExecutionProvider::MiGraphExecutionProvider(MiGraphExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::AmdMiGraphXExecutionProvider} {

  // ORT_ENFORCE(info.target_device == "CPU", "nGraph Execution Provider for onnxruntime currently is only supported for CPU backend.");
  // Set GPU device to be used
  hipSetDevice(info.device_id);

  DeviceAllocatorRegistrationInfo default_memory_info(
      {OrtMemTypeDefault, [](int id) { return onnxruntime::make_unique<HIPAllocator>(id, TRT); }, std::numeric_limits<size_t>::max()});
  allocator_ = CreateAllocator(default_memory_info, device_id_);
  InsertAllocator(allocator_);

  DeviceAllocatorRegistrationInfo cpu_memory_info{
    OrtMemTypeCPUOutput,
    std::move(cpu_allocator_factory),
    std::numeric_limits<size_t>::max()
  };

  InsertAllocator(CreateAllocator(cpu_memory_info));

  // create the target based on the device_id
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, device_id_);

  // // GPU device
  // if (contains(prop.name, "gfx"))
  // {
  //   t = migraphx::gpu::target{};
  // }
  // else
  // {
  //   t = migraphx::cpu::target{};
  // }  

  if (info.target_device == "cpu")
  {
    t = migraphx::cpu::target{};    
  }
  else if (info.target_device == "gpu")
  {
    t = migraphx::gpu::target{};
  }
  else
  {
    LOGS_DEFAULT(FATAL) << "Device " << info.ng_backend_type << " are not supported";    
  }
}

// Returns true only if op is in a mode that is not currently supported
static bool IsUnsupportedOpMode(const Node* node, const onnxruntime::GraphViewer& graph_viewer) {
  const auto& optype = node->OpType();
  const auto& initializers = graph_viewer.GetAllInitializedTensors();

  // To do: add unsuppored mode in MIGRAPHX later
  return false;
}

static bool IsTypeSupported(const NodeArg* node_arg) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  switch (type_proto->tensor_type().elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
      return true;
    default:
      return false;
  }
}

static migraphx::shape::type_t get_migraphx_type(ONNXTensorElementDataType type)
{
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
      return migraphx::shape::half_type;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
      return migraphx::shape::float_type;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE:
      return migraphx::shape::double_type;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
      return migraphx::shape::int8_type;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
      return migraphx::shape::int16_type;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
      return migraphx::shape::int32_type;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
      return migraphx::shape::int64_type;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
      return migraphx::shape::uint8_type;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
      return migraphx::shape::uint16_type;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
      return migraphx::shape::uint32_type;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
      return migraphx::shape::uint64_type;
    default:
      MIGRAPHX_THROW("Migraphx: unsupported data type");
}

static bool IsNodeSupported(const std::set<std::string>& op_set,
                            const onnxruntime::GraphViewer& graph_viewer,
                            const NodeIndex node_idx) {
  const auto& node = graph_viewer.GetNode(node_idx);
  const auto& optype = node->OpType();
  const auto& domain = node->Domain();

  // 1. Check input and output data types are supported.
  // 2. Check Op is supported

  //Check 1
  bool are_types_supported = true;

  node->ForEachDef([&are_types_supported](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
    are_types_supported &= IsTypeSupported(&node_arg);
  });

  if (!are_types_supported) {
    return false;
  }

  //Check 2a
  if (domain == kOnnxDomain && IsUnsupportedOpMode(node, graph_viewer)) {
    return false;
  }

  //Check 2b
  if (op_set == op_map.end() || op_set->second.find(optype) == op_set->second.end()) {
    return false;
  } else {
    return true;
  }
}

static void AppendNodesToSubGraph(const std::vector<NodeIndex>& nodes,
                                    const std::vector<std::string>& inputs,
                                    const std::vector<std::string>& outputs,
                                    std::vector<std::unique_ptr<ComputeCapability>>& result) {
  static size_t op_counter = 0;

  auto meta_def = onnxruntime::make_unique<IndexedSubGraph::MetaDef>();
  meta_def->name = "MIGraphX_" + std::to_string(++op_counter);
  meta_def->domain = kMiGraphXDomain;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs = inputs;
  meta_def->outputs = outputs;

  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
  sub_graph->nodes = nodes;
  sub_graph->SetMetaDef(meta_def);
  result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
}

static int GetOnnxOpSet(const GraphViewer& graph_viewer) {
  const auto& dm_to_ver = graph_viewer.DomainToVersionMap();
  return dm_to_ver.at(kOnnxDomain);
}

static std::set<std::string> GetMiGraphXSupportedOps() {
  std::map<std::string, std::set<std::string>> mgx_supported_ops = migraphx::get_supported_ops();
  return mgx_supported_ops;
}



static std::vector<NodeIndex>
GetUnsupportedNodeIndices(const GraphViewer& graph_viewer, /*out*/ std::unordered_set<std::string>& mgx_required_initializers) {
  const auto mgx_supported_ops = GetMiGraphXSupportedOps();

  std::vector<NodeIndex> unsupported_nodes_idx;

  for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    if (IsNodeSupported(mgx_supported_ops, graph_viewer, node_idx)) {
      // Collect inputs that are initializers
      graph_viewer.GetNode(node_idx)->ForEachDef([&mgx_required_initializers, &graph_viewer](const onnxruntime::NodeArg& node_arg, bool is_input) {
              if(is_input && graph_viewer.GetAllInitializedTensors().count(node_arg.Name())) {
                mgx_required_initializers.insert(node_arg.Name());
              } }, true);
    } else {
      unsupported_nodes_idx.push_back(node_idx);
    }
  }

  return unsupported_nodes_idx;
}

// Returns a vector clusters(or node_idx). For each unsupported node, the graph
// is split into 3 parts. supported_cluster + (UNsupported_node + rest_of_the_graph). 
// This functions returns vector of all supported_subgraphx by amdmigraphx
static std::vector<std::vector<NodeIndex>>
GetPartitionedSubGraphs(const std::vector<NodeIndex>& topological_order, const std::vector<NodeIndex>& unsupported_nodes) {
  std::vector<std::vector<NodeIndex>> mgx_subgraphx;

  auto prev = topological_order.begin();

  for (const auto& unsup_node : unsupported_nodes) {
    auto it = std::find(prev, topological_order.end(), unsup_node);
    // Create a cluster vector[supported_node_idx, unsupported_node_idx) 
    // and append it to return list.
    std::vector<NodeIndex> this_subgraph{prev, it};
    if (!this_subgraph.empty()) {
      mgx_subgraphx.push_back(std::move(this_subgraph));
    }
    // Point prev to node idx past this unsuported node.
    prev = ++it;
  }

  // Tail
  std::vector<NodeIndex> this_subgraph{prev, topological_order.end()};
  if (!this_subgraph.empty()) {
    mgx_subgraphx.push_back(std::move(this_subgraph));
  }

  return mgx_subgraphx;
}

static void GetInputsOutputsOfSubgraph(const GraphViewer& graph_viewer,
                                      const std::vector<NodeIndex>& nodes,
                                      const std::unordered_set<std::string>& mgx_required_initializers,
                                      /*output*/ std::vector<std::string>& nodes_inputs,
                                      /*output*/ std::vector<std::string>& nodes_outputs) {
  std::unordered_set<std::string> input_args;
  std::vector<std::string> ordered_input_args;
  std::unordered_set<std::string> output_args;
  std::unordered_set<std::string> external_output_args;
s
  for (const auto& node_idx : nodes) {
    const auto& node = graph_viewer.GetNode(node_idx);

    // Collect all inputs and outputs
    node->ForEachDef(
        [&input_args, &ordered_input_args, &output_args](const NodeArg& node_arg, bool is_input) {
          if (is_input) {
            if (!input_args.count(node_arg.Name())) {
              ordered_input_args.push_back(node_arg.Name());
            }
            input_args.insert(node_arg.Name());
          } else {
            output_args.insert(node_arg.Name());
          }
        },
        true);

    // Check if output of this node is used by nodes outside 
    // subgraph. If yes add this to cluster outputs
    for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
      const auto& ext_node = graph_viewer.GetNode((*it).Index());

      if (std::find(nodes.begin(), nodes.end(), ext_node->Index()) == nodes.end()) {
        // Node is external to subgraph. Search through its 
        // inputs to find the output that is generated by subgraph.
        std::set<std::string> ext_node_inputs;
        ext_node->ForEachDef(
            [&ext_node_inputs](const onnxruntime::NodeArg& arg, bool is_input) {
              if (is_input) {
                ext_node_inputs.insert(arg.Name());
              }
            },
            true);

        for (const auto& out_def : node->OutputDefs()) {
          if (ext_node_inputs.find(out_def->Name()) != ext_node_inputs.end()) {
            external_output_args.insert(out_def->Name());
          }
        }
      }
    }
  }

  //Extract initializers used by subgraph.
  std::unordered_set<std::string> original_graph_inputs;
  for (const auto& node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    original_graph_inputs.insert(node_arg->Name());
  }

  const auto& initializers = graph_viewer.GetAllInitializedTensors();
  std::vector<std::string> const_inputs;
  for (const auto& in_arg : ordered_input_args) {
    if ((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
        mgx_required_initializers.count(in_arg)) {
      const_inputs.push_back(in_arg);
    }
  }

  for (const auto& in_arg : ordered_input_args) {
    if (!output_args.count(in_arg) &&
        !((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
        mgx_required_initializers.count(in_arg))) {
      nodes_inputs.push_back(in_arg);
    }
  }

  for (const auto& in_arg : const_inputs) {
    nodes_inputs.push_back(in_arg);
  }

  std::copy(external_output_args.begin(), external_output_args.end(), std::back_inserter(cluster_outputs));
  for (const auto& node_arg : graph_viewer.GetOutputs()) {
    const auto& name = node_arg->Name();
    if (output_args.count(name) && !external_output_args.count(name)) {
      nodes_outputs.push_back(name);
    }
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
MiGraphXExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {

  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  // Need access to model_path_
  for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() && tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(WARNING) << "MIGraphX: Initializers with external data location are not currently supported";
      return result;
    }
  }

  // Construct modelproto from graph
  onnxruntime::Model model(graph.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), graph.DomainToVersionMap());
  onnxruntime::Graph& graph_build = model.MainGraph();
  for (const auto& node : graph.Nodes()) {
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (auto input : node.InputDefs()) {
      auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
    }
    for (auto output : node.OutputDefs()) {
      auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    graph_build.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, &node.GetAttributes(), node.Domain());
  }

  auto status = graph_build.Resolve();

  //Add initializer to graph
  const auto& init_tensors = graph.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    graph_build.AddInitializedTensor(*(tensor.second));
  }

  ORT_ENFORCE(status.IsOK(), status);
  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string string_buf;
  model_proto.SerializeToString(&string_buf);

  // may not be needed since it can return false in many scenarios
  bool ret = migraphx::parse_model_string(string_buf);
  if (!ret)
  {
    return result;
  }

  // This is a list of initializers that migraphx considers as constants. 
  // Example weights, reshape shape etc.
  std::unordered_set<std::string> mgx_required_initializers;
  const auto unsupported_nodes = GetUnsupportedNodeIndices(graph_viewer, mgx_required_initializers);

  //If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
  if (unsupported_nodes.empty()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    //Fill inputs with names
    std::for_each(graph_viewer.GetInputs().begin(), graph_viewer.GetInputs().end(),
                  [&inputs](const NodeArg* node_arg) { inputs.push_back(node_arg->Name()); });

    // In scenarios, when there are no inputs or all inputs being initializers,
    // ConstantFolding optimization in onnxruntime pre-computes the value.
    if (inputs.empty()) {
      return result;
    }

    // Initializers need to be part of meta_def->inputs
    std::for_each(mgx_required_initializers.begin(), mgx_required_initializers.end(),
                  [&inputs](const std::string& initializer) { inputs.push_back(initializer); });

    // Fill outputs with names
    std::for_each(graph_viewer.GetOutputs().begin(), graph_viewer.GetOutputs().end(),
                  [&outputs](const NodeArg* node_arg) { outputs.push_back(node_arg->Name()); });

    // Create and add this graph to result.
    AppendNodesToSubGraph(graph_viewer.GetNodesInTopologicalOrder(), graph_viewer, inputs, outputs, result);

  } else {  // unsupported_nodes_idx.empty()
    const auto ng_clusters = GetPartitionedSubgraphs(graph_viewer.GetNodesInTopologicalOrder(), unsupported_nodes);

    for (const auto& this_cluster : ng_clusters) {
      std::vector<std::string> cluster_inputs, cluster_outputs;
      GetInputsOutputsOfSubgraph(graph_viewer, this_cluster, ng_required_initializers, cluster_inputs, cluster_outputs);

      if (!cluster_inputs.empty()) {
        AppendNodesToSubGraph(this_cluster, graph_viewer, cluster_inputs, cluster_outputs, result);
      }
    }
  }

  return result;
}

static ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(const onnxruntime::Node* fused_node) {
  const auto* node_function = fused_node->GetFunctionBody();

  ORT_ENFORCE(node_function != nullptr, "Could not extract function body for node: ", fused_node->Name());

  const Graph& node_subgraph = node_function->Body();
  onnxruntime::Model model{node_subgraph.Name(), true};

  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  *(model_proto.mutable_graph()) = node_subgraph.ToGraphProto();

  auto opset = model_proto.add_opset_import();
  opset->set_domain(kOnnxDomain);
  opset->set_version(node_subgraph.DomainToVersionMap().at(kOnnxDomain));

  return model_proto;
}

Status MiGraphXExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                        std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node : fused_nodes) {
    // record name of each input
    std::vector<std::string> input_names;
    const auto& input_defs = fused_node->InputDefs();
    input_names.reserve(input_defs.size());
    for (std::size_t i = 0; i < input_defs.size(); ++i) {
      input_names.push_back(input_defs[i]->Name());
    }
    map_input_names_[fused_node->Name()].push_back(input_names);

    // record name of each output
    std:vectors<std::string> output_names;
    const auto& output_defs = fused_node->OutputDefs();
    output_names.reserve(output_defs.size());
    for (std::size_t i = 0; i < output_defs.size(); ++i) {
      output_names.push_back(output_defs[i]->Name());
    }
    map_output_names_[fused_node->Name()].push_back(output_names);

    // reconstruct the subgraph proto from fused nodes
    onnx::ModelProto model_proto = GetModelProtoFromFusedNode(fused_node);
    std::string string_buf;
    model_proto.SerializeToString(&string_buf);

    // by parsing the model_proto, create a program corresponding to
    // the input fused_node
    std::vector<std::string> unsupported_nodes;
    migraphx::program prog = migraphx::parse_model(string_buf, unsupported_nodes);

    // compile the program
    prog.compile(t);
    map_prog_[fused_node->Name()].push_back(prog);

    std::unordered_map<std::string, migraphx::shape> param_shapes = prog.get_parameter_shapes();
    migraphx::program::parameter_map m;
    m.reserve(param_shapes);
    auto num_inputs = param_shapes.size();
    for (auto&& x : prog.get_parameter_shape()) {
      const std::string& name = x.first;
      size_t bindingIndex = trt_engine->getBindingIndex(name.c_str());
      nvinfer1::Dims dimensions = trt_engine->getBindingDimensions(static_cast<int>(bindingIndex));
      auto iter = input_map.find(name);
      if (iter != input_map.end()) {
        input_indexes[bindingIndex] = iter->second;
      }
      size_t dim_size = 1;
      for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
        dim_size *= dimensions.d[j];
      }
      input_dim_sizes[bindingIndex] = dim_size;
    }

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      std::unique_ptr<TensorrtFuncState> p = onnxruntime::make_unique<MiGraphXFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocate_handle, map_prog_[context->node_name], 
            map_input_names_[context->node_name], map_output_names_[context->node_name], &mgx_mu_};
      *state = p.release();
      return 0;
    }

    compute_info.release_state_func = [](FunctionState state) {F
      if (state)
        delete static_cast<MiGraphXFuncState*>(state);
    };

struct MiGraphXFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocate_handle = nullptr;
  migprahx::program prog{};
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  OrtMutex* mgx_mu_ptr = nullptr;
};

    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      Ort::CustomOpApi ort{*api};
      MiGraphXFuncState* mgx_state = reinterpret_cast<MiGraphXFuncState*>(state);
      std::vector<std::string>& input_names = mgx_state->input_names;
      std::vector<std::string>& output_names = mgx_state->output_names;
      migraphx::program prog = mgx_state->prog;

      std::unordered_map<std::string, migraphx::shape> param_shapes = prog.get_parameter_shapes();
      migraphx::program::parameter_map m;
      m.reserve(param_shapes);

      std::size_t input_num = input_names.size();
      for (std::size_t i = 0; i < input_num; ++i)
      {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        const auto& tensor_shape = ort.GetTensorShape(tensor_info);
        auto tensor_type = ort.GetTensorElementType(tensor_info);
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

        auto param_name = input_names[i];
        auto mgx_type = get_migraphx_type(tensor_type);
        auto mgx_s = param_shapes[param_name];

        if (mgx_type != mgx_s.type())
        {
          MIGRAPHX_THROW("MIGraphX: param type mismatch");
        }
        m[param_name] = migraphx::argument(param_shapes[param_name], const_cast<void*>(ort.GetTensorData<void>(input_tensor)));
      }

      for (auto&& x : param_shapes)
      {
        // scratch memory
        if (!contains(m, x.first))
        {
          m[param_name] = t.copy_to(migraphx::generate_argument(x.sceond));
        }
      }

      // migraphx can only handle one output now
      {
        std::lock_guard<std::mutex> lock(mgx_mu_);
        auto result = prog.eval(m);
        auto res_shape = result.get_shape();
        auto dtype = res_shape.type();

        unsigned int output_index = 0;
        std::vector<int64_t> ort_shape{res_shape.lens().begin(), res_shape.lens().end()};
        OrtValue* output_tensor = ort.KernelContext_GetOutput(context, output_index++, ort_shape.data(), ort_shape.size());
        void* output_data = ort.GetTensorMutableData<void>(output_tensor);

        // copy data from result to output_data
        hipMemcpy(output_data, result.data(), res_shape.bytes(), hipMemcpyDeviceToDevice);
      }
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

}  // namespace onnxruntime
