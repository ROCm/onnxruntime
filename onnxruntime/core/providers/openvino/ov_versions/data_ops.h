// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once
#include <unordered_set>
#include <utility>
#include <map>
#include <set>
#include <vector>
#include <string>

namespace onnxruntime {
namespace openvino_ep {

using VarianceFunc = std::function<bool(const Node*, const InitializedTensorSet&)>;

enum versionNum {
  V_2020_4,
  V_2021_1,
  V_2021_2,
  V_2021_3,
  V_2021_4,
  V_2022_1,
  V_2022_2,
  V_2022_3,
  V_2023_0,
  V_2023_1,
  V_2023_2,
  V_2023_3,
  V_2024_0,
  V_2024_1,
  V_2024_2,
  V_2024_3,
  V_2024_4,
  V_2024_5,
  V_2024_6,
  V_2025_0,
  V_2025_1,
  V_2025_2
};

using VersionNum = enum versionNum;

struct supportedOp {
  std::string optype;
  VersionNum version;
  std::vector<std::string> device_type;
};

struct unsupportedOpMode {
  std::vector<VersionNum> ver;
  VarianceFunc func;
};

using SupportedOp = struct supportedOp;
using UnsupportedOpMode = struct unsupportedOpMode;
using Pairs = std::pair<VersionNum, int>;

class DataOps {
 private:
  const GraphViewer& graph_viewer_;
  VersionNum version_id_;
  std::string device_id_;
  std::string device_precision_;
  std::multimap<std::string, UnsupportedOpMode> op_list_;
  std::vector<SupportedOp> subgraph_supported_;
  std::vector<SupportedOp> no_dimension_supported_;
  std::set<Pairs> supported_types_npu_;
  std::set<Pairs> supported_types_cpu_;
  std::set<Pairs> supported_types_gpu_;
  std::set<Pairs> supported_types_initializer_;
  bool npu_qdq_optimizer_enabled_;

 protected:
  void populate_op_mode_supported();
  void populate_types_supported();
  bool op_is_supported(std::string name, std::vector<SupportedOp>& list);
  bool dimension_unsupported(const Node* node);
  bool unsupported_op_mode(const Node* node, bool& has_external_weights_);
  bool type_is_supported(const NodeArg* node_arg, bool is_initializer);
  bool node_is_supported(const NodeIndex node_idx, bool& has_external_weights_);

 public:
  DataOps(const GraphViewer& graph_viewer_param, VersionNum ver,
          const std::string dev_id, const bool npu_qdq_optimizer_enabled)
      : graph_viewer_(graph_viewer_param),
        version_id_(ver),
        device_id_(std::move(dev_id)),
        npu_qdq_optimizer_enabled_(npu_qdq_optimizer_enabled) {
    populate_op_mode_supported();
    populate_types_supported();
  }

  virtual std::vector<NodeIndex> GetUnsupportedNodeIndices(
      std::unordered_set<std::string>& ng_required_initializers, bool& has_external_weights_);
  virtual bool IsOpSupportedOnlyInModel(std::string name);
  virtual bool SpecialConditionForClusterSizeOne(
      std::unordered_set<std::string>& ng_required_initializers, const Node* node);
  virtual bool DoNotOmitSubGraph(const std::string& name);
  virtual bool InsertNode(const std::string& name);
  VersionNum GetVersion() const { return version_id_; }
};

}  // namespace openvino_ep
}  // namespace onnxruntime
