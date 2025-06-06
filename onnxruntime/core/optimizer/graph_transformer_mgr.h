// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

// Manages a list of graph transformers. It is initialized with a list of graph
// transformers. Each inference session can further register additional ones.
class GraphTransformerManager {
 public:
  explicit GraphTransformerManager(unsigned steps) : steps_(steps) {
  }

  // Update (set) the maximum number of graph transformation steps
  common::Status SetSteps(unsigned steps);

  // Get the maximum number of graph transformation steps
  common::Status GetSteps(unsigned& steps) const;

  // Set the cancellation flag ptr from session_options
  void SetLoadCancellationFn(CheckLoadCancellationFn check_load_cancellation_fn) {
    check_load_cancellation_fn_ = std::move(check_load_cancellation_fn);
  }

  // Get the cancellation flag ptr
  bool IsLoadCancellationFlagSet() const noexcept {
    return check_load_cancellation_fn_ && check_load_cancellation_fn_();
  }

  // Register a transformer with a level.
  common::Status Register(std::unique_ptr<GraphTransformer> transformer, TransformerLevel level);

  // Apply all transformers registered for the given level on the given graph
  common::Status ApplyTransformers(Graph& graph, TransformerLevel level, const logging::Logger& logger) const;

  // Get if the graph is modified while applying the registered transformers
  const bool& IsGraphModified(void) const;
  // Set/Re-Set graph modified to "false" (generally) to remove any trace of previous application
  void ClearGraphModified(void);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformerManager);

  // maximum number of graph transformation steps
  unsigned steps_;

  InlinedHashMap<TransformerLevel, InlinedVector<std::unique_ptr<GraphTransformer>>> level_to_transformer_map_;
  InlinedHashMap<std::string, GraphTransformer*> transformers_info_;
  CheckLoadCancellationFn check_load_cancellation_fn_;
  mutable bool _is_graph_modified = false;
};
}  // namespace onnxruntime
