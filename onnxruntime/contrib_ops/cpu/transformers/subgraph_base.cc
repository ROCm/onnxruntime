// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <utility>
#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include <gsl/gsl>
#include "contrib_ops/cpu/transformers/subgraph_base.h"
#include "contrib_ops/cpu/utils/dump_tensor.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

Subgraph::Subgraph(
    const onnxruntime::Node& node_in,
    const std::string& attribute_name,
    const GraphViewer& subgraph_in)
    : node(node_in),
      attribute(attribute_name),
      subgraph(subgraph_in),
      num_heads(0),
      head_size(0),
      vocab_size(0),
      num_layers(0),
      past_present_share_buffer_(false),
      has_decoder_masked_attention_(false),
      allocator_(nullptr),
      is_output_float16_(false) {
  num_implicit_inputs = static_cast<int>(node.ImplicitInputDefs().size());
  used_implicit_inputs = std::vector<bool>(num_implicit_inputs, true);

  auto& subgraph_inputs = subgraph.GetInputs();
  auto& subgraph_outputs = subgraph.GetOutputs();

  num_subgraph_inputs = static_cast<int>(subgraph_inputs.size());
  num_subgraph_outputs = static_cast<int>(subgraph_outputs.size());

  subgraph_input_names.reserve(num_subgraph_inputs);
  for (int i = 0; i < num_subgraph_inputs; ++i) {
    subgraph_input_names.push_back(subgraph_inputs[i]->Name());
  }

  subgraph_output_names.reserve(num_subgraph_outputs);
  for (int i = 0; i < num_subgraph_outputs; ++i) {
    subgraph_output_names.push_back(subgraph_outputs[i]->Name());
  }

  for (const auto& n : subgraph.Nodes()) {
    if (n.OpType() == "DecoderMaskedSelfAttention" || n.OpType() == "DecoderMaskedMultiHeadAttention") {
      has_decoder_masked_attention_ = true;
      break;
    }
  }
}

Status Subgraph::Setup(const SessionState& session_state,
                       const SessionState& subgraph_session_state) {
  session_state_ = &session_state;
  subgraph_session_state_ = &subgraph_session_state;

  InlinedVector<std::string_view> feed_names;
  feed_names.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));

  // Use the first output to find device location.
  const OrtDevice& default_location = utils::FindDeviceForValue(subgraph_session_state, subgraph_output_names[0]);

  feed_names.insert(feed_names.end(), subgraph_input_names.begin(), subgraph_input_names.end());

  const auto& subgraph_map = subgraph_session_state.GetOrtValueNameIdxMap();

  const auto& implicit_input_defs = node.ImplicitInputDefs();
  for (size_t i = 0, end = num_implicit_inputs; i < end; ++i) {
    const auto* entry = implicit_input_defs[i];
    int idx;
    if (subgraph_map.GetIdx(entry->Name(), idx).IsOK()) {
      feed_names.push_back(entry->Name());
    } else {
      --num_implicit_inputs;
      used_implicit_inputs[i] = false;
    }
  }

  InlinedVector<OrtDevice> feed_locations;
  feed_locations.reserve(feed_names.size());

  for (size_t i = 0, end = feed_names.size(); i < end; ++i) {
    if (i >= subgraph_input_names.size()) {  // Implicit inputs
      const auto& location = utils::FindDeviceForValue(session_state, feed_names[i]);
      feed_locations.push_back(location);
    } else {
      if (feed_names[i] == "past_sequence_length") {
        // when past_sequence_length is needed in subgraph, treat it as past_present_share_buffer
        past_present_share_buffer_ = true;
        // past_sequence_length is on CPU memory
        feed_locations.push_back(OrtDevice());
      } else if (feed_names[i] == "beam_width") {
        // beam_width is on CPU memory
        feed_locations.push_back(OrtDevice());
      } else {
        feed_locations.push_back(default_location);
      }
    }
  }

  ORT_RETURN_IF_ERROR(FeedsFetchesManager::Create(feed_names, subgraph_output_names,
                                                  subgraph_session_state.GetOrtValueNameIdxMap(), feeds_fetches_manager_));
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(subgraph_session_state, *feeds_fetches_manager_));

  // Setup the locations where we want the subgraph output to end up on
  InlinedVector<const OrtDevice*> fetch_locations;
  fetch_locations.reserve(num_subgraph_outputs);

  // Past state need to be where we can feed them in to the next iteration, so set the location to match the feed.
  for (int i = 0; i < num_subgraph_outputs; ++i) {
    fetch_locations.push_back(&default_location);
  }

  utils::FinalizeFeedFetchCopyInfo(*feeds_fetches_manager_, feed_locations, fetch_locations);

  // Check subgraph only need once so put in Setup function.
  auto& inputs = subgraph.GetInputs();
  auto& outputs = subgraph.GetOutputs();
  ORT_RETURN_IF_ERROR(Validate(inputs, outputs));
  return Status::OK();
}

const IExecutionProvider* Subgraph::GetProvider() const {
  const ExecutionProviders& providers = session_state_->GetExecutionProviders();
  const IExecutionProvider* cpu_provider = providers.Get(onnxruntime::kCpuExecutionProvider);
  const IExecutionProvider* cuda_provider = providers.Get(onnxruntime::kCudaExecutionProvider);
  const IExecutionProvider* rocm_provider = providers.Get(onnxruntime::kRocmExecutionProvider);
  const IExecutionProvider* gpu_provider = cuda_provider ? cuda_provider : rocm_provider;
  const IExecutionProvider* provider = gpu_provider ? gpu_provider : cpu_provider;
  return provider;
}

Status Subgraph::GetParameters(const ONNX_NAMESPACE::TensorShapeProto* past_shape,
                               const ONNX_NAMESPACE::TensorShapeProto* logits_shape,
                               bool merged_past) {
  if (merged_past) {
    // Merged past state shape is like (2, batch_size, num_heads, past_seq_len, hidden_size/num_heads)
    ORT_RETURN_IF(past_shape->dim_size() != 5,
                  "subgraph past state is expected to have 5 dimension, got ", past_shape->dim_size());
    ORT_RETURN_IF(!past_shape->dim(0).has_dim_value() || past_shape->dim(0).dim_value() != 2,
                  "subgraph past state dimension 0 shall have length of 2");

    ORT_RETURN_IF(!past_shape->dim(2).has_dim_value() || past_shape->dim(2).dim_value() <= 0,
                  "subgraph past state dimension 2 shall have a positive value for number of heads");

    ORT_RETURN_IF(!past_shape->dim(4).has_dim_value() || past_shape->dim(4).dim_value() <= 0,
                  "subgraph past state dimension 4 shall have a positive value for hidden size per head");
    this->num_heads = static_cast<int>(past_shape->dim(2).dim_value());
    this->head_size = static_cast<int>(past_shape->dim(4).dim_value());
  } else {
    // Past state shape is like (batch_size, num_heads, past_seq_len, hidden_size/num_heads).
    ORT_RETURN_IF(past_shape->dim_size() != 4,
                  "subgraph output present_key_self_0 is expected to have 4 dimension, got ", past_shape->dim_size());

    ORT_RETURN_IF(!past_shape->dim(1).has_dim_value() || past_shape->dim(1).dim_value() <= 0,
                  "subgraph past state dimension 2 shall have a positive value for number of heads");

    ORT_RETURN_IF(!past_shape->dim(3).has_dim_value() || past_shape->dim(3).dim_value() <= 0,
                  "subgraph past state dimension 4 shall have a positive value for hidden size per head");
    this->num_heads = static_cast<int>(past_shape->dim(1).dim_value());
    this->head_size = static_cast<int>(past_shape->dim(3).dim_value());
  }

  // Logits shape is like (batch_size, seq_len, vocabulary_size)
  if (logits_shape != nullptr) {
    ORT_RETURN_IF(logits_shape->dim_size() != 3,
                  "subgraph logits output is expected to have 3 dimension, got ", logits_shape->dim_size());

    ORT_RETURN_IF(!logits_shape->dim(2).has_dim_value() || logits_shape->dim(2).dim_value() <= 0,
                  "subgraph past state dimension 2 shall have a positive value for vocabulary size");

    this->vocab_size = static_cast<int>(logits_shape->dim(2).dim_value());
  }

  return Status::OK();
}

Status Subgraph::AppendPastSequenceLength(std::vector<OrtValue>& feeds,
                                          AllocatorPtr cpu_allocator,
                                          const int32_t init_value) {
  int64_t past_seq_len_dims[] = {1};
  TensorShape past_seq_len_shape(&past_seq_len_dims[0], 1);
  OrtValue past_seq_len_tensor_value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), past_seq_len_shape, cpu_allocator, past_seq_len_tensor_value);
  feeds.push_back(past_seq_len_tensor_value);
  *past_seq_len_tensor_value.GetMutable<Tensor>()->MutableData<int32_t>() = init_value;

  return Status::OK();
}

Status Subgraph::AppendBeamWidthAndCacheIndir(std::vector<OrtValue>& feeds,
                                              AllocatorPtr cpu_allocator,
                                              AllocatorPtr default_allocator,
                                              const int64_t batch_size,
                                              const int64_t num_beams,
                                              const int64_t max_seq_len) {
  // Beam width feed
  int64_t num_beams_dims[] = {1};
  TensorShape num_beams_shape(&num_beams_dims[0], 1);
  OrtValue num_beams_tensor_value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), num_beams_shape, cpu_allocator, num_beams_tensor_value);
  feeds.push_back(num_beams_tensor_value);
  *num_beams_tensor_value.GetMutable<Tensor>()->MutableData<int32_t>() = static_cast<int32_t>(num_beams);

  // Cache indirection feed
  int64_t cache_indirection_dims[] = {batch_size, num_beams, max_seq_len};
  TensorShape cache_indirection_shape(&cache_indirection_dims[0], 3);
  OrtValue default_cache_indirection;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), cache_indirection_shape,
                       default_allocator, default_cache_indirection);
  feeds.push_back(default_cache_indirection);

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
