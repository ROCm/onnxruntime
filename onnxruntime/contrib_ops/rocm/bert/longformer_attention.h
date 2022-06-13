// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "contrib_ops/cpu/bert/longformer_attention_base.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T>
class LongformerAttention final : public RocmKernel, public LongformerAttentionBase {
 public:
  LongformerAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool use_compact_memory_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
