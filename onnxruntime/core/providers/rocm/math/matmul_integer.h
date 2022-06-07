// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <hip/hip_runtime.h>


#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {
template <typename T1, typename T2>
class MatMulInteger final : public RocmKernel {
  using Base = RocmKernel;

 public:
  MatMulInteger(const OpKernelInfo& info) : RocmKernel(info) {
    has_a_zero_point_ = false;
    has_b_zero_point_ = false;
    if (info.GetInputCount() > 2) {
      has_a_zero_point_ = true;
    }
    if (info.GetInputCount() > 3) {
      has_b_zero_point_ = true;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool has_a_zero_point_;
  bool has_b_zero_point_;
};

}  // namespace rocm
}  // namespace onnxruntime
