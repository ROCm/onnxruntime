// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

// A wrapper class of hipEvent_t to destroy the event automatically for avoiding memory leak.
class AutoDestoryCudaEvent {
 public:
  AutoDestoryCudaEvent() : rocm_event_(nullptr) {
  }

  ~AutoDestoryCudaEvent() {
    if (rocm_event_ != nullptr)
      (void)hipEventDestroy(rocm_event_);
  }

  hipEvent_t& Get() {
    return rocm_event_;
  }

 private:
  hipEvent_t rocm_event_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
