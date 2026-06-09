// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/migraphx_call.h"
#include "core/providers/migraphx/migraphx_allocator.h"
#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/providers/migraphx/gpu_data_transfer.h"

namespace onnxruntime {

void MIGraphXAllocator::CheckDevice() const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call hipSetDevice instead of the check
  int current_device;
  auto hip_err = hipGetDevice(&current_device);
  if (hip_err == hipSuccess) {
    ORT_ENFORCE(current_device == Info().device.Id());
  }
#endif
}

void MIGraphXAllocator::EnablePoolMode() {
  std::lock_guard<std::mutex> lock(pool_mu_);
  pool_enabled_ = true;
}

void* MIGraphXAllocator::Alloc(size_t size) {
  CheckDevice();
  if (size == 0) return nullptr;

  if (pool_enabled_) {
    std::lock_guard<std::mutex> lock(pool_mu_);
    auto it = free_list_.find(size);
    if (it != free_list_.end() && !it->second.empty()) {
      void* p = it->second.back();
      it->second.pop_back();
      return p;
    }
  }

  void* p = nullptr;
  HIP_CALL_THROW(hipMalloc((void**)&p, size));

  if (pool_enabled_) {
    std::lock_guard<std::mutex> lock(pool_mu_);
    alloc_sizes_[p] = size;
  }

  return p;
}

void MIGraphXAllocator::Free(void* p) {
  CheckDevice();
  if (!p) return;

  if (pool_enabled_) {
    std::lock_guard<std::mutex> lock(pool_mu_);
    auto it = alloc_sizes_.find(p);
    if (it != alloc_sizes_.end()) {
      free_list_[it->second].push_back(p);
      return;
    }
  }

  (void)hipFree(p);
}

void* MIGraphXExternalAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = alloc_(size);

    // review(codemzs): ORT_ENFORCE does not seem appropriate.
    ORT_ENFORCE(p != nullptr);
  }

  return p;
}

void MIGraphXExternalAllocator::Free(void* p) {
  free_(p);
  std::lock_guard<std::mutex> lock(lock_);
  auto it = reserved_.find(p);
  if (it != reserved_.end()) {
    reserved_.erase(it);
    if (empty_cache_ != nullptr) {
      empty_cache_();
    }
  }
}

void* MIGraphXExternalAllocator::Reserve(size_t size) {
  void* p = Alloc(size);
  if (!p) return nullptr;
  std::lock_guard<std::mutex> lock(lock_);
  ORT_ENFORCE(reserved_.find(p) == reserved_.end());
  reserved_.insert(p);
  return p;
}

void* MIGraphXPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    HIP_CALL_THROW(hipHostMalloc((void**)&p, size));
  }
  return p;
}

void MIGraphXPinnedAllocator::Free(void* p) {
  HIP_CALL_THROW(hipHostFree(p));
}

}  // namespace onnxruntime
