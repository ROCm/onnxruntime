// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "core/providers/migraphx/migraphx_inc.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {

// Thread-safe pool of hipHostMalloc'd staging buffers used to avoid the
// silent synchronous fallback that hipMemcpyAsync performs when handed
// pageable (non-pinned) host memory.  Buffers are grown on demand and
// recycled between copies via hipLaunchHostFunc callbacks.
class PinnedStagingPool {
  struct Buffer {
    void* ptr;
    size_t capacity;
  };

 public:
  PinnedStagingPool() = default;

  ~PinnedStagingPool() {
    (void)hipDeviceSynchronize();
    for (auto& b : pool_) {
      (void)hipHostFree(b.ptr);
    }
  }

  PinnedStagingPool(const PinnedStagingPool&) = delete;
  PinnedStagingPool& operator=(const PinnedStagingPool&) = delete;

  // Returns a pinned buffer with at least `bytes` capacity, or nullptr on
  // allocation failure.  Prefers the smallest adequate buffer already in
  // the pool to minimise waste.
  void* Acquire(size_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    auto best = pool_.end();
    for (auto it = pool_.begin(); it != pool_.end(); ++it) {
      if (it->capacity >= bytes &&
          (best == pool_.end() || it->capacity < best->capacity)) {
        best = it;
      }
    }
    if (best != pool_.end()) {
      void* p = best->ptr;
      pool_.erase(best);
      return p;
    }
    void* p = nullptr;
    if (hipHostMalloc(&p, bytes) != hipSuccess) return nullptr;
    return p;
  }

  void Release(void* ptr, size_t capacity) {
    std::lock_guard<std::mutex> lock(mu_);
    if (pool_.size() >= kMaxPoolSize) {
      auto smallest = std::min_element(
          pool_.begin(), pool_.end(),
          [](const Buffer& a, const Buffer& b) { return a.capacity < b.capacity; });
      if (smallest != pool_.end() && smallest->capacity < capacity) {
        (void)hipHostFree(smallest->ptr);
        pool_.erase(smallest);
      } else {
        (void)hipHostFree(ptr);
        return;
      }
    }
    pool_.push_back({ptr, capacity});
  }

 private:
  static constexpr size_t kMaxPoolSize = 8;
  std::mutex mu_;
  std::vector<Buffer> pool_;
};

class GPUDataTransfer : public IDataTransfer {
 public:
  explicit GPUDataTransfer(hipStream_t stream = nullptr) : stream_(stream) {}
  ~GPUDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;
  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;
  common::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const override;

 private:
  static constexpr size_t kStagingThreshold = SIZE_MAX; //64 * 1024;  // 64 KiB
  hipStream_t stream_;
  mutable PinnedStagingPool staging_pool_;
};

}  // namespace onnxruntime
