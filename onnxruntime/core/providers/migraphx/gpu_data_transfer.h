// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "core/providers/migraphx/migraphx_inc.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {

// Thread-safe pool of hipHostMalloc'd staging buffers used to avoid the
// silent synchronous fallback that hipMemcpyAsync performs when handed
// pageable (non-pinned) host memory.  Buffers are grown on demand and
// recycled between copies once the StagingReaper observes the associated
// HIP event has completed (so Release never runs on the compute stream).
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
  // Sized to comfortably cover the working set at large batch sizes
  // (~12 inputs * a few size classes) so eviction (and its hipHostFree
  // call) effectively never fires on the hot path.
  static constexpr size_t kMaxPoolSize = 32;
  std::mutex mu_;
  std::vector<Buffer> pool_;
};

// Reusable free-list of hipEvent_t handles.  Events are created with
// hipEventDisableTiming because we only need ordering semantics.
class HipEventPool {
 public:
  HipEventPool() = default;

  ~HipEventPool() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto e : free_) {
      (void)hipEventDestroy(e);
    }
  }

  HipEventPool(const HipEventPool&) = delete;
  HipEventPool& operator=(const HipEventPool&) = delete;

  hipEvent_t Acquire() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!free_.empty()) {
        hipEvent_t e = free_.back();
        free_.pop_back();
        return e;
      }
    }
    hipEvent_t e = nullptr;
    if (hipEventCreateWithFlags(&e, hipEventDisableTiming) != hipSuccess) {
      return nullptr;
    }
    return e;
  }

  void Release(hipEvent_t e) {
    if (!e) return;
    std::lock_guard<std::mutex> lock(mu_);
    free_.push_back(e);
  }

 private:
  std::mutex mu_;
  std::vector<hipEvent_t> free_;
};

// Background worker that polls HIP events and returns staging buffers to
// their pool once the recorded event reports complete.  This replaces the
// previous hipLaunchHostFunc-based release path, which serialised on the
// compute stream's host-function dispatcher and could deadlock when
// Release happened to invoke hipHostFree under load.
class StagingReaper {
  struct Item {
    hipEvent_t event;
    void* buffer;
    size_t capacity;
  };

 public:
  StagingReaper(PinnedStagingPool* pool, HipEventPool* events)
      : pool_(pool), events_(events), worker_([this] { Run(); }) {}

  ~StagingReaper() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    // Drain anything still in flight so the pool can free its buffers
    // safely in its own destructor.
    std::deque<Item> remaining;
    {
      std::lock_guard<std::mutex> lock(mu_);
      remaining.swap(queue_);
    }
    for (auto& it : remaining) {
      (void)hipEventSynchronize(it.event);
      pool_->Release(it.buffer, it.capacity);
      events_->Release(it.event);
    }
  }

  StagingReaper(const StagingReaper&) = delete;
  StagingReaper& operator=(const StagingReaper&) = delete;

  void Submit(hipEvent_t e, void* buffer, size_t capacity) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push_back({e, buffer, capacity});
    }
    cv_.notify_one();
  }

 private:
  void Run() {
    using namespace std::chrono_literals;
    std::unique_lock<std::mutex> lock(mu_);
    while (!stop_ || !queue_.empty()) {
      if (queue_.empty()) {
        cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        continue;
      }
      Item front = queue_.front();
      lock.unlock();

      hipError_t st = hipEventQuery(front.event);
      if (st == hipErrorNotReady) {
        // Not ready yet — sleep briefly to avoid burning a core.
        lock.lock();
        cv_.wait_for(lock, 100us);
        continue;
      }
      // hipSuccess (event complete) or any other error: release the buffer
      // either way.  If the event genuinely errored, the alternative would
      // be to spin forever; we'd rather leak any in-flight DMA's privacy
      // guarantees than hang the reaper.
      pool_->Release(front.buffer, front.capacity);
      events_->Release(front.event);
      lock.lock();
      if (!queue_.empty()) {
        queue_.pop_front();
      }
    }
  }

  PinnedStagingPool* pool_;
  HipEventPool* events_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Item> queue_;
  bool stop_ = false;
  // worker_ must be declared last so the thread sees fully-constructed
  // members when it starts running Run().
  std::thread worker_;
};

class GPUDataTransfer : public IDataTransfer {
 public:
  explicit GPUDataTransfer(hipStream_t stream = nullptr) : stream_(stream) {}
  ~GPUDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;
  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;
  common::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const override;

 private:
  static constexpr size_t kStagingThreshold = 64 * 1024;  // 64 KiB
  hipStream_t stream_;
  // staging_pool_ and event_pool_ MUST be declared before reaper_ because
  // the reaper holds raw pointers into them and starts its worker thread
  // in its constructor.
  mutable PinnedStagingPool staging_pool_;
  mutable HipEventPool event_pool_;
  mutable StagingReaper reaper_{&staging_pool_, &event_pool_};
};

}  // namespace onnxruntime
