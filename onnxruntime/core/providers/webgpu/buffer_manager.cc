// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

namespace {

constexpr size_t NormalizeBufferSize(size_t size) {
  return (size + 15) / 16 * 16;
}

void EnforceBufferUnmapped(WebGpuContext& context, WGPUBuffer buffer) {
  if (context.ValidationMode() > ValidationMode::Basic) {
    ORT_ENFORCE(wgpuBufferGetMapState(buffer) == WGPUBufferMapState_Unmapped, "Buffer is still mapped.");
  }
}

}  // namespace

class DisabledCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t /*buffer_size*/) override {
    // always return empty buffer
    return nullptr;
  }
  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }
  void ReleaseBuffer(WGPUBuffer buffer) override {
    wgpuBufferRelease(buffer);
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    // no-op
  }
};

class LazyReleaseCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t /*buffer_size*/) override {
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    Release();
    pending_buffers_.clear();
  }

 public:
  ~LazyReleaseCacheManager() {
    Release();
  }

 protected:
  void Release() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
  }

  std::vector<WGPUBuffer> pending_buffers_;
};

class SimpleCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buffers_.find(buffer_size);
    if (it != buffers_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }

    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    for (auto& buffer : pending_buffers_) {
      buffers_[static_cast<size_t>(wgpuBufferGetSize(buffer))].emplace_back(buffer);
    }
    pending_buffers_.clear();
  }

 public:
  ~SimpleCacheManager() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
    for (auto& pair : buffers_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
  }

 protected:
  std::map<size_t, std::vector<WGPUBuffer>> buffers_;
  std::vector<WGPUBuffer> pending_buffers_;
};

// TODO: maybe use different bucket size for storage and uniform buffers?
constexpr std::initializer_list<std::pair<const size_t, size_t>> BUCKET_DEFAULT_LIMIT_TABLE = {
    {64, 250},
    {128, 200},
    {256, 200},
    {512, 200},
    {2048, 230},
    {4096, 200},
    {8192, 50},
    {16384, 50},
    {32768, 50},
    {65536, 50},
    {131072, 50},
    {262144, 50},
    {524288, 50},
    {1048576, 50},
    {2097152, 30},
    {4194304, 20},
    {8388608, 10},
    {12582912, 10},
    {16777216, 10},
    {26214400, 15},
    {33554432, 22},
    {44236800, 2},
    {58982400, 6},
    // we don't want to cache the bucket sizes below but not caching them
    // results in some major performance hits for models like sd-turbo.
    {67108864, 6},
    {134217728, 6},
    {167772160, 6},
};

class BucketCacheManager : public IBufferCacheManager {
 public:
  BucketCacheManager() : buckets_limit_{BUCKET_DEFAULT_LIMIT_TABLE} {
    Initialize();
  }
  BucketCacheManager(std::unordered_map<size_t, size_t>&& buckets_limit) : buckets_limit_{buckets_limit} {
    Initialize();
  }

  size_t CalculateBufferSize(size_t request_size) override {
    // binary serch size
    auto it = std::lower_bound(buckets_keys_.begin(), buckets_keys_.end(), request_size);
    if (it == buckets_keys_.end()) {
      return NormalizeBufferSize(request_size);
    } else {
      return *it;
    }
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    auto buffer_size = static_cast<size_t>(wgpuBufferGetSize(buffer));

    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && it->second.size() < buckets_limit_[buffer_size]) {
      it->second.emplace_back(buffer);
    } else {
      pending_buffers_.emplace_back(buffer, buffer_size);
    }
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    for (auto& [buffer, buffer_size] : pending_buffers_) {
      auto it = buckets_.find(buffer_size);
      if (it != buckets_.end() && it->second.size() < buckets_limit_[buffer_size]) {
        it->second.emplace_back(buffer);
      } else {
        wgpuBufferRelease(buffer);
      }
    }
    pending_buffers_.clear();
  }

  ~BucketCacheManager() {
    for (auto& pair : buckets_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
  }

 protected:
  void Initialize() {
    buckets_keys_.reserve(buckets_limit_.size());
    buckets_.reserve(buckets_limit_.size());
    for (const auto& pair : buckets_limit_) {
      buckets_keys_.push_back(pair.first);
      buckets_.emplace(pair.first, std::vector<WGPUBuffer>());
    }
    std::sort(buckets_keys_.begin(), buckets_keys_.end());

#ifndef NDEBUG  // if debug build
    ORT_ENFORCE(std::all_of(buckets_keys_.begin(), buckets_keys_.end(), [](size_t size) { return size % 16 == 0; }),
                "Bucket sizes must be multiples of 16.");

    for (size_t i = 1; i < buckets_keys_.size(); ++i) {
      ORT_ENFORCE(buckets_keys_[i] > buckets_keys_[i - 1], "Bucket sizes must be in increasing order.");
    }
#endif
  }
  std::unordered_map<size_t, size_t> buckets_limit_;
  std::unordered_map<size_t, std::vector<WGPUBuffer>> buckets_;
  std::vector<std::pair<WGPUBuffer, size_t>> pending_buffers_;
  std::vector<size_t> buckets_keys_;
};

class GraphCacheManager : public IBufferCacheManager {
 public:
  GraphCacheManager() : buckets_limit_{BUCKET_DEFAULT_LIMIT_TABLE} {
    Initialize();
  }
  GraphCacheManager(std::unordered_map<size_t, size_t>&& buckets_limit) : buckets_limit_{buckets_limit} {
    Initialize();
  }

  size_t CalculateBufferSize(size_t request_size) override {
    // binary serch size
    auto it = std::lower_bound(buckets_keys_.begin(), buckets_keys_.end(), request_size);
    if (it == buckets_keys_.end()) {
      return NormalizeBufferSize(request_size);
    } else {
      return *it;
    }
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    // Initialize buckets if they don't exist yet
    if (buckets_.empty()) {
      for (const auto& pair : buckets_limit_) {
        buckets_.emplace(pair.first, std::vector<WGPUBuffer>());
      }
    }

    for (auto& buffer : pending_buffers_) {
      auto buffer_size = static_cast<size_t>(wgpuBufferGetSize(buffer));
      auto it = buckets_.find(buffer_size);
      if (it != buckets_.end()) {
        it->second.emplace_back(buffer);
      } else {
        // insert a new bucket if it doesn't exist
        buckets_[buffer_size] = std::vector<WGPUBuffer>{buffer};
      }
    }

    pending_buffers_.clear();
  }

  ~GraphCacheManager() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
    for (auto& pair : buckets_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
  }

 protected:
  void Initialize() {
    buckets_keys_.reserve(buckets_limit_.size());
    for (const auto& pair : buckets_limit_) {
      buckets_keys_.push_back(pair.first);
    }
    std::sort(buckets_keys_.begin(), buckets_keys_.end());

#ifndef NDEBUG  // if debug build
    ORT_ENFORCE(std::all_of(buckets_keys_.begin(), buckets_keys_.end(), [](size_t size) { return size % 16 == 0; }),
                "Bucket sizes must be multiples of 16.");

    for (size_t i = 1; i < buckets_keys_.size(); ++i) {
      ORT_ENFORCE(buckets_keys_[i] > buckets_keys_[i - 1], "Bucket sizes must be in increasing order.");
    }
#endif
  }
  std::unordered_map<size_t, size_t> buckets_limit_;
  std::unordered_map<size_t, std::vector<WGPUBuffer>> buckets_;
  std::vector<WGPUBuffer> pending_buffers_;
  std::vector<size_t> buckets_keys_;
};

class GraphSimpleCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buffers_.find(buffer_size);
    if (it != buffers_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }

    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh(GraphCaptureState graph_capture_state) override {
    for (auto& buffer : pending_buffers_) {
      if (graph_capture_state == GraphCaptureState::Default) {
        buffers_[static_cast<size_t>(wgpuBufferGetSize(buffer))].emplace_back(buffer);
      } else {
        captured_buffers_.emplace_back(buffer);
      }
    }
    pending_buffers_.clear();
  }

 public:
  ~GraphSimpleCacheManager() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
    for (auto& pair : buffers_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
    for (auto& buffer : captured_buffers_) {
      wgpuBufferRelease(buffer);
    }
  }

 protected:
  std::map<size_t, std::vector<WGPUBuffer>> buffers_;
  std::vector<WGPUBuffer> pending_buffers_;
  std::vector<WGPUBuffer> captured_buffers_;
};

std::unique_ptr<IBufferCacheManager> CreateBufferCacheManager(BufferCacheMode cache_mode) {
  switch (cache_mode) {
    case BufferCacheMode::Disabled:
      return std::make_unique<DisabledCacheManager>();
    case BufferCacheMode::LazyRelease:
      return std::make_unique<LazyReleaseCacheManager>();
    case BufferCacheMode::Simple:
      return std::make_unique<SimpleCacheManager>();
    case BufferCacheMode::Bucket:
      return std::make_unique<BucketCacheManager>();
    case BufferCacheMode::Graph:
      return std::make_unique<GraphCacheManager>();
    case BufferCacheMode::GraphSimple:
      return std::make_unique<GraphSimpleCacheManager>();
    default:
      ORT_NOT_IMPLEMENTED("Unsupported buffer cache mode");
  }
}

std::ostream& operator<<(std::ostream& os, BufferCacheMode mode) {
  switch (mode) {
    case BufferCacheMode::Disabled:
      os << "Disabled";
      break;
    case BufferCacheMode::LazyRelease:
      os << "LazyRelease";
      break;
    case BufferCacheMode::Simple:
      os << "Simple";
      break;
    case BufferCacheMode::Bucket:
      os << "Bucket";
      break;
    case BufferCacheMode::Graph:
      os << "Graph";
      break;
    case BufferCacheMode::GraphSimple:
      os << "GraphSimple";
      break;
    default:
      os << "Unknown(" << static_cast<int>(mode) << ")";
  }
  return os;
}

BufferManager::BufferManager(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode)
    : context_{context},
      storage_cache_{CreateBufferCacheManager(storage_buffer_cache_mode)},
      uniform_cache_{CreateBufferCacheManager(uniform_buffer_cache_mode)},
      query_resolve_cache_{CreateBufferCacheManager(query_resolve_buffer_cache_mode)},
      default_cache_{CreateBufferCacheManager(BufferCacheMode::Disabled)} {
}

void BufferManager::Upload(void* src, WGPUBuffer dst, size_t size) const {
  // If the buffer is mapped, we can directly write to it.
  void* mapped_data = wgpuBufferGetMappedRange(dst, 0, WGPU_WHOLE_MAP_SIZE);  // ensure the buffer is mapped
  if (mapped_data) {
    memcpy(mapped_data, src, size);
    wgpuBufferUnmap(dst);
    return;
  }

  // Otherwise, we need to use a staging buffer to upload data.
  auto buffer_size = NormalizeBufferSize(size);

  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite;
  desc.mappedAtCreation = true;

  auto staging_buffer = context_.Device().CreateBuffer(&desc);
  mapped_data = staging_buffer.GetMappedRange();
  memcpy(mapped_data, src, size);
  staging_buffer.Unmap();

  auto& command_encoder = context_.GetCommandEncoder();
  context_.EndComputePass();
  command_encoder.CopyBufferToBuffer(staging_buffer, 0, dst, 0, buffer_size);
  context_.Flush(*this);
}

void BufferManager::MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size) const {
  ORT_ENFORCE(src != dst, "Source and destination buffers must be different.");
  EnforceBufferUnmapped(context_, src);
  EnforceBufferUnmapped(context_, dst);

  auto buffer_size = NormalizeBufferSize(size);
  auto src_size = static_cast<size_t>(wgpuBufferGetSize(src));
  auto dst_size = static_cast<size_t>(wgpuBufferGetSize(dst));
  ORT_ENFORCE(buffer_size <= src_size && buffer_size <= dst_size,
              "Source and destination buffers must have enough space for the copy operation. src_size=",
              src_size, ", dst_size=", dst_size, ", copy_size=", buffer_size, ".");

  auto& command_encoder = context_.GetCommandEncoder();
  context_.EndComputePass();
  command_encoder.CopyBufferToBuffer(src, 0, dst, 0, buffer_size);
}

WGPUBuffer BufferManager::Create(size_t size, wgpu::BufferUsage usage) const {
  auto& cache = GetCacheManager(usage);
  auto buffer_size = cache.CalculateBufferSize(size);

  auto buffer = cache.TryAcquireCachedBuffer(buffer_size);
  if (buffer) {
    return buffer;
  }

  // cache miss, create a new buffer
  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = usage;
  buffer = context_.Device().CreateBuffer(&desc).MoveToCHandle();

  ORT_ENFORCE(buffer, "Failed to create GPU buffer: size=", buffer_size, ", usage=", uint64_t(usage), ".");

  cache.RegisterBuffer(buffer, size);
  return buffer;
}

WGPUBuffer BufferManager::CreateUMA(size_t size, wgpu::BufferUsage usage) const {
  ORT_ENFORCE(usage & wgpu::BufferUsage::Storage, "UMA buffer must be a storage buffer.");
  auto& cache = GetCacheManager(usage);
  auto buffer_size = cache.CalculateBufferSize(size);

  // Ensure the buffer is mapped for writing at creation.
  usage |= wgpu::BufferUsage::MapWrite;

  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = usage;
  desc.mappedAtCreation = true;
  auto buffer = context_.Device().CreateBuffer(&desc).MoveToCHandle();

  ORT_ENFORCE(buffer, "Failed to create GPU buffer: size=", buffer_size, ", usage=", uint64_t(usage), ".");

  cache.RegisterBuffer(buffer, size);
  return buffer;
}

bool BufferManager::SupportsUMA() const {
#if !defined(__wasm__)
  // Check if the device supports the BufferMapExtendedUsages feature
  return context_.DeviceHasFeature(wgpu::FeatureName::BufferMapExtendedUsages);
#else
  return false;
#endif  // !defined(__wasm__)
}

void BufferManager::Release(WGPUBuffer buffer) const {
  EnforceBufferUnmapped(context_, buffer);
  GetCacheManager(buffer).ReleaseBuffer(buffer);
}

void BufferManager::Download(WGPUBuffer src, void* dst, size_t size) const {
  EnforceBufferUnmapped(context_, src);
  auto buffer_size = NormalizeBufferSize(size);

  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;

  auto staging_buffer = context_.Device().CreateBuffer(&desc);
  auto& command_encoder = context_.GetCommandEncoder();
  context_.EndComputePass();
  command_encoder.CopyBufferToBuffer(src, 0, staging_buffer, 0, buffer_size);
  context_.Flush(*this);

  // TODO: revise wait in whole project

  ORT_ENFORCE(context_.Wait(staging_buffer.MapAsync(wgpu::MapMode::Read, 0, buffer_size, wgpu::CallbackMode::WaitAnyOnly, [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
    ORT_ENFORCE(status == wgpu::MapAsyncStatus::Success, "Failed to download data from buffer: ", std::string_view{message});
  })) == Status::OK());

  auto mapped_data = staging_buffer.GetConstMappedRange();
  memcpy(dst, mapped_data, size);
  staging_buffer.Unmap();
}

void BufferManager::RefreshPendingBuffers(GraphCaptureState graph_capture_state) const {
  storage_cache_->OnRefresh(graph_capture_state);
  uniform_cache_->OnRefresh(graph_capture_state);
  query_resolve_cache_->OnRefresh(graph_capture_state);
  default_cache_->OnRefresh(graph_capture_state);
}

IBufferCacheManager& BufferManager::GetCacheManager(wgpu::BufferUsage usage) const {
  if (usage & wgpu::BufferUsage::Storage) {
    return *storage_cache_;
  } else if (usage & wgpu::BufferUsage::Uniform) {
    return *uniform_cache_;
  } else if (usage & wgpu::BufferUsage::QueryResolve) {
    return *query_resolve_cache_;
  } else {
    return *default_cache_;
  }
}

IBufferCacheManager& BufferManager::GetCacheManager(WGPUBuffer buffer) const {
  auto usage = static_cast<wgpu::BufferUsage>(wgpuBufferGetUsage(buffer));
  return GetCacheManager(usage);
}

std::unique_ptr<BufferManager> BufferManagerFactory::Create(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode) {
  return std::make_unique<BufferManager>(context, storage_buffer_cache_mode, uniform_buffer_cache_mode, query_resolve_buffer_cache_mode);
}

}  // namespace webgpu
}  // namespace onnxruntime
