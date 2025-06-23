// Copyright 2019 AMD AMDMIGraphX

#pragma once

#include <core/providers/migraphx/migraphx_execution_provider_info.h>

namespace onnxruntime {
class IAllocator;

struct ProviderInfo_MIGraphX {
  virtual std::unique_ptr<IAllocator> CreateMIGraphXAllocator(OrtDevice::DeviceId device_id, const char* name) = 0;
  virtual std::unique_ptr<IAllocator> CreateMIGraphXPinnedAllocator(OrtDevice::DeviceId device_id, const char* name) = 0;
  virtual void MIGraphXMemcpy_HostToDevice(void* dst, const void* src, size_t count) = 0;
  virtual void MIGraphXMemcpy_DeviceToHost(void* dst, const void* src, size_t count) = 0;
  virtual std::shared_ptr<IAllocator> CreateMIGraphXAllocator(OrtDevice::DeviceId device_id, size_t mem_limit,
    ArenaExtendStrategy arena_extend_strategy, void* alloc_fn, void* free_fn, void* empty_cache_fn, const OrtArenaCfg* default_memory_arena_cfg) = 0;

 protected:
  ~ProviderInfo_MIGraphX() = default;  // Can only be destroyed through a subclass instance
};

}  // namespace onnxruntime
