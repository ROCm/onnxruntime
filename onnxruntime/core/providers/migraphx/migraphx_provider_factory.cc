// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
#include <atomic>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <shlwapi.h>
#endif

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/migraphx_provider_factory.h"
#include "core/providers/migraphx/migraphx_execution_provider.h"
#include "core/providers/migraphx/migraphx_execution_provider_info.h"
#include "core/providers/migraphx/migraphx_provider_factory_creator.h"
#include "core/providers/migraphx/migraphx_allocator.h"
#include "core/providers/migraphx/gpu_data_transfer.h"
#include "core/framework/provider_options.h"

#include "core/session/onnxruntime_c_api.h"
#include "core/providers/migraphx/migraphx_call.h"

using namespace onnxruntime;

namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct MIGraphXProviderFactory final : IExecutionProviderFactory {
  explicit MIGraphXProviderFactory(MIGraphXExecutionProviderInfo info) : info_{std::move(info)} {}
  ~MIGraphXProviderFactory() override = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  MIGraphXExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> MIGraphXProviderFactory::CreateProvider() {
  return std::make_unique<MIGraphXExecutionProvider>(info_);
}

struct ProviderInfo_MIGraphX_Impl final : ProviderInfo_MIGraphX {
  std::unique_ptr<IAllocator> CreateMIGraphXAllocator(OrtDevice::DeviceId device_id, const char* name) override {
    return std::make_unique<MIGraphXAllocator>(device_id, name);
  }

  std::unique_ptr<IAllocator> CreateMIGraphXPinnedAllocator(OrtDevice::DeviceId device_id, const char* name) override {
    return std::make_unique<MIGraphXPinnedAllocator>(device_id, name);
  }

  void MIGraphXMemcpy_HostToDevice(void* dst, const void* src, size_t count) override {
    // hipMemcpy() operates on the default stream
    HIP_CALL_THROW(hipMemcpy(dst, src, count, hipMemcpyHostToDevice));

    // To ensure that the copy has completed, invoke a stream sync for the default stream.
    // For transfers from pageable host memory to device memory, a stream sync is performed before the copy is initiated.
    // The function will return once the pageable buffer has been copied to the staging memory for DMA transfer
    // to device memory, but the DMA to final destination may not have completed.

    HIP_CALL_THROW(hipStreamSynchronize(0));
  }

  // Used by onnxruntime_pybind_state.cc
  void MIGraphXMemcpy_DeviceToHost(void* dst, const void* src, size_t count) override {
    // For transfers from device to either pageable or pinned host memory, the function returns only once the copy has completed.
    HIP_CALL_THROW(hipMemcpy(dst, src, count, hipMemcpyDeviceToHost));
  }

  std::shared_ptr<IAllocator> CreateMIGraphXAllocator(OrtDevice::DeviceId device_id, size_t migx_mem_limit, ArenaExtendStrategy arena_extend_strategy, MIGraphXExecutionProviderExternalAllocatorInfo external_allocator_info, const OrtArenaCfg* default_memory_arena_cfg) override {
    return MIGraphXExecutionProvider::CreateMIGraphXAllocator(device_id, migx_mem_limit, arena_extend_strategy, external_allocator_info, default_memory_arena_cfg);
  }
} g_info;

struct MIGraphX_Provider final : Provider {
  virtual ~MIGraphX_Provider() = default;
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    MIGraphXExecutionProviderInfo info;
    info.device_id = static_cast<OrtDevice::DeviceId>(device_id);
    info.target_device = "gpu";
    return std::make_shared<MIGraphXProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    auto& options = *static_cast<const OrtMIGraphXProviderOptions*>(provider_options);
    MIGraphXExecutionProviderInfo info;
    info.device_id = static_cast<OrtDevice::DeviceId>(options.device_id);
    info.target_device = "gpu";
    info.fp16_enable = options.migraphx_fp16_enable;
    info.fp8_enable = options.migraphx_fp8_enable;
    info.exhaustive_tune = options.migraphx_exhaustive_tune;
    info.int8_enable = options.migraphx_int8_enable;
    info.int8_calibration_table_name = "";
    if (options.migraphx_int8_calibration_table_name != nullptr) {
      info.int8_calibration_table_name = options.migraphx_int8_calibration_table_name;
    }
    info.int8_use_native_calibration_table = options.migraphx_use_native_calibration_table != 0;
    info.model_cache_dir = "";
    if (options.migraphx_cache_dir != nullptr) {
      info.model_cache_dir = options.migraphx_cache_dir;
    }
    info.arena_extend_strategy = static_cast<onnxruntime::ArenaExtendStrategy>(options.migraphx_arena_extend_strategy);
    info.mem_limit = options.migraphx_mem_limit;
    return std::make_shared<MIGraphXProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto internal_options = MIGraphXExecutionProviderInfo::FromProviderOptions(options);
    auto& migx_options = *static_cast<OrtMIGraphXProviderOptions*>(provider_options);
    migx_options.device_id = internal_options.device_id;
    migx_options.migraphx_fp16_enable = internal_options.fp16_enable;
    migx_options.migraphx_fp8_enable = internal_options.fp8_enable;
    migx_options.migraphx_int8_enable = internal_options.int8_enable;
    migx_options.migraphx_exhaustive_tune = internal_options.exhaustive_tune;

    char* dest = nullptr;
    auto str_size = internal_options.int8_calibration_table_name.size();
    if (str_size == 0) {
      migx_options.migraphx_int8_calibration_table_name = nullptr;
    } else {
      dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.int8_calibration_table_name.c_str(), str_size);
#else
      strncpy(dest, internal_options.int8_calibration_table_name.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      migx_options.migraphx_int8_calibration_table_name = dest;
    }

    migx_options.migraphx_use_native_calibration_table = internal_options.int8_use_native_calibration_table;
    migx_options.migraphx_cache_dir = internal_options.model_cache_dir.string().c_str();
    migx_options.migraphx_arena_extend_strategy = static_cast<int>(internal_options.arena_extend_strategy);
    migx_options.migraphx_mem_limit = internal_options.mem_limit;
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *static_cast<const OrtMIGraphXProviderOptions*>(provider_options);
    return MIGraphXExecutionProviderInfo::ToProviderOptions(options);
  }

  void Initialize() override {
#ifdef _WIN32
    HMODULE module = nullptr;
    if(GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          static_cast<LPCSTR>(static_cast<void*>(InitializeRegistry)),
                          &module) != 0) {
      char buffer[MAX_PATH];
      if(GetModuleFileName(module, buffer, sizeof(buffer)) != 0) {
        PathRemoveFileSpec(buffer);
        SetDllDirectory(buffer);
      }
    }
#endif
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &g_provider;
}
}
