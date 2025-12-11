// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <string>
#include <memory>

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "amdgpu_data_transfer.h"

/// <summary>
/// AMD GPU EP factory that can create an OrtEp and return information about the supported hardware devices.
/// </summary>
class AmdGpuEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  AmdGpuEpFactory(const char* ep_name, ApiPtrs apis, const OrtLogger& default_logger);

  OrtDataTransferImpl* GetDataTransfer() const {
    return data_transfer_impl_.get();
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              const OrtHardwareDevice* const* devices,
                                              const OrtKeyValuePairs* const* ep_metadata,
                                              size_t num_devices,
                                              const OrtSessionOptions* session_options,
                                              const OrtLogger* logger,
                                              OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* allocator_options,
                                                     OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                               const OrtMemoryDevice* memory_device,
                                                               const OrtKeyValuePairs* stream_options,
                                                               OrtSyncStreamImpl** stream) noexcept;

  const OrtLogger& default_logger_;        // default logger for the EP factory
  const std::string ep_name_;              // EP name
  const std::string vendor_{"AMD"};        // EP vendor name
  const uint32_t vendor_id_{0x1002};       // AMD PCI vendor ID
  const std::string ep_version_{"0.1.0"};  // EP version

  // CPU allocator for host memory
  Ort::MemoryInfo default_memory_info_;
  Ort::MemoryInfo readonly_memory_info_;  // used for initializers

  std::mutex mutex_;  // mutex to protect shared resources

  std::unique_ptr<AmdGpuDataTransfer> data_transfer_impl_;  // data transfer implementation for this factory
};
