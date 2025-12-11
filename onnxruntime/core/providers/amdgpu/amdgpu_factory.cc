// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "amdgpu_factory.h"
#include "amdgpu_execution_provider.h"
#include "amdgpu_stream_support.h"

#include <sstream>

AmdGpuEpFactory::AmdGpuEpFactory(const char* ep_name, ApiPtrs apis, const OrtLogger& default_logger)
    : OrtEpFactory{},
      ApiPtrs(apis),
      default_logger_(default_logger),
      ep_name_(ep_name),
      default_memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
      readonly_memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUInput)),
      data_transfer_impl_(std::make_unique<AmdGpuDataTransfer>(*this)) {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
}

/*static*/
const char* ORT_API_CALL AmdGpuEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const AmdGpuEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL AmdGpuEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const AmdGpuEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

/*static*/
uint32_t ORT_API_CALL AmdGpuEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const AmdGpuEpFactory*>(this_ptr);
  return factory->vendor_id_;
}

/*static*/
const char* ORT_API_CALL AmdGpuEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const AmdGpuEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                 const OrtHardwareDevice* const* devices,
                                                                 size_t num_devices,
                                                                 OrtEpDevice** ep_devices,
                                                                 size_t max_ep_devices,
                                                                 size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  auto* factory = static_cast<AmdGpuEpFactory*>(this_ptr);

  try {
    // Skeleton implementation: Look for AMD GPU devices
    // In a real implementation, this would query ROCm/HIP for available AMD GPUs
    for (size_t i = 0; i < num_devices; ++i) {
      Ort::ConstHardwareDevice device{devices[i]};

      // Check if this is an AMD GPU device
      if (device.VendorId() == factory->vendor_id_ && device.Type() == OrtHardwareDeviceType_GPU) {
        if (num_ep_devices >= max_ep_devices) {
          return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Not enough space to return all EP devices.");
        }

        // Create an EP device for this hardware device
        OrtEpDevice* ep_device = nullptr;
        OrtKeyValuePairs* ep_metadata = nullptr;
        OrtKeyValuePairs* ep_options = nullptr;

        RETURN_IF_ERROR(factory->ep_api.CreateEpDevice(factory, devices[i], ep_metadata, ep_options, &ep_device));

        // Add memory info for the device
        Ort::MemoryInfo gpu_memory_info(
            factory->ep_name_.c_str(), OrtDeviceAllocator, static_cast<int>(i), OrtMemTypeDefault);
        RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, gpu_memory_info));

        ep_devices[num_ep_devices++] = ep_device;
      }
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return factory->ort_api.CreateStatus(ORT_EP_FAIL, ex.what());
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                      const OrtHardwareDevice* const* /*devices*/,
                                                      const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                      size_t num_devices,
                                                      const OrtSessionOptions* session_options,
                                                      const OrtLogger* logger,
                                                      OrtEp** ep) noexcept {
  auto* factory = static_cast<AmdGpuEpFactory*>(this_ptr);
  *ep = nullptr;

  try {
    // Get EP configuration from session options
    std::string ep_context_enable;
    RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, "ep.context.enable", "0", ep_context_enable));

    AmdGpuEp::Config config = {};
    config.enable_ep_context = ep_context_enable == "1";

    auto amd_gpu_ep = std::make_unique<AmdGpuEp>(*factory, factory->ep_name_, config, *logger);

    *ep = amd_gpu_ep.release();
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return factory->ort_api.CreateStatus(ORT_EP_FAIL, ex.what());
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL AmdGpuEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  AmdGpuEp* amd_gpu_ep = static_cast<AmdGpuEp*>(ep);
  delete amd_gpu_ep;
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuEpFactory::CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                             const OrtMemoryInfo* memory_info,
                                                             const OrtKeyValuePairs* /*allocator_options*/,
                                                             OrtAllocator** allocator) noexcept {
  auto& factory = *static_cast<AmdGpuEpFactory*>(this_ptr);
  *allocator = nullptr;

  try {
    // Skeleton implementation: Return CPU allocator for now
    // In a real implementation, this would create a ROCm/HIP device allocator
    Ort::ConstMemoryInfo mem_info{memory_info};

    if (mem_info.GetDeviceType() == OrtMemoryInfoDeviceType_CPU) {
      // Skeleton: Use ORT's default CPU allocator
      RETURN_IF_ERROR(factory.ort_api.GetAllocatorWithDefaultOptions(allocator));
    } else {
      return factory.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED, "AMD GPU device allocator not yet implemented in skeleton.");
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return factory.ort_api.CreateStatus(ORT_EP_FAIL, ex.what());
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL AmdGpuEpFactory::ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept {
  auto& factory = *static_cast<AmdGpuEpFactory*>(this_ptr);
  std::lock_guard<std::mutex> lock{factory.mutex_};

  // Release the allocator
  factory.ort_api.ReleaseAllocator(allocator);
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuEpFactory::CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                                OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<AmdGpuEpFactory*>(this_ptr);
  *data_transfer = factory.data_transfer_impl_.get();
  return nullptr;
}

/*static*/
bool ORT_API_CALL AmdGpuEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return true;  // AMD GPU EP implements stream synchronization
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuEpFactory::CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                                       const OrtMemoryDevice* memory_device,
                                                                       const OrtKeyValuePairs* stream_options,
                                                                       OrtSyncStreamImpl** stream) noexcept {
  auto& factory = *static_cast<AmdGpuEpFactory*>(this_ptr);
  *stream = nullptr;

  try {
    auto stream_impl = std::make_unique<AmdGpuStreamImpl>(factory, nullptr, stream_options);
    *stream = stream_impl.release();
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return factory.ort_api.CreateStatus(ORT_EP_FAIL, ex.what());
  }

  return nullptr;
}
