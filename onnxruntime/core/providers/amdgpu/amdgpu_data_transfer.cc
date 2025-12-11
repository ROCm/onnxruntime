// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "amdgpu_data_transfer.h"
#include "amdgpu_factory.h"

#include <cassert>
#include <cstring>
#include <gsl/span>

AmdGpuDataTransfer::AmdGpuDataTransfer(AmdGpuEpFactory& factory)
    : OrtDataTransferImpl{},
      ApiPtrs(factory),
      factory_(factory) {
  ort_version_supported = ORT_API_VERSION;
  CanCopy = CanCopyImpl;
  CopyTensors = CopyTensorsImpl;
  Release = ReleaseImpl;
}

/*static*/
bool ORT_API_CALL AmdGpuDataTransfer::CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                                  const OrtMemoryDevice* src_memory_device,
                                                  const OrtMemoryDevice* dst_memory_device) noexcept {
  const auto& impl = *static_cast<const AmdGpuDataTransfer*>(this_ptr);

  // Skeleton implementation: Check if we can copy between devices
  OrtMemoryInfoDeviceType src_device_type = impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
  OrtMemoryInfoDeviceType dst_device_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);
  OrtDeviceMemoryType src_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(src_memory_device);
  OrtDeviceMemoryType dst_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(dst_memory_device);

  // For skeleton, we can copy to/from CPU or CPU accessible memory
  bool src_is_cpu = (src_device_type == OrtMemoryInfoDeviceType_CPU ||
                     src_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE);
  bool dst_is_cpu = (dst_device_type == OrtMemoryInfoDeviceType_CPU ||
                     dst_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE);

  return src_is_cpu || dst_is_cpu;
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuDataTransfer::CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                            const OrtValue** src_tensors_ptr,
                                                            OrtValue** dst_tensors_ptr,
                                                            OrtSyncStream** streams_ptr,
                                                            size_t num_tensors) noexcept {
  auto& impl = *static_cast<AmdGpuDataTransfer*>(this_ptr);

  auto src_tensors = gsl::make_span<const OrtValue*>(src_tensors_ptr, num_tensors);
  auto dst_tensors = gsl::make_span<OrtValue*>(dst_tensors_ptr, num_tensors);

  try {
    for (size_t i = 0; i < num_tensors; ++i) {
      const void* src_data = nullptr;
      void* dst_data = nullptr;
      size_t bytes;

      RETURN_IF_ERROR(impl.ort_api.GetTensorData(src_tensors[i], &src_data));
      RETURN_IF_ERROR(impl.ort_api.GetTensorMutableData(dst_tensors[i], &dst_data));
      RETURN_IF_ERROR(impl.ort_api.GetTensorSizeInBytes(src_tensors[i], &bytes));

      // Skeleton implementation: Use standard memcpy
      // In a real implementation, this would use hipMemcpy with appropriate flags
      // based on source and destination device types
      if (streams_ptr && streams_ptr[i]) {
        // Could do async copy with stream
      }

      std::memcpy(dst_data, src_data, bytes);
    }
  } catch (const std::exception& ex) {
    return impl.ort_api.CreateStatus(ORT_EP_FAIL, ex.what());
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL AmdGpuDataTransfer::ReleaseImpl(OrtDataTransferImpl* /*this_ptr*/) noexcept {
  // Data transfer is owned by the factory, so we don't delete it here
}
