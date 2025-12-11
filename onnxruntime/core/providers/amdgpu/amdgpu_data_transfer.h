// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "amdgpu_provider_utils.h"

class AmdGpuEpFactory;

/// <summary>
/// Data transfer implementation for AMD GPU EP.
/// Handles copying data between CPU and GPU memory.
/// </summary>
class AmdGpuDataTransfer : public OrtDataTransferImpl, public ApiPtrs {
 public:
  explicit AmdGpuDataTransfer(AmdGpuEpFactory& factory);

 private:
  static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                       const OrtMemoryDevice* src_memory_device,
                                       const OrtMemoryDevice* dst_memory_device) noexcept;

  static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                 const OrtValue** src_tensors,
                                                 OrtValue** dst_tensors,
                                                 OrtSyncStream** streams,
                                                 size_t num_tensors) noexcept;

  static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

  AmdGpuEpFactory& factory_;
  const OrtMemoryDevice* device_mem_info{nullptr};
};
