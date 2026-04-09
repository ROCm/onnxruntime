// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/gpu_data_transfer.h"
#include "core/providers/migraphx/migraphx_call.h"

// If you make change below, please also update onnxruntime/core/providers/rocm/gpu_data_transfer.cc

namespace onnxruntime {

namespace {

struct StagingReturnInfo {
  PinnedStagingPool* pool;
  void* buffer;
  size_t capacity;
};

void HIPAPI StagingReturnCallback(void* raw) {
  std::unique_ptr<StagingReturnInfo> info(static_cast<StagingReturnInfo*>(raw));
  info->pool->Release(info->buffer, info->capacity);
}

}  // namespace

GPUDataTransfer::~GPUDataTransfer() = default;

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  OrtDevice::DeviceType src_type = src_device.Type();
  OrtDevice::DeviceType dst_type = dst_device.Type();

  // check that only our GPU is involved
  if ((src_type == OrtDevice::GPU && src_device.Vendor() != OrtDevice::VendorIds::AMD) ||
      (dst_type == OrtDevice::GPU && dst_device.Vendor() != OrtDevice::VendorIds::AMD)) {
    return false;
  }

  // copy must involve a GPU, and be device to device or cpu (exclude other device types)
  return (src_type == OrtDevice::GPU || dst_type == OrtDevice::GPU) &&
         (src_type == OrtDevice::GPU || src_type == OrtDevice::CPU) &&
         (dst_type == OrtDevice::GPU || dst_type == OrtDevice::CPU);
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  const bool dst_is_gpu_default = dst_device.Type() == OrtDevice::GPU &&
                                  dst_device.MemType() == OrtDevice::MemType::DEFAULT;
  const bool src_is_gpu_default = src_device.Type() == OrtDevice::GPU &&
                                  src_device.MemType() == OrtDevice::MemType::DEFAULT;

  // for the sync version of memcpy, launch to hip default stream
  if (dst_is_gpu_default) {
    if (src_is_gpu_default) {
      // Copy only if the two addresses are different.
      if (dst_data != src_data) {
        HIP_RETURN_IF_ERROR(hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToDevice));
        // Follow core/providers/cuda/gpu_data_transfer.cc to synchronize the default stream here.
        HIP_RETURN_IF_ERROR(hipStreamSynchronize(nullptr));
      }
    } else {
      // copy from other CPU memory to GPU, this is blocking
      HIP_RETURN_IF_ERROR(hipMemcpy(dst_data, src_data, bytes, hipMemcpyHostToDevice));
      if (src_device.MemType() != OrtDevice::MemType::HOST_ACCESSIBLE) {
        // Follow core/providers/cuda/gpu_data_transfer.cc to synchronize the default stream here.
        HIP_RETURN_IF_ERROR(hipStreamSynchronize(nullptr));
      }
    }
  } else if (src_is_gpu_default) {
    // copying from GPU to CPU memory, this is blocking
    HIP_RETURN_IF_ERROR(hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToHost));
  } else {
    // copying between cpu memory
    ORT_ENFORCE(dst_data != src_data);
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

common::Status GPUDataTransfer::CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  const bool dst_is_gpu_default = dst_device.Type() == OrtDevice::GPU &&
                                  dst_device.MemType() == OrtDevice::MemType::DEFAULT;
  const bool src_is_gpu_default = src_device.Type() == OrtDevice::GPU &&
                                  src_device.MemType() == OrtDevice::MemType::DEFAULT;

  auto hip_stream = static_cast<hipStream_t>(stream.GetHandle());

  if (dst_is_gpu_default) {
    if (src_is_gpu_default) {
      // D2D — always non-blocking
      HIP_CALL_THROW(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToDevice, hip_stream));
    } else if (src_device.MemType() == OrtDevice::MemType::HOST_ACCESSIBLE || bytes < kStagingThreshold) {
      // Pinned source or small transfer — hipMemcpyAsync is already truly async for pinned memory;
      // for tiny pageable transfers the staging overhead isn't worth it.
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyHostToDevice, hip_stream));
    } else {
      // Pageable source above threshold — stage through a pinned buffer so the
      // H2D DMA is truly async and the host thread returns immediately.
      void* pinned = staging_pool_.Acquire(bytes);
      if (pinned) {
        std::memcpy(pinned, src_data, bytes);
        auto err = hipMemcpyAsync(dst_data, pinned, bytes, hipMemcpyHostToDevice, hip_stream);
        if (err != hipSuccess) {
          staging_pool_.Release(pinned, bytes);
          HIP_RETURN_IF_ERROR(err);
        }
        auto cb = std::make_unique<StagingReturnInfo>(StagingReturnInfo{&staging_pool_, pinned, bytes});
        HIP_RETURN_IF_ERROR(hipLaunchHostFunc(hip_stream, StagingReturnCallback, cb.release()));
      } else {
        // hipHostMalloc failed — fall back to the (synchronous) direct path
        HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyHostToDevice, hip_stream));
      }
    }
  } else if (src_is_gpu_default) {
    if (dst_device.MemType() == OrtDevice::MemType::HOST_ACCESSIBLE || bytes < kStagingThreshold) {
      // Pinned dest or small transfer — hipMemcpyAsync is already efficient.
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToHost, hip_stream));
    } else {
      // Pageable dest above threshold — stage through pinned so the GPU→host
      // DMA runs as one large transfer instead of the driver's internal chunking.
      void* pinned = staging_pool_.Acquire(bytes);
      if (pinned) {
        auto err = hipMemcpyAsync(pinned, src_data, bytes, hipMemcpyDeviceToHost, hip_stream);
        if (err != hipSuccess) {
          staging_pool_.Release(pinned, bytes);
          HIP_RETURN_IF_ERROR(err);
        }
        HIP_RETURN_IF_ERROR(hipStreamSynchronize(hip_stream));
        std::memcpy(dst_data, pinned, bytes);
        staging_pool_.Release(pinned, bytes);
      } else {
        HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToHost, hip_stream));
      }
    }
  } else {
    if (src_device.MemType() == OrtDevice::MemType::HOST_ACCESSIBLE) {
      // sync the stream first to make sure the data arrived
      HIP_RETURN_IF_ERROR(hipStreamSynchronize(hip_stream));
    }
    ORT_ENFORCE(dst_data != src_data);
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

}  // namespace onnxruntime
