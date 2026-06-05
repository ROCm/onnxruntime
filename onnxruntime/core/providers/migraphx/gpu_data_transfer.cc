// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/gpu_data_transfer.h"
#include "core/providers/migraphx/migraphx_call.h"

// If you make change below, please also update onnxruntime/core/providers/rocm/gpu_data_transfer.cc

namespace onnxruntime {

GPUDataTransfer::~GPUDataTransfer() {
  // Make sure no outstanding async copies are still referencing pinned
  // staging buffers before we tear the reaper / pools down.
  (void)hipDeviceSynchronize();
}

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

  // Use the EP's compute stream (non-blocking) instead of the default (null)
  // stream to avoid the implicit cross-stream serialisation that the default
  // stream imposes on all other streams.
  if (dst_is_gpu_default) {
    if (src_is_gpu_default) {
      if (dst_data != src_data) {
        HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes,
                                           hipMemcpyDeviceToDevice, stream_));
        HIP_RETURN_IF_ERROR(hipStreamSynchronize(stream_));
      }
    } else {
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes,
                                         hipMemcpyHostToDevice, stream_));
      HIP_RETURN_IF_ERROR(hipStreamSynchronize(stream_));
    }
  } else if (src_is_gpu_default) {
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes,
                                       hipMemcpyDeviceToHost, stream_));
    HIP_RETURN_IF_ERROR(hipStreamSynchronize(stream_));
  } else {
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
        // Hand the pinned buffer to the reaper, which will return it to the
        // pool once the recorded event reports complete.  This replaces the
        // previous hipLaunchHostFunc-based release, which serialised on the
        // compute stream's host-function dispatcher and could deadlock when
        // Release happened to invoke hipHostFree under heavy load.
        hipEvent_t e = event_pool_.Acquire();
        if (!e) {
          // Event allocation failed — degrade gracefully by syncing on the
          // calling thread and releasing the buffer immediately.
          HIP_RETURN_IF_ERROR(hipStreamSynchronize(hip_stream));
          staging_pool_.Release(pinned, bytes);
        } else {
          auto rec_err = hipEventRecord(e, hip_stream);
          if (rec_err != hipSuccess) {
            // Recording failed — same fallback as above, plus return the
            // event to its pool for reuse.
            (void)hipStreamSynchronize(hip_stream);
            staging_pool_.Release(pinned, bytes);
            event_pool_.Release(e);
            HIP_RETURN_IF_ERROR(rec_err);
          }
          reaper_.Submit(e, pinned, bytes);
        }
      } else {
        // hipHostMalloc failed — fall back to the direct path.  Note that
        // hipMemcpyAsync on pageable memory is effectively synchronous from
        // the host's point of view (the runtime stages internally), which
        // is the desired behaviour in this degraded path.
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
