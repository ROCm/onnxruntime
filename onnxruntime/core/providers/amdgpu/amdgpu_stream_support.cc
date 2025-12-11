// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "amdgpu_stream_support.h"
#include "amdgpu_factory.h"

//
// AmdGpuStreamImpl implementation
//

/*static*/
OrtStatus* ORT_API_CALL AmdGpuStreamImpl::CreateNotificationImpl(_In_ OrtSyncStreamImpl* this_ptr,
                                                                 _Outptr_ OrtSyncNotificationImpl** notification) noexcept {
  auto& impl = *static_cast<AmdGpuStreamImpl*>(this_ptr);
  *notification = std::make_unique<AmdGpuNotificationImpl>(impl).release();
  return nullptr;
}

/*static*/
void* ORT_API_CALL AmdGpuStreamImpl::GetHandleImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  auto& impl = *static_cast<AmdGpuStreamImpl*>(this_ptr);
  return impl.handle_;
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuStreamImpl::FlushImpl(_In_ OrtSyncStreamImpl* /*this_ptr*/) noexcept {
  // Skeleton: No actual stream to flush
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuStreamImpl::OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  // Skeleton: No arena allocator to reset
  return nullptr;
}

/*static*/
void ORT_API_CALL AmdGpuStreamImpl::ReleaseImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  delete static_cast<AmdGpuStreamImpl*>(this_ptr);
}

//
// AmdGpuNotificationImpl implementation
//

/*static*/
OrtStatus* ORT_API_CALL AmdGpuNotificationImpl::ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
  auto& impl = *static_cast<AmdGpuNotificationImpl*>(this_ptr);
  static_cast<void>(impl);

  // Skeleton: In real implementation, would call hipEventRecord
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuNotificationImpl::WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                                                 _In_ OrtSyncStream* stream) noexcept {
  if (stream == nullptr) {
    return nullptr;
  }

  auto& impl = *static_cast<AmdGpuNotificationImpl*>(this_ptr);

  void* handle = impl.ort_api.SyncStream_GetHandle(stream);
  static_cast<void>(handle);

  auto event = impl.event_;
  static_cast<void>(event);

  // Skeleton: In real implementation, would call hipStreamWaitEvent
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL AmdGpuNotificationImpl::WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
  auto& impl = *static_cast<AmdGpuNotificationImpl*>(this_ptr);

  auto event = impl.event_;
  static_cast<void>(event);

  // Skeleton: In real implementation, would call hipEventSynchronize
  return nullptr;
}

/*static*/
void ORT_API_CALL AmdGpuNotificationImpl::ReleaseImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
  delete static_cast<AmdGpuNotificationImpl*>(this_ptr);
}
