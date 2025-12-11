// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"
#include "amdgpu_provider_utils.h"

class AmdGpuEpFactory;

/// <summary>
/// Stream implementation for AMD GPU EP.
/// Handles synchronization for async GPU operations.
/// This is a skeleton implementation.
/// </summary>
class AmdGpuStreamImpl : public OrtSyncStreamImpl, public ApiPtrs {
 public:
  AmdGpuStreamImpl(AmdGpuEpFactory& factory, const OrtEp* ep, const OrtKeyValuePairs* stream_options)
      : ApiPtrs(factory), factory_{&factory} {
    // `ep` is the EP instance if the stream is being created internally for inferencing.
    // nullptr when the stream is created outside of an inference session for data copies.
    static_cast<void>(ep);
    static_cast<void>(stream_options);

    ort_version_supported = ORT_API_VERSION;
    CreateNotification = CreateNotificationImpl;
    GetHandle = GetHandleImpl;
    Flush = FlushImpl;
    OnSessionRunEnd = OnSessionRunEndImpl;
    Release = ReleaseImpl;
  }

 private:
  static OrtStatus* ORT_API_CALL CreateNotificationImpl(_In_ OrtSyncStreamImpl* this_ptr,
                                                        _Outptr_ OrtSyncNotificationImpl** sync_notification) noexcept;
  static void* ORT_API_CALL GetHandleImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL FlushImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;

  void* handle_{nullptr};  // Skeleton: would be hipStream_t in real implementation

  AmdGpuEpFactory* factory_{nullptr};
};

/// <summary>
/// Notification implementation for AMD GPU EP.
/// Handles synchronization events.
/// This is a skeleton implementation.
/// </summary>
class AmdGpuNotificationImpl : public OrtSyncNotificationImpl, public ApiPtrs {
 public:
  AmdGpuNotificationImpl(ApiPtrs apis) : ApiPtrs(apis) {
    ort_version_supported = ORT_API_VERSION;
    Activate = ActivateImpl;
    Release = ReleaseImpl;
    WaitOnDevice = WaitOnDeviceImpl;
    WaitOnHost = WaitOnHostImpl;
  }

 private:
  static OrtStatus* ORT_API_CALL ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                                  _In_ OrtSyncStream* stream) noexcept;
  static OrtStatus* ORT_API_CALL WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;

  void* event_{nullptr};  // Skeleton: would be hipEvent_t in real implementation
};
