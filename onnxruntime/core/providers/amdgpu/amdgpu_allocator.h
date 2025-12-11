// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "amdgpu_provider_utils.h"

/// <summary>
/// Allocator for AMD GPU memory.
/// This is a skeleton implementation that uses CPU memory.
/// A real implementation would use hipMalloc/hipFree for GPU memory allocation.
/// </summary>
class AmdGpuAllocator {
 public:
  static void* Alloc(size_t size) {
    // Skeleton: Use CPU allocation
    // Real implementation would use hipMalloc
    return std::malloc(size);
  }

  static void Free(void* ptr) {
    // Skeleton: Use CPU free
    // Real implementation would use hipFree
    std::free(ptr);
  }
};
