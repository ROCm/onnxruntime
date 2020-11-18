// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

// unit for handling indexing and counting of gathered indices
using GatheredIndexIndex_t = int32_t;

template <typename T, typename TIndex>
void GatherGradImpl(
    const RocmKernel& rocm_kernel,
    const T* dY_data,
    const TIndex* dX_indices,
    const GatheredIndexIndex_t num_gathered_indices,
    const int64_t gather_dimension_size,
    const int64_t num_gathered_per_index,
    const int64_t num_batches,
    T* dX_data);

}  // namespace rocm
}  // namespace onnxruntime
