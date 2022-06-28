// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "hip/hip_runtime.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

template<typename T>
void ReverseBySequence(hipStream_t stream,
                       const int32_t seq_length,
                       const int32_t batch_size,
                       const int32_t input_or_hidden_size,
                       const T* data,
                       T* reversed_data,
                       const size_t N);

template <typename T>
void ReorderBidirectionalDataInSequence(hipStream_t stream,
                                        const int32_t seq_length,
                                        const int32_t batch_size,
                                        const int32_t hidden_size,
                                        const T* data,
                                        T* reordered_data,
                                        const size_t N);

template <typename T>
void RnnMaskImpl(hipStream_t stream,
                 const int32_t num_directions,
                 const int32_t seq_length,
                 const int32_t batch_size,
                 const int32_t hidden_size,
                 const int32_t* sequence_lens,
                 T* y_output_data,
                 T* y_h_output_data,
                 const size_t N);

template <typename T>
void MaskZeroSequences(hipStream_t stream,
                       const int32_t hidden_size,
                       T* y_output_data,
                       T* y_h_output_data,
                       T* y_c_output_data,
                       const int32_t* zeor_seq_index_cache_async_buffer,
                       const size_t N);
}  // namespace rocm
}  // namespace onnxruntime
