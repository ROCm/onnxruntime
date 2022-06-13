// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/longformer_global_impl.h"
#include "contrib_ops/rocm/bert/longformer_attention.h"
#include "contrib_ops/rocm/bert/longformer_attention_impl.h"
#include "contrib_ops/rocm/bert/transformer_rocm_common.h"
#include "contrib_ops/rocm/bert/transformer_common.h"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LongformerAttention,                                        \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LongformerAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
LongformerAttention<T>::LongformerAttention(const OpKernelInfo& info)
  : RocmKernel(info), LongformerAttentionBase(info) {
  use_compact_memory_ = ParseEnvironmentVariableWithDefault<bool>(longformer::kUseCompactMemory, true);
}

template <typename T>
Status LongformerAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask = context->Input<Tensor>(3);
  const Tensor* global_weights = context->Input<Tensor>(4);
  const Tensor* global_bias = context->Input<Tensor>(5);
  const Tensor* global_attention = context->Input<Tensor>(6);
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), weights->Shape(), bias->Shape(), mask->Shape(),
                                  global_weights->Shape(), global_bias->Shape(), global_attention->Shape()));

  // Input and output shapes:
  //   Input 0 - input       : (batch_size, sequence_length, hidden_size)
  //   Output 0 - output     : (batch_size, sequence_length, hidden_size)
  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int hidden_size = static_cast<int>(shape[2]);
  int head_size = hidden_size / num_heads_;

  Tensor* output = context->Output(0, shape);

  rocblas_handle rocblas = RocblasHandle();
  hipStream_t stream = Stream();
  ROCBLAS_RETURN_IF_ERROR(rocblas_set_stream(rocblas, stream));

  constexpr size_t element_size = sizeof(T);

  // TODO: only calculate once per model.
  // Build Global Index
  auto global_index_buffer = GetScratchBuffer<int>(batch_size * sequence_length);
  auto batch_global_num_buffer = GetScratchBuffer<int>(batch_size);

  size_t global_scratch_bytes = GetGlobalScratchSize(batch_size, sequence_length);
  auto global_scratch_buffer = GetScratchBuffer<void>(global_scratch_bytes);

  BuildGlobalIndex(
      stream,
      global_attention->template Data<int>(),
      batch_size,
      sequence_length,
      global_index_buffer.get(),
      batch_global_num_buffer.get(),
      global_scratch_buffer.get(),
      global_scratch_bytes);

  // Copy batch_global_num to CPU
  size_t pinned_buffer_bytes = GetPinnedBufferSize(batch_size);
  auto pinned_buffer = AllocateBufferOnCPUPinned<void>(pinned_buffer_bytes);
  int* batch_global_num_pinned = reinterpret_cast<int*>(pinned_buffer.get());
  HIP_RETURN_IF_ERROR(hipMemcpyAsync(batch_global_num_pinned,
                                       batch_global_num_buffer.get(),
                                       batch_size * sizeof(int),
                                       hipMemcpyDeviceToHost,
                                       stream));

  // Create an event to make sure the async copy is finished before reading the data.
  AutoDestoryCudaEvent new_event;
  hipEvent_t& isCopyDone = new_event.Get();

  HIP_RETURN_IF_ERROR(hipEventCreate(&isCopyDone));
  HIP_RETURN_IF_ERROR(hipEventRecord(isCopyDone, stream));

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = hidden_size;

  size_t qkv_size = batch_size * sequence_length * 3 * hidden_size * element_size;
  auto gemm_buffer = GetScratchBuffer<void>(qkv_size);

  typedef typename ToHipType<T>::MappedType HipT;
  HipT one = ToHipType<T>::FromFloat(1.0f);
  HipT zero = ToHipType<T>::FromFloat(0.0f);

  // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  auto& device_prop = GetDeviceProp();
  ROCBLAS_RETURN_IF_ERROR(rocblasGemmHelper(
      rocblas, rocblas_operation_none, rocblas_operation_none, n, m, 1, &one,
      reinterpret_cast<const HipT*>(bias->template Data<T>()), n,
      GetConstOnes<HipT>(m), 1,
      &zero, reinterpret_cast<HipT*>(gemm_buffer.get()), n));

  // Gemm, note that ROCM assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
  ROCBLAS_RETURN_IF_ERROR(rocblasGemmHelper(
      rocblas, rocblas_operation_none, rocblas_operation_none, n, m, k, &one,
      reinterpret_cast<const HipT*>(weights->template Data<T>()), n,
      reinterpret_cast<const HipT*>(input->template Data<T>()), k,
      &one, reinterpret_cast<HipT*>(gemm_buffer.get()), n));

  // Wait for async copy of batch_global_num
  HIP_RETURN_IF_ERROR(hipEventSynchronize(isCopyDone));

  // Find the maximum number of global tokens in all batches
  int max_num_global = 0;
  for (int i = 0; i < batch_size; ++i) {
    if (max_num_global < batch_global_num_pinned[i]) {
      max_num_global = batch_global_num_pinned[i];
    }
  }

  // Do not use compact kernel in the following situations:
  // (1) global tokens > windows size, compact memory kernel cannot be used due to its assumptions.
  // (2) sequence_length == 2 * attention_window, compact memory kernel has parity issue.
  // (3) user sets environment variable ORT_LONGFORMER_COMPACT_MEMORY=0
  bool disable_compact_memory = (max_num_global > window_ || sequence_length == 2 * window_ || !use_compact_memory_);

  // Fully connection for global projection.
  // Note that Q only need handle global query tokens if we split GEMM to global Q/K/V separately.
  // When there is no global token, need not run glboal GEMM.
  auto global_gemm_buffer = GetScratchBuffer<void>(max_num_global > 0 ? qkv_size : 0);

  if (max_num_global > 0) {
    ROCBLAS_RETURN_IF_ERROR(rocblasGemmHelper(
        rocblas, rocblas_operation_none, rocblas_operation_none, n, m, 1, &one,
        reinterpret_cast<const HipT*>(global_bias->template Data<T>()), n,
        GetConstOnes<HipT>(m), 1,
        &zero, reinterpret_cast<HipT*>(global_gemm_buffer.get()), n));

    ROCBLAS_RETURN_IF_ERROR(rocblasGemmHelper(
        rocblas, rocblas_operation_none, rocblas_operation_none, n, m, k, &one,
        reinterpret_cast<const HipT*>(global_weights->template Data<T>()), n,
        reinterpret_cast<const HipT*>(input->template Data<T>()), k,
        &one, reinterpret_cast<HipT*>(global_gemm_buffer.get()), n));
  }

  size_t workSpaceSize = GetLongformerAttentionWorkspaceSize(element_size,
                                                             batch_size,
                                                             num_heads_,
                                                             head_size,
                                                             sequence_length,
                                                             max_num_global,
                                                             window_,
                                                             disable_compact_memory);
  auto workspace_buffer = GetScratchBuffer<void>(workSpaceSize);
  if (!LaunchLongformerAttentionKernel(
          device_prop,
          rocblas,
          stream,
          reinterpret_cast<const HipT*>(gemm_buffer.get()),
          reinterpret_cast<const HipT*>(mask->template Data<T>()),
          reinterpret_cast<const HipT*>(global_gemm_buffer.get()),
          global_attention->template Data<int>(),
          global_index_buffer.get(),
          batch_global_num_buffer.get(),
          pinned_buffer.get(),
          workspace_buffer.get(),
          output->template MutableData<T>(),
          batch_size,
          sequence_length,
          num_heads_,
          head_size,
          window_,
          max_num_global,
          element_size,
          disable_compact_memory)) {
    // Get last error to reset it to hipSuccess.
    HIP_CALL(hipGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  // Defer release of pinnned memory since hipStreamSynchronize is not used here and kernel need access the buffer.
  this->AddDeferredReleaseCPUPtr(pinned_buffer.release());

  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
