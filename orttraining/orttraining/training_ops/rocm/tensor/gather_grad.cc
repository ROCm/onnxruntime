// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/rocm/tensor/gather_grad.h"

#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "orttraining/training_ops/rocm/tensor/gather_grad_impl.h"

namespace onnxruntime {
namespace rocm {

ONNX_OPERATOR_KERNEL_EX(
    GatherGrad,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    GatherGrad);

namespace {
template <typename T, typename TIndex>
Status CallGatherGradImpl(
    const RocmKernel& rocm_kernel,
    int64_t num_gathered_per_index, int64_t gather_dimension_size, int64_t num_batches,
    const Tensor& dY, const Tensor& gathered_indices,
    Tensor& dX) {
  using HipT = typename ToHipType<T>::MappedType;

  const T* dY_data = dY.template Data<T>();
  T* dX_data = dX.template MutableData<T>();
  const TIndex* indices_data = gathered_indices.template Data<TIndex>();

  const SafeInt<GatheredIndexIndex_t> num_gathered_indices{gathered_indices.Shape().Size()};

  GatherGradImpl(
      rocm_kernel,
      reinterpret_cast<const HipT*>(dY_data),
      indices_data,
      num_gathered_indices,
      gather_dimension_size,
      num_gathered_per_index,
      num_batches,
      reinterpret_cast<HipT*>(dX_data));

  return Status::OK();
}

template <typename T>
Status DispatchToGatherGradImplByTindex(
    MLDataType tindex_data_type,
    const RocmKernel& rocm_kernel,
    int64_t num_gathered_per_index, int64_t gather_dimension_size, int64_t num_batches,
    const Tensor& dY, const Tensor& gathered_indices,
    Tensor& dX) {
  if (utils::IsPrimitiveDataType<int32_t>(tindex_data_type)) {
    return CallGatherGradImpl<T, int32_t>(
        rocm_kernel, num_gathered_per_index, gather_dimension_size, num_batches, dY, gathered_indices, dX);
  } else if (utils::IsPrimitiveDataType<int64_t>(tindex_data_type)) {
    return CallGatherGradImpl<T, int64_t>(
        rocm_kernel, num_gathered_per_index, gather_dimension_size, num_batches, dY, gathered_indices, dX);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GatherGrad unsupported TIndex type: ", tindex_data_type);
}

Status DispatchToGatherGradImpl(
    MLDataType t_data_type, MLDataType tindex_data_type,
    const RocmKernel& rocm_kernel,
    int64_t num_gathered_per_index, int64_t gather_dimension_size, int64_t num_batches,
    const Tensor& dY, const Tensor& gathered_indices,
    Tensor& dX) {
  if (utils::IsPrimitiveDataType<float>(t_data_type)) {
    return DispatchToGatherGradImplByTindex<float>(
        tindex_data_type, rocm_kernel, num_gathered_per_index, gather_dimension_size, num_batches, dY, gathered_indices, dX);
  } else if (utils::IsPrimitiveDataType<MLFloat16>(t_data_type)) {
    return DispatchToGatherGradImplByTindex<MLFloat16>(
        tindex_data_type, rocm_kernel, num_gathered_per_index, gather_dimension_size, num_batches, dY, gathered_indices, dX);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GatherGrad unsupported T type: ", t_data_type);
}
}  // namespace

Status GatherGrad::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X_shape_tensor = context->Input<Tensor>(0);
  const TensorShape X_shape(X_shape_tensor->template Data<int64_t>(), X_shape_tensor->Shape().Size());
  const Tensor* gathered_indices = context->Input<Tensor>(1);
  const Tensor* dY = context->Input<Tensor>(2);

  Tensor* dX = context->Output(0, X_shape);
  HIP_RETURN_IF_ERROR(hipMemsetAsync(dX->MutableDataRaw(), 0, dX->SizeInBytes()));

  if (gathered_indices->Shape().Size() == 0) {
    // nothing else to do
    return Status::OK();
  }

  MLDataType t_type = dY->DataType();
  MLDataType tindex_type = gathered_indices->DataType();

  const auto axis = HandleNegativeAxis(axis_, X_shape.NumDimensions());
  const int64_t num_gathered_per_index = X_shape.SizeFromDimension(axis + 1);
  const int64_t gather_dimension_size = X_shape[axis];
  const int64_t num_batches = X_shape.SizeToDimension(axis);

  return DispatchToGatherGradImpl(
      t_type, tindex_type, *this,
      num_gathered_per_index, gather_dimension_size, num_batches,
      *dY, *gathered_indices, *dX);
}

}  // namespace rocm
}  // namespace onnxruntime
