// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cfloat>

#include "core/providers/rocm/rocm_common.h"

#include <miopen/miopen.h>

const double MIOPEN_BN_MIN_EPSILON = 1e-5;

namespace onnxruntime {
namespace rocm {

#define MIOPEN_CONVOLUTION_FWD_ALGO_COUNT 6
#define MIOPEN_CONVOLUTION_BWD_FILTER_ALGO_COUNT 4
#define MIOPEN_CONVOLUTION_BWD_DATA_ALGO_COUNT 6

class MiopenTensor final {
 public:
  MiopenTensor();
  ~MiopenTensor();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MiopenTensor);

  Status Set(gsl::span<const int64_t> input_dims, miopenDataType_t dataType);
  Status Set(const MiopenTensor& x_desc, miopenBatchNormMode_t mode);

  operator miopenTensorDescriptor_t() const { return tensor_; }

  template <typename T>
  static miopenDataType_t GetDataType();

 private:
  Status CreateTensorIfNeeded();

  miopenTensorDescriptor_t tensor_;
};

class MiopenTensorDescriptor final {
 public:
  MiopenTensorDescriptor();
  ~MiopenTensorDescriptor();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MiopenTensorDescriptor);

  Status Set(gsl::span<const int64_t> filter_dims, miopenDataType_t data_typ);

  operator miopenTensorDescriptor_t() const { return desc_; }

 private:
  miopenTensorDescriptor_t desc_;
};

class MiopenDropout final {
 public:
  MiopenDropout() : dropout_desc_(nullptr) {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MiopenDropout);

  Status GetMiopenDropoutStatesSize(const miopenTensorDescriptor_t& miopenhandle, size_t& stateSize) {
    MIOPEN_RETURN_IF_ERROR(miopenDropoutGetReserveSpaceSize(miopenhandle, &stateSize));

    return Status::OK();
  }

  Status Set(const miopenTensorDescriptor_t& miopenhandle,
             void* states,
             size_t stateSize,
             float dropout = 0.0f,
             unsigned long long seed = 1) {
    ORT_RETURN_IF_ERROR(CreateDescriptorIfNeeded());
    MIOPEN_RETURN_IF_ERROR(miopenSetDropoutDescriptor(dropout_desc_,
                                                    miopenhandle,
                                                    dropout,
                                                    states,
                                                    stateSize,
						    seed,
						    false,//use mask?
						    false,
						    MIOPEN_RNG_PSEUDO_XORWOW));

    return Status::OK();
  }

  ~MiopenDropout() {
    if (dropout_desc_ != nullptr) {
      miopenDestroyDropoutDescriptor(dropout_desc_);
    }
  }

  operator miopenDropoutDescriptor_t() const {
    return dropout_desc_;
  }

  Status CreateDescriptorIfNeeded() {
    if (!dropout_desc_)
      MIOPEN_RETURN_IF_ERROR(miopenCreateDropoutDescriptor(&dropout_desc_));
    return Status::OK();
  }

 private:
  miopenDropoutDescriptor_t dropout_desc_;
};

template <typename ElemType>
struct Consts {
  static const ElemType Zero;
  static const ElemType One;
};

template <>
struct Consts<half> {
  static const float Zero;
  static const float One;
};

template <>
struct Consts<BFloat16> {
  static const float Zero;
  static const float One;
};

template <typename ElemType>
struct ReduceConsts {
  static const ElemType Zero;
  static const ElemType One;
};

#if ROCM_VERSION >= 40300
// Up until ROCm 4.2 miopenReduceTensor() required alpha/beta to be the same data
// type as the input type. This differs from cudnnReduceTensor() and other
// MIOpen/cuDNN APIs where alpha/beta are float when input type is half (float16).
template <>
struct ReduceConsts<half> {
  static const float Zero;
  static const float One;
};

template <>
struct ReduceConsts<BFloat16> {
  static const float Zero;
  static const float One;
};
#endif

inline double ClampMiopenBatchNormEpsilon(double epsilon) {
  if (epsilon < MIOPEN_BN_MIN_EPSILON) {
    if (MIOPEN_BN_MIN_EPSILON - epsilon > FLT_EPSILON)
      LOGS_DEFAULT(WARNING) << "Provided epsilon is smaller than MIOPEN_BN_MIN_EPSILON. Setting it to MIOPEN_BN_MIN_EPSILON";
    return MIOPEN_BN_MIN_EPSILON;
  }
  return epsilon;
}

}  // namespace rocm
}  // namespace onnxruntime
