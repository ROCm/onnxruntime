// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"

#include <miopen/miopen.h>

#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/miopen_common.h"

namespace onnxruntime {
namespace rocm {

enum RNN_Input_Index {
  X = 0,
  W = 1,
  R = 2,
  B = 3,
  sequence_lens = 4,
  initial_h = 5,
  initial_c = 6
};

// Onnx RNN/GRU/LSTM only support 1 layer
const int RNN_NUM_LAYERS = 1;

class CudnnRNN {
 public:
  CudnnRNN() : cudnn_rnn_desc_(nullptr) {
  }

  ~CudnnRNN() {
    if (cudnn_rnn_desc_ != nullptr) {
      miopenDestroyRNNDescriptor(cudnn_rnn_desc_);
      cudnn_rnn_desc_ = nullptr;
    }
  }

  Status Set(const miopenHandle_t& cudnnHandle, int64_t hidden_size, int num_layers,
             miopenDropoutDescriptor_t cudnn_dropout_desc, miopenRNNDirectionMode_t cudnn_direction_model,
             miopenRNNMode_t rnn_mode, miopenDataType_t dataType, const hipDeviceProp_t& prop) {
    if (!cudnn_rnn_desc_)
      MIOPEN_RETURN_IF_ERROR(miopenCreateRNNDescriptor(&cudnn_rnn_desc_));

    MIOPEN_RETURN_IF_ERROR(miopenSetRNNDescriptor_V2(cudnnHandle,
                                                cudnn_rnn_desc_,
                                                gsl::narrow_cast<int>(hidden_size),
                                                num_layers,
                                                cudnn_dropout_desc,
                                                miopenRNNlinear,  // We can also skip the input matrix transformation
                                                cudnn_direction_model,
                                                rnn_mode,
                                                miopenRNNdefault,  //HIPDNN_RNN_ALGO_PERSIST_STATIC, HIPDNN_RNN_ALGO_PERSIST_DYNAMIC
                                                dataType));
    return Status::OK();
  }

  operator miopenRNNDescriptor_t() const {
    return cudnn_rnn_desc_;
  }

 private:
  miopenRNNDescriptor_t cudnn_rnn_desc_;
};

template <typename T>
class CudnnRnnBase : public CudaKernel {
  const std::set<std::string> allowed_directions{"forward", "reverse", "bidirectional"};

 public:
  CudnnRnnBase(const OpKernelInfo& info) : CudaKernel{info} {
    reverse_ = false;
    std::string direction = "forward";
    direction = info.GetAttrOrDefault<std::string>("direction", "forward");
    cudnn_direction_mode_ = miopenRNNunidirection;
    if (direction == "bidirectional") {
      cudnn_direction_mode_ = miopenRNNbidirection;
    } else if (direction == "forward") {
      cudnn_direction_mode_ = miopenRNNunidirection;
    } else if (direction == "reverse") {
      cudnn_direction_mode_ = miopenRNNunidirection;
      // need to reverse data
      reverse_ = true;
    }

    num_directions_ = cudnn_direction_mode_ == miopenRNNbidirection ? 2 : 1;
    ORT_ENFORCE(allowed_directions.find(direction) != allowed_directions.end());

    ORT_ENFORCE(info.GetAttr("hidden_size", &hidden_size_).IsOK() && hidden_size_ > 0);
    rnn_mode_ = miopenLSTM;
    weight_cached_ = false;
    w_data_cache_ = nullptr;

    size_t state_size;
    ORT_THROW_IF_ERROR(cudnn_dropout_desc_.CreateDescriptorIfNeeded());
    ORT_THROW_IF_ERROR(cudnn_dropout_desc_.GetMiopenDropoutStatesSize(MiopenHandle(), state_size));
    state_buffer_ = GetScratchBuffer<void>(state_size);
    ORT_THROW_IF_ERROR(cudnn_dropout_desc_.Set(MiopenHandle(), state_buffer_.get(), state_size));

    layout_ = info.GetAttrOrDefault("layout", static_cast<int64_t>(0));
    ORT_ENFORCE(layout_ == 0,
                "Batchwise recurrent operations (layout == 1) are not supported. If you need support create a github issue with justification.");
  }

  Status CacheCudnnRnnWeights(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

  void SetRNNMode(miopenRNNMode_t rnn_mode) { rnn_mode_ = rnn_mode; }

 private:
  Status SetCudnnRnnWeightBias(const miopenHandle_t cudnn_handle,
                               const miopenOperatorDescriptor_t rnn_desc,
                               const miopenTensorDescriptor_t x_desc,
                               const miopenTensorDescriptor_t w_desc,
                               void* w_data,
                               const T* W_data,
                               const T* R_data,
                               const T* B_data) const;

  Status ReorganizeWeights(const Tensor* W, const Tensor* R, const Tensor* B,
                           IAllocatorUniquePtr<void>& target_w_data,
                           CudnnFilterDescriptor& target_w_desc,
                           CudnnRNN& rnn_desc) const;

  void SetWeightBias(const miopenHandle_t handle,
                     const miopenOperatorDescriptor_t rnn_desc,
                     const int pseudo_layer,
                     const miopenTensorDescriptor_t x_desc,
                     const miopenTensorDescriptor_t w_desc,
                     const miopenTensorDescriptor_t filter_desc,
                     const void* w_data,
                     const int lin_layer_id,
                     const T* pos,
                     int& offset,
                     bool is_matrix) const;

  void SetZeroSequences(const int64_t zero_seq_index_cache_size,
                        const std::vector<int32_t> zero_seq_index_cache,
                        T* y_data,
                        T* y_h_data,
                        T* y_c_data) const;

 protected:
  // W_lin_layer_id_ & R_lin_layer_id_ are set in Constructor
  std::vector<int> W_lin_layer_id_;
  std::vector<int> R_lin_layer_id_;

 private:
  miopenRNNDirectionMode_t cudnn_direction_mode_;
  bool reverse_;
  int64_t num_directions_;
  // hidden_size_ from attribute
  int64_t hidden_size_;
  miopenRNNMode_t rnn_mode_;
  // w_desc_cache_ & w_data_cache_ are changed in Constructor if we can get the weights as constant input
  CudnnFilterDescriptor w_desc_cache_;
  IAllocatorUniquePtr<void> w_data_cache_;
  bool weight_cached_;
  int64_t layout_;

  // cudnn_dropout_desc_ is a cache, never to be changed
  IAllocatorUniquePtr<void> state_buffer_;
  MiopenDropout cudnn_dropout_desc_;

  enum Output_Index {
    Y = 0,
    Y_h = 1,
    Y_c = 2
  };
};

}  // namespace rocm
}  // namespace onnxruntime
