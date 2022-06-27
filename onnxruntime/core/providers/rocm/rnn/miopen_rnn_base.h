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
      hipdnnDestroyRNNDescriptor(cudnn_rnn_desc_);
      cudnn_rnn_desc_ = nullptr;
    }
  }

  Status Set(const hipdnnHandle_t& cudnnHandle, int64_t hidden_size, int num_layers,
             hipdnnDropoutDescriptor_t cudnn_dropout_desc, hipdnnDirectionMode_t cudnn_direction_model,
             hipdnnRNNMode_t rnn_mode, hipdnnDataType_t dataType, const hipDeviceProp_t& prop) {
    if (!cudnn_rnn_desc_)
      CUDNN_RETURN_IF_ERROR(hipdnnCreateRNNDescriptor(&cudnn_rnn_desc_));

    CUDNN_RETURN_IF_ERROR(hipdnnSetRNNDescriptor_v6(cudnnHandle,
                                                cudnn_rnn_desc_,
                                                gsl::narrow_cast<int>(hidden_size),
                                                num_layers,
                                                cudnn_dropout_desc,
                                                HIPDNN_LINEAR_INPUT,  // We can also skip the input matrix transformation
                                                cudnn_direction_model,
                                                rnn_mode,
                                                HIPDNN_RNN_ALGO_STANDARD,  //HIPDNN_RNN_ALGO_PERSIST_STATIC, HIPDNN_RNN_ALGO_PERSIST_DYNAMIC
                                                dataType));

    if (prop.major >= 7 && dataType == HIPDNN_DATA_HALF) {
      cudnnSetRNNMatrixMathType(cudnn_rnn_desc_, HIPDNN_TENSOR_OP_MATH);
    }

    return Status::OK();
  }

  operator hipdnnRNNDescriptor_t() const {
    return cudnn_rnn_desc_;
  }

 private:
  hipdnnRNNDescriptor_t cudnn_rnn_desc_;
};

template <typename T>
class CudnnRnnBase : public CudaKernel {
  const std::set<std::string> allowed_directions{"forward", "reverse", "bidirectional"};

 public:
  CudnnRnnBase(const OpKernelInfo& info) : CudaKernel{info} {
    reverse_ = false;
    std::string direction = "forward";
    direction = info.GetAttrOrDefault<std::string>("direction", "forward");
    cudnn_direction_mode_ = HIPDNN_UNIDIRECTIONAL;
    if (direction == "bidirectional") {
      cudnn_direction_mode_ = HIPDNN_BIDIRECTIONAL;
    } else if (direction == "forward") {
      cudnn_direction_mode_ = HIPDNN_UNIDIRECTIONAL;
    } else if (direction == "reverse") {
      cudnn_direction_mode_ = HIPDNN_UNIDIRECTIONAL;
      // need to reverse data
      reverse_ = true;
    }

    num_directions_ = cudnn_direction_mode_ == HIPDNN_BIDIRECTIONAL ? 2 : 1;
    ORT_ENFORCE(allowed_directions.find(direction) != allowed_directions.end());

    ORT_ENFORCE(info.GetAttr("hidden_size", &hidden_size_).IsOK() && hidden_size_ > 0);
    rnn_mode_ = HIPDNN_LSTM;
    weight_cached_ = false;
    w_data_cache_ = nullptr;

    size_t state_size;
    ORT_THROW_IF_ERROR(cudnn_dropout_desc_.CreateDescriptorIfNeeded());
    ORT_THROW_IF_ERROR(cudnn_dropout_desc_.GetCudnnDropoutStatesSize(CudnnHandle(), state_size));
    state_buffer_ = GetScratchBuffer<void>(state_size);
    ORT_THROW_IF_ERROR(cudnn_dropout_desc_.Set(CudnnHandle(), state_buffer_.get(), state_size));

    layout_ = info.GetAttrOrDefault("layout", static_cast<int64_t>(0));
    ORT_ENFORCE(layout_ == 0,
                "Batchwise recurrent operations (layout == 1) are not supported. If you need support create a github issue with justification.");
  }

  Status CacheCudnnRnnWeights(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

  void SetRNNMode(hipdnnRNNMode_t rnn_mode) { rnn_mode_ = rnn_mode; }

 private:
  Status SetCudnnRnnWeightBias(const hipdnnHandle_t cudnn_handle,
                               const hipdnnRNNDescriptor_t rnn_desc,
                               const hipdnnTensorDescriptor_t x_desc,
                               const hipdnnFilterDescriptor_t w_desc,
                               void* w_data,
                               const T* W_data,
                               const T* R_data,
                               const T* B_data) const;

  Status ReorganizeWeights(const Tensor* W, const Tensor* R, const Tensor* B,
                           IAllocatorUniquePtr<void>& target_w_data,
                           CudnnFilterDescriptor& target_w_desc,
                           CudnnRNN& rnn_desc) const;

  void SetWeightBias(const hipdnnHandle_t handle,
                     const hipdnnRNNDescriptor_t rnn_desc,
                     const int pseudo_layer,
                     const hipdnnTensorDescriptor_t x_desc,
                     const hipdnnFilterDescriptor_t w_desc,
                     const hipdnnFilterDescriptor_t filter_desc,
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
  hipdnnDirectionMode_t cudnn_direction_mode_;
  bool reverse_;
  int64_t num_directions_;
  // hidden_size_ from attribute
  int64_t hidden_size_;
  hipdnnRNNMode_t rnn_mode_;
  // w_desc_cache_ & w_data_cache_ are changed in Constructor if we can get the weights as constant input
  CudnnFilterDescriptor w_desc_cache_;
  IAllocatorUniquePtr<void> w_data_cache_;
  bool weight_cached_;
  int64_t layout_;

  // cudnn_dropout_desc_ is a cache, never to be changed
  IAllocatorUniquePtr<void> state_buffer_;
  CudnnDropout cudnn_dropout_desc_;

  enum Output_Index {
    Y = 0,
    Y_h = 1,
    Y_c = 2
  };
};

}  // namespace rocm
}  // namespace onnxruntime
