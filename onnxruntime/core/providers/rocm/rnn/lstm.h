// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "miopen_rnn_base.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
class LSTM final : public MiopenRnnBase<T> {
 public:
  LSTM(const OpKernelInfo& info) : MiopenRnnBase<T>(info) {
    MiopenRnnBase<T>::SetRNNMode(miopenLSTM);

    // ONNX W layout is W[iofc], WB[iofc], mapping to RNNLinLayerMatrixParams the linLayerID is 0, 3, 1, 2
    MiopenRnnBase<T>::W_lin_layer_id_.assign({0, 3, 1, 2});
    // ONNX R layout is R[iofc], RB[iofc], mapping to RNNLinLayerMatrixParams the linLayerID is 4, 7, 5, 6
    MiopenRnnBase<T>::R_lin_layer_id_.assign({4, 7, 5, 6});
    // ONNX B layout is Wb[iofc], Rb[iofc], mapping to RNNLinLayerMatrixParams
    // the linLayerID is 0, 3, 1, 2, 4, 7, 5, 6, we can reuse it from W_lin_layer_id & R_lin_layer_id

    ORT_THROW_IF_ERROR(MiopenRnnBase<T>::CacheMiopenRnnWeights(info));
  }
};

}  // namespace rocm
}  // namespace onnxruntime
