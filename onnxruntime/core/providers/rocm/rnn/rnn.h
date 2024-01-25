// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "miopen_rnn_base.h"
#include "core/providers/rocm/rocm_common.h"
#include <miopen/miopen.h>

namespace onnxruntime {
namespace rocm {

template <typename T>
class RNN final : public MiopenRnnBase<T> {
  const std::set<std::string> allowed_activations{"Relu", "Tanh" /*, "Sigmoid"*/};

 public:
  RNN(const OpKernelInfo& info) : MiopenRnnBase<T>(info) {
    std::vector<std::string> activations_;
    ORT_ENFORCE(info.GetAttrs("activations", activations_).IsOK());
    if (activations_[0] == "Relu")
      MiopenRnnBase<T>::SetRNNMode(miopenRNNRELU);
    else if (activations_[0] == "Tanh")
      MiopenRnnBase<T>::SetRNNMode(miopenRNNTANH);

    // ONNX W mapping to RNNLinLayerMatrixParams the linLayerID is 0
    MiopenRnnBase<T>::W_lin_layer_id_.assign({0});
    // ONNX R mapping to RNNLinLayerMatrixParams the linLayerID is 1
    MiopenRnnBase<T>::R_lin_layer_id_.assign({1});
    // ONNX B layout is Wb, Rb, mapping to RNNLinLayerMatrixParams
    // the linLayerID is 0, 1, we can reuse it from W_lin_layer_id & R_lin_layer_id

    ORT_THROW_IF_ERROR(MiopenRnnBase<T>::CacheMiopenRnnWeights(info));
  }
};

}  // namespace rocm
}  // namespace onnxruntime
