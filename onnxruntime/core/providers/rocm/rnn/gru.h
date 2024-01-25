// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "miopen_rnn_base.h"
#include "core/common/gsl.h"
#include "core/providers/rocm/rocm_common.h"
#include <miopen/miopen.h>

namespace onnxruntime {
namespace rocm {

template <typename T>
class GRU final : public MiopenRnnBase<T> {
 public:
  GRU(const OpKernelInfo& info) : MiopenRnnBase<T>(info) {
    MiopenRnnBase<T>::SetRNNMode(miopenGRU);

    // ONNX W layout is Wzrh, WBzrh, mapping to RNNLinLayerMatrixParams the linLayerID is 1, 0, 2
    MiopenRnnBase<T>::W_lin_layer_id_.assign({1, 0, 2});
    // ONNX R layout is Rzrh, RBzrh, mapping to RNNLinLayerMatrixParams the linLayerID is 4, 3, 5
    MiopenRnnBase<T>::R_lin_layer_id_.assign({4, 3, 5});
    // ONNX B layout is Wbzrh, Rbzrh, mapping to RNNLinLayerMatrixParams
    // the linLayerID is 1, 0, 2, 4, 3, 5, we can reuse it from W_lin_layer_id & R_lin_layer_id

    ORT_THROW_IF_ERROR(MiopenRnnBase<T>::CacheMiopenRnnWeights(info));
  }
};

}  // namespace rocm
}  // namespace onnxruntime
