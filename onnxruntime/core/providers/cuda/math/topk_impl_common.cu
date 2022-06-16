#include "topk_impl.cuh"
namespace onnxruntime {
namespace cuda {
__global__ void ExcludeOutput(int64_t* output_i, int64_t K, int64_t dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, dimension);
  if (id >= K) {
    output_i[id] = dimension;
  }
}

} // namespace cuda
} // namespace onnxruntime
