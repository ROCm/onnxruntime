// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/migraphx/migraphx_execution_provider_info.h"
#include "core/providers/migraphx/migraphx_call.h"

using namespace std::literals::string_view_literals;

namespace onnxruntime {

namespace migraphx_env_vars {
constexpr auto kCompileTarget = "ORT_MIGRAPHX_COMPILE_TARGET"sv;
constexpr auto kFP16Enable = "ORT_MIGRAPHX_FP16_ENABLE"sv;
constexpr auto kBF16Enable = "ORT_MIGRAPHX_BF16_ENABLE"sv;
constexpr auto kFP8Enable = "ORT_MIGRAPHX_FP8_ENABLE"sv;
constexpr auto kINT8Enable = "ORT_MIGRAPHX_INT8_ENABLE"sv;
constexpr auto kDumpModelOps = "ORT_MIGRAPHX_DUMP_MODEL_OPS"sv;
constexpr auto kINT8CalibrationTableName = "ORT_MIGRAPHX_INT8_CALIBRATION_TABLE_NAME"sv;
constexpr auto kCachePath = "ORT_MIGRAPHX_CACHE_PATH"sv;
constexpr auto kINT8UseNativeMIGraphXCalibrationTable = "ORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE"sv;
constexpr auto kExhaustiveTune = "ORT_MIGRAPHX_EXHAUSTIVE_TUNE"sv;
constexpr auto kModelCachePath = "ORT_MIGRAPHX_MODEL_CACHE_PATH"sv;
constexpr auto kModelMaxDynamicBatch = "ORT_MIGRAPHX_MAX_DYNAMIC_BATCH"sv;
constexpr auto kCompileBatches = "ORT_MIGRAPHX_COMPILE_BATCHES"sv;
constexpr auto kHipGraphEnable = "ORT_MIGRAPHX_HIP_GRAPH_ENABLE"sv;
}  // namespace migraphx_env_vars

// Tracks which dimensions are symbolic for a given input
struct SymbolicDimInfo {
  int dim_index;                // The dimension index (0 = batch, 1, 2, ...)
  std::string dim_param;        // The symbolic parameter name (e.g., "batch", "sequence_length")
};

// Information to construct kernel function state.
struct MIGraphXFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocate_handle = nullptr;
  migraphx::program prog{};
  std::string onnx_string;
  migraphx::onnx_options options;
  migraphx::target t{};
  std::unordered_map<std::string, std::size_t> input_name_indexes;
  std::mutex* mgx_mu_ptr = nullptr;
  hipStream_t stream = nullptr;
  bool defer_compilation = false;
  bool fp16_enable = false;
  bool bf16_enable = false;
  bool fp8_enable = false;
  bool int8_enable = false;
  bool int8_calibration_cache_available = false;
  std::unordered_map<std::string, float> dynamic_range_map;
  std::filesystem::path model_cache_dir;
  bool dump_model_ops = false;
  bool exhaustive_tune = false;
  size_t max_dynamic_batch = 0;
  // Reference to the cached programs map for this node (keyed by input shape hash)
  std::optional<std::reference_wrapper<std::unordered_map<std::string, migraphx::program>>> cached_programs_ref = std::nullopt;
  
  // Dynamic batch support
  bool has_dynamic_batch = false;
  std::vector<std::size_t> compiled_batch_sizes;
  
  // Pinned I/O buffers: allocated once at max compiled batch, reused across all inferences.
  // Eliminates per-inference hipMalloc/hipFree for padding and temp outputs.
  struct PinnedIOBuffer {
    void* data = nullptr;
    std::size_t size_bytes = 0;
    migraphx::shape max_shape;     // Shape at max_batch_size
  };

  struct PinnedIOSet {
    std::vector<PinnedIOBuffer> inputs;
    std::vector<PinnedIOBuffer> outputs;
    std::unordered_map<std::string, std::size_t> input_name_to_idx;
    std::unordered_map<std::string, std::size_t> output_name_to_idx;
    std::size_t max_batch_size = 0;
    bool allocated = false;
  };

  PinnedIOSet pinned_io;

  // ═══════════════════════════════════════════════════════════════════════════
  // SCRATCH BUFFERS (one per compiled program / shape_hash)
  //
  // MIGraphX programs expose a "scratch" parameter that the EP must bind to
  // a device buffer; otherwise MIGraphX falls back to its own internal scratch
  // arena whose contents persist across runs and whose lifetime is opaque to
  // hipGraph capture/replay.  When we capture a hipGraph that contains kernels
  // which read scratch before writing within a single invocation (common with
  // split-K reductions, fused-attention epilogues, etc.), the captured kernel
  // sees whatever bytes happened to be in that opaque arena at capture time,
  // and every subsequent replay inherits the same dependency on whatever the
  // *previous* replay left behind.  That is the root cause of non-deterministic
  // back-to-back replays on identical input.
  //
  // By owning the scratch buffer in the EP and zeroing it before every replay
  // (and before capture), we anchor each replay to the same memory baseline
  // and eliminate the cross-run state bleed.  One buffer per shape_hash means
  // every compiled batch-size variant gets its own correctly-sized arena.
  // ═══════════════════════════════════════════════════════════════════════════
  struct ScratchBuf {
    void* data = nullptr;
    std::size_t size_bytes = 0;
    migraphx::shape mgx_shape;
  };
  std::unordered_map<std::string, ScratchBuf> scratch_bufs;

  // ═══════════════════════════════════════════════════════════════════════════
  // PERFORMANCE CACHES - Avoid redundant MIGraphX API calls per inference
  // ═══════════════════════════════════════════════════════════════════════════

  // Cached input parameter info (name as const char*, ORT index, MIGraphX shape)
  struct CachedInputParam {
    std::string name;              // Parameter name (owns the string)
    std::size_t ort_index;         // ORT input index
    migraphx::shape mgx_shape;     // MIGraphX shape for this input
  };

  // Cached output parameter info (name as const char*, output index, MIGraphX shape)
  struct CachedOutputParam {
    std::string name;              // Parameter name (owns the string)
    int output_index;              // ORT output index
    migraphx::shape mgx_shape;     // MIGraphX shape for this output
  };

  // Separated input/output parameter lists for O(1) iteration without map lookups
  std::vector<CachedInputParam> cached_inputs;
  std::vector<CachedOutputParam> cached_outputs;

  // Pre-allocated output shapes in ORT format (avoids vector allocation per inference)
  std::vector<std::vector<int64_t>> cached_output_ort_shapes;

  // Cached program_parameters object for ultra-fast rebinding
  std::optional<migraphx::program_parameters> cached_prog_params;

  // Cached output indices for pre-allocated outputs (used by run_migraphx_program)
  std::vector<std::size_t> cached_prog_output_indices;
  std::vector<std::size_t> cached_pinned_output_indices;

  // Last input shapes for quick comparison (avoids hash computation in ultra-fast path)
  std::vector<std::int64_t> last_input_shapes_raw;

  // Last input shape hash (only computed when shapes change, used for cache lookup)
  std::string last_input_shape_hash;

  // Flag indicating caches are valid
  bool caches_valid = false;
  
  // ═══════════════════════════════════════════════════════════════════════════
  // OPTIMIZATION: Cached MIGraphX API results (avoid redundant API calls)
  // ═══════════════════════════════════════════════════════════════════════════
  
  // Cached program parameter shapes (from prog.get_parameter_shapes())
  std::optional<migraphx::program_parameter_shapes> cached_mgx_param_shapes;
  
  // Cached output shapes (from prog.get_output_shapes())
  std::optional<migraphx::shapes> cached_mgx_output_shapes;
  
  // Flag indicating ultra-fast caches are populated (avoid redundant populate calls)
  bool ultra_fast_caches_populated = false;
  
  // Track which program hash the cached shapes belong to (invalidate when program changes)
  std::string cached_program_hash;

  // ═══════════════════════════════════════════════════════════════════════════
  // hipGraph CAPTURE / REPLAY
  // ═══════════════════════════════════════════════════════════════════════════

  struct ExtraOutputInfo {
    std::size_t output_index;
    std::vector<int64_t> ort_shape;
    void* gpu_data;
    std::size_t bytes;
  };

  struct CapturedHipGraph {
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    bool captured = false;
    std::vector<ExtraOutputInfo> extra_outputs;

    // Addresses captured in the graph for direct-bind mode.
    // Used to detect pointer drift and trigger re-capture.
    std::unordered_map<std::string, void*> captured_input_ptrs;
    std::unordered_map<std::string, void*> captured_output_ptrs;

    // Scratch buffer pointer baked into the captured graph.  Compared against
    // the current scratch buffer pointer on every replay; a mismatch (e.g.
    // because the buffer was reallocated for a shape-size change) forces a
    // re-capture.  nullptr means the program has no "scratch" parameter and
    // no EP-owned scratch was bound.
    void* captured_scratch_ptr = nullptr;

    // Output buffers (ptr + byte size) we need to memset to zero before every
    // replay.  Required because some captured kernels do read-modify-write on
    // their output (split-K reductions, fused-attention accumulators, etc.).
    // Without this the first replay after a batch-size transition inherits
    // residue from the previously-recycled ORT-pool buffer and produces a
    // small but real numerical drift relative to eager (observed on the
    // larger-reduction outputs 2/7/11 of feed-gen-rec).  Populated at capture
    // time from output_ptrs + param_shapes; size is the program-side bytes
    // (not the original-batch slice), which is what the captured kernels
    // actually touch.  "Extra" outputs (those returned by prog.run_async
    // rather than pre-allocated) are intentionally excluded -- they live in
    // MIGraphX-managed memory and are materialized via a fresh memcpy after
    // every replay.
    std::vector<std::pair<void*, std::size_t>> captured_output_zeroes;
  };

  bool hip_graph_enabled = false;
  // When true, capture/replay binds ORT tensor pointers directly (no pinned copies).
  // Requires the pool allocator to provide stable addresses.
  bool use_direct_hip_graph = false;
  // If pointer drift causes too many re-captures, disable direct mode permanently.
  static constexpr int kMaxDirectRecaptures = 3;
  int direct_recapture_count = 0;
  // shape_hash -> captured graph (one per compiled program variant)
  std::unordered_map<std::string, CapturedHipGraph> hip_graph_cache;
};

// Logical device representation.
class MIGraphXExecutionProvider : public IExecutionProvider {
 public:
  explicit MIGraphXExecutionProvider(const MIGraphXExecutionProviderInfo& info);
  ~MIGraphXExecutionProvider() override {
    if (!external_stream_ && stream_) {
      (void)hipStreamDestroy(stream_);
    }
  }

  Status Sync() const override;

  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;

  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;

  void dump_model_as_onnx(const std::string& onnx_buffer,
                          const std::string& model_name) const;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/,
                const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                IResourceAccountant* /* resource_accountant */) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::unique_ptr<IndexedSubGraph> GetSubGraph(const std::vector<std::size_t>& graph_nodes_index, const GraphViewer& graph, bool is_graph_split) const;
  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const override;
  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  int GetDeviceId() const override { return device_id_; }
  ProviderOptions GetProviderOptions() const override {
    return {
        {std::string{migraphx_provider_option::kDeviceId}, MakeStringWithClassicLocale(device_id_)},
        {std::string{migraphx_provider_option::kCompileTarget}, target_device_},
        {std::string{migraphx_provider_option::kFp16Enable}, MakeStringWithClassicLocale(fp16_enable_)},
        {std::string{migraphx_provider_option::kBf16Enable}, MakeStringWithClassicLocale(bf16_enable_)},
        {std::string{migraphx_provider_option::kFp8Enable}, MakeStringWithClassicLocale(fp8_enable_)},
        {std::string{migraphx_provider_option::kInt8Enable}, MakeStringWithClassicLocale(int8_enable_)},
        {std::string{migraphx_provider_option::kInt8CalibTable}, MakeStringWithClassicLocale(int8_calibration_table_name_)},
        {std::string{migraphx_provider_option::kInt8UseNativeCalibTable}, MakeStringWithClassicLocale(int8_use_native_calibration_table_)},
        {std::string{migraphx_provider_option::kExhaustiveTune}, MakeStringWithClassicLocale(exhaustive_tune_)},
        {std::string{migraphx_provider_option::kMemLimit}, MakeStringWithClassicLocale(mem_limit_)},
        {std::string{migraphx_provider_option::kArenaExtendStrategy}, EnumToName(arena_extend_strategy_mapping, arena_extend_strategy_)},
        {std::string{migraphx_provider_option::kGpuExternalAlloc}, MakeStringWithClassicLocale(external_alloc_)},
        {std::string{migraphx_provider_option::kGpuExternalFree}, MakeStringWithClassicLocale(external_free_)},
        {std::string{migraphx_provider_option::kGpuExternalEmptyCache}, MakeStringWithClassicLocale(external_empty_cache_)},
        {std::string{migraphx_provider_option::kModelCacheDir}, MakeStringWithClassicLocale(model_cache_path_)},
        {std::string{migraphx_provider_option::kModelMaxDynamicBatch}, MakeStringWithClassicLocale(max_dynamic_batch_)},
        {std::string{migraphx_provider_option::kCompileBatches}, compile_batches_},
        {std::string{migraphx_provider_option::kHipGraphEnable}, MakeStringWithClassicLocale(hip_graph_enable_)},
        {std::string{migraphx_provider_option::kHasUserComputeStream}, MakeStringWithClassicLocale(external_stream_)},
        {std::string{migraphx_provider_option::kUserComputeStream}, MakeStringWithClassicLocale(reinterpret_cast<size_t>(stream_))}};
   }

 private:
  OrtDevice::DeviceId device_id_{0};
  // MIGraphX compile target: "gpu" (default), "ref", "cpu", or "mps".
  std::string target_device_{"gpu"};
  bool fp16_enable_ = false;
  bool bf16_enable_ = false;
  bool fp8_enable_ = false;
  bool int8_enable_ = false;
  std::string int8_calibration_table_name_;
  bool int8_calibration_cache_available_ = false;
  bool int8_use_native_calibration_table_ = false;
  std::filesystem::path calibration_cache_path_{};
  std::unordered_map<std::string, float> dynamic_range_map_;
  std::filesystem::path model_cache_path_{};
  // Map of model input names per node (excludes weights/constants)
  std::unordered_map<std::string, std::set<std::string>> map_session_input_names_;
  bool dump_model_ops_ = false;
  migraphx::target t_;
  std::mutex mgx_mu_;
  bool external_stream_ = false;
  hipStream_t stream_ = nullptr;
  hipDeviceProp_t device_prop_{};
  bool exhaustive_tune_ = false;
  mutable std::filesystem::path model_path_{};
  size_t mem_limit_{std::numeric_limits<size_t>::max()};
  ArenaExtendStrategy arena_extend_strategy_{ArenaExtendStrategy::kNextPowerOfTwo};

  std::unordered_map<std::string, migraphx::program> map_progs_;
  std::unordered_map<std::string, std::string> map_onnx_string_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::size_t>> map_input_index_;
  std::unordered_map<std::string, bool> map_defer_compilation_;
  // Map of cached programs per node: node_name -> (input_shape_hash -> program)
  std::unordered_map<std::string, std::unordered_map<std::string, migraphx::program>> cached_programs_;

  AllocatorPtr allocator_;
  std::unique_ptr<ModelMetadefIdGenerator> metadef_id_generator_;
  void* external_alloc_{nullptr};
  void* external_free_{nullptr};
  void* external_empty_cache_{nullptr};
  bool first_start_ = true;
  size_t max_dynamic_batch_{0};
  std::string compile_batches_{};  // Comma-separated list of batch sizes to compile, e.g. "1,4,8,16,32"
  bool hip_graph_enable_{false};
};

}; // namespace onnxruntime
