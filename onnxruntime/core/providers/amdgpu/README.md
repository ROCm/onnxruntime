# AMD GPU Plugin Execution Provider

This is a skeleton implementation of an AMD GPU plugin execution provider for ONNX Runtime.

## Overview

This plugin EP provides the foundation for AMD GPU acceleration using ROCm/HIP. The current implementation is a skeleton that implements all required interfaces but does not yet include actual GPU kernel execution.

## Architecture

The plugin follows the ONNX Runtime plugin EP architecture as documented in `docs/plugin-ep-libraries.md`:

### Core Components

1. **amd_gpu_plugin_ep.cc** - Entry point
   - Exports `CreateEpFactories()` and `ReleaseEpFactory()` functions
   - Initializes the ONNX Runtime C++ API

2. **AmdGpuEpFactory** (ep_factory.h/cc) - Factory class
   - Implements `OrtEpFactory` interface
   - Vendor: AMD (PCI ID: 0x1002)
   - Version: 0.1.0
   - Creates EP instances and manages shared resources

3. **AmdGpuEp** (ep.h/cc) - Execution provider class
   - Implements `OrtEp` interface
   - Handles graph capability queries
   - Compiles graphs for execution (skeleton)

4. **AmdGpuDataTransfer** (ep_data_transfer.h/cc)
   - Implements `OrtDataTransferImpl` interface
   - Handles CPU ↔ GPU memory transfers (currently uses memcpy)

5. **AmdGpuAllocator** (ep_allocator.h)
   - Memory allocator for GPU memory (currently uses CPU malloc/free)

6. **AmdGpuStreamImpl** (ep_stream_support.h/cc)
   - Implements `OrtSyncStreamImpl` interface
   - Handles stream synchronization (skeleton)

### Symbol Export Files

- **amd_gpu_plugin_ep_library.def** - Windows symbol exports
- **amd_gpu_plugin_ep_library.lds** - Linux symbol exports

## Current Status

This is a **skeleton implementation** with the following characteristics:

### Implemented
- ✅ All required OrtEpFactory interface methods
- ✅ All required OrtEp interface methods
- ✅ Data transfer interface (using CPU memcpy)
- ✅ Allocator interface (using CPU malloc/free)
- ✅ Stream synchronization interface
- ✅ Proper error handling and logging
- ✅ Cross-platform symbol exports

### Not Yet Implemented
- ❌ Actual GPU kernel execution
- ❌ ROCm/HIP integration
- ❌ GPU memory allocation (hipMalloc/hipFree)
- ❌ GPU memory transfers (hipMemcpy)
- ❌ Stream synchronization (hipStreamSynchronize)
- ❌ Node capability detection (GetCapability returns no supported nodes)
- ❌ Graph compilation for GPU execution

## Future Development

To make this a fully functional AMD GPU EP, the following needs to be added:

1. **ROCm/HIP Integration**
   - Add HIP headers and link against ROCm libraries
   - Implement GPU memory allocation using hipMalloc/hipFree
   - Implement memory transfers using hipMemcpy
   - Implement stream synchronization using hipStreamSynchronize

2. **Kernel Implementation**
   - Identify which ONNX operators to accelerate
   - Implement HIP kernels for supported operators
   - Update GetCapability to report supported nodes
   - Implement actual computation in Compile method

3. **Performance Optimization**
   - Add arena allocator for GPU memory
   - Implement kernel fusion
   - Optimize memory transfers
   - Add async execution support

4. **Testing**
   - Add unit tests
   - Add integration tests
   - Add performance benchmarks

## Building

This plugin EP can be built as part of the ONNX Runtime build system. The build configuration would need to be added to the appropriate CMakeLists.txt file.

## Usage

Once built, the plugin can be loaded and used with ONNX Runtime:

```cpp
// Register the plugin EP library
OrtStatus* status = ort_api->RegisterExecutionProviderLibrary(
    env,
    "amd_gpu_plugin_ep",  // registration name
    "/path/to/amd_gpu_plugin_ep.dll"  // library path
);

// The EP will be available for use in inference sessions
```

## References

- Plugin EP documentation: `docs/plugin-ep-libraries.md`
- Example plugin EP: `onnxruntime/test/autoep/library/example_plugin_ep/`
- ROCm documentation: https://rocm.docs.amd.com/
- HIP programming guide: https://rocm.docs.amd.com/projects/HIP/
