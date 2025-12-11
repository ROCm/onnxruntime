# Plugin Execution Provider Libraries

## Background
An ONNX Runtime Execution Provider (EP) executes model operations on one or more hardware accelerators (e.g., GPU, NPU, etc.). ONNX Runtime provides a variety of built-in EPs, such as the default CPU EP. To enable further extensibility, ONNX Runtime supports user-defined plugin EP libraries that an application can register with ONNX Runtime for use in an ONNX Runtime inference session.

This page provides a reference for the APIs necessary to develop and use plugin EP libraries with ONNX Runtime.

## Creating a plugin EP library
A plugin EP is built as a dynamic/shared library that exports the functions `CreateEpFactories()` and `ReleaseEpFactory()`. ONNX Runtime calls `CreateEpFactories()` to obtain one or more instances of `OrtEpFactory`. An `OrtEpFactory` creates `OrtEp` instances and specifies the hardware devices supported by the EPs it creates.

The ONNX Runtime repository includes a sample plugin EP library, which is referenced in the following sections.

### Defining an OrtEp
An `OrtEp` represents an instance of an EP that is used by an ONNX Runtime session to identify and execute the model operations supported by the EP.

The following table lists the **required** variables and functions that an implementer must define for an `OrtEp`.

| Field | Summary | Example implementation |
|-------|---------|------------------------|
| `ort_version_supported` | The ONNX Runtime version with which the EP was compiled. Implementation should set to `ORT_API_VERSION`. | `ExampleEp()` |
| `GetName`	| Get the execution provider name. | `ExampleEp::GetNameImpl()` |
| `GetCapability` | Get information about the nodes/subgraphs supported by the OrtEp instance. | `ExampleEp::GetCapabilityImpl()` |
| `Compile` | Compile OrtGraph instances assigned to the `OrtEp`. Implementation must set a `OrtNodeComputeInfo` instance for each OrtGraph in order to define its computation function.<br><br>If the session is configured to generate a pre-compiled model, the execution provider must return count number of EPContext nodes. | `ExampleEp::CompileImpl()` |
|`ReleaseNodeComputeInfos` | Release OrtNodeComputeInfo instances. | `ExampleEp::ReleaseNodeComputeInfosImpl()` |

The following table lists the optional functions that an implementor may define for an `OrtEp`. If an optional `OrtEp` function is not defined, ONNX Runtime uses a default implementation.

| Field | Summary |
|-------|---------|
| `GetPreferredDataLayout` | Get the EP's preferred data layout.<br><br>If this function is not implemented, ORT assumes that the EP prefers the data layout `OrtEpDataLayout::NCHW`. |
| `ShouldConvertDataLayoutForOp` | Given an op with domain `domain` and type `op_type`, determine whether an associated node's data layout should be converted to a `target_data_layout`. If the EP prefers a non-default data layout, this function will be called during layout transformation with `target_data_layout` set to the EP's preferred data layout<br><br>Implementation of this function is optional. If an EP prefers a non-default data layout, it may implement this to customize the specific op data layout preferences at a finer granularity.	|
| `SetDynamicOptions` | Set dynamic options on this EP. Dynamic options can be set by the application at any time after session creation with `OrtApi::SetEpDynamicOptions()`.<br><br>Implementation of this function is optional. An EP should only implement this function if it needs to handle any dynamic options. |
| `OnRunStart` | Called by ORT to notify the EP of the start of a run.<br><br>Implementation of this function is optional. An EP should only implement this function if it needs to handle application-provided options at the start of a run. |
| `OnRunEnd` | Called by ORT to notify the EP of the end of a run.<br><br>Implementation of this function is optional. An EP should only implement this function if it needs to handle application-provided options at the end of a run. |
| `CreateAllocator` | Create an `OrtAllocator` for the given `OrtMemoryInfo` for an `OrtSession`.<br><br>The `OrtMemoryInfo` instance will match one of the values set in the `OrtEpDevice` using `EpDevice_AddAllocatorInfo`. Any allocator specific options should be read from the session options.<br><br>Implementation of this function is optional. If not provided, ORT will use `OrtEpFactory::CreateAllocator()`.	|
|`CreateSyncStreamForDevice` | Create a synchronization stream for the given memory device for an `OrtSession`.<br><br>This is used to create a synchronization stream for the execution provider and is used to synchronize operations on the device during model execution. Any stream specific options should be read from the session options.<br><br>Implementation of this function is optional. If not provided, ORT will use `OrtEpFactory::CreateSyncStreamForDevice()`.	|
| `GetCompiledModelCompatibilityInfo` | Get a string with details about the EP stack used to produce a compiled model.<br><br>The compatibility information string can be used with `OrtEpFactory::ValidateCompiledModelCompatibilityInfo` to determine if a compiled model is compatible with the EP. |

### Defining an OrtEpFactory
An OrtEpFactory represents an instance of an EP factory that is used by an ONNX Runtime session to query device support, create allocators, create data transfer objects, and create instances of an EP (i.e., an OrtEp).

The following table lists the required variables and functions that an implementer must define for an OrtEpFactory.

| Field | Summary | Example implementation |
|-------|---------|------------------------|
| `ort_version_supported` | The ONNX Runtime version with which the EP was compiled. Implementation should set this to `ORT_API_VERSION`. \ `ExampleEpFactory()` |
| `GetName` | Get the name of the EP that the factory creates. Must match `OrtEp::GetName()`. | `ExampleEpFactory::GetNameImpl()` |
| `GetVendor` | Get the name of the name of the vendor that owns the EP that the factory creates. | `ExampleEpFactory::GetVendor()` |
| `GetVendorId` | Get the vendor ID of the vendor that owns the EP that the factory creates. This is typically the PCI vendor ID.| `ExampleEpFactory::GetVendorId()` |
| `GetVersion` | Get the version of the EP that the factory creates. The version string should adhere to the Semantic Versioning 2.0 specification. | `ExampleEpFactory::GetVersionImpl()` |
| `GetSupportedDevices` | Get information about the OrtHardwareDevice instances supported by an EP created by the factory.	| `ExampleEpFactory::GetSupportedDevicesImpl()` |
| `CreateEp` | Creates an OrtEp instance for use in an ONNX Runtime session. ORT calls OrtEpFactory::ReleaseEp() to release the instance. | `ExampleEpFactory::CreateEpImpl()` |

The following table lists the optional functions that an implementer may define for an OrtEpFactory.

| Field | Summary | Example implementation |
|-------|---------|------------------------|
| `ValidateCompiledModelCompatibilityInfo` | Validate the compatibility of a compiled model with the EP.<br><br>This function validates if a model produced with the supllied compatibility information string is supported by the underlying EP. The implementation should check if a compiled model is compatible with the EP and return the appropriate `OrtCompiledModelCompatibility` value.
CreateAllocator	Create an OrtAllocator that can be shared across sessions for the given `OrtMemoryInfo`.<br><br>The factory that creates the EP is responsible for providing the allocators required by the EP. The `OrtMemoryInfo` instance will match one of the values set in the `OrtEpDevice` using `EpDevice_AddAllocatorInfo`. | `ExampleEpFactory::CreateAllocatorImpl()` |
| `ReleaseAllocator` | Releases an OrtAllocator instance created by the factory. | `ExampleEpFactory::ReleaseAllocatorImpl()` |
| `CreateDataTransfer` | Creates an OrtDataTransferImpl instance for the factory.<br><br>An `OrtDataTransferImpl` can be used to copy data between devices that the EP supports. | `ExampleEpFactory::CreateDataTransferImpl()` |
| `IsStreamAware` | Returns true if the EPs created by the factory are stream-aware. | `ExampleEpFactory::IsStreamAwareImpl()` |
| `CreateSyncStreamForDevice` | Creates a synchronization stream for the given `OrtMemoryDevice`.<br><br>This is use to create a synchronization stream for the OrtMemoryDevice that can be used for operations outside of a session. |`ExampleEpFactory::CreateSyncStreamForDeviceImpl()` |

### Exporting functions to create and release factories
ONNX Runtime expects a plugin EP library to export certain functions/symbols. The following table lists the functions that have to be exported from the plugin EP library.

| Function	| Description |
|-----------|-------------|
| CreateEpFactories	| ONNX Runtime calls this function to create OrtEpFactory instances. |
| ReleaseEpFactory |ONNX Runtime calls this function to release an OrtEpFactory instance. |
