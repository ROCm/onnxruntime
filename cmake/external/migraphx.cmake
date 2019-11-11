# Building from source is currently unsupported.
# Please follow instructions in the documentation to install MIGRAPHX binaries.

include (ExternalProject)

if(onnxruntime_USE_MIGRAPHX)
    set(migraphx_LIBRARIES ${AMD_MIGRAPHX_BUILD}/lib)
    set(MIGRAPHX_SHARED_LIB libmigraphx.so.0 libmigraphx_cpu.so.0 libmigraphx_gpu.so.0 libmigraphx_device.so.0
            libmigraphx_tf.so.0 libmigraphx_onnx.so.0)
    set(miopen_LIBRARIES ${AMD_MIGRAPHX_DEPS}/lib)
    set(MIOPEN_SHARED_LIB libMIOpen.so)
endif()
