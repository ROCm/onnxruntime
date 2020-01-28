# Building from source is currently unsupported.
# Please follow instructions in the documentation to install MIGRAPHX binaries.

include (ExternalProject)

if(onnxruntime_USE_MIGRAPHX)
    set(migraphx_LIBRARIES ${AMD_MIGRAPHX_BUILD}/lib)
    set(MIGRAPHX_SHARED_LIB libmigraphx_c.so.0)
endif()
