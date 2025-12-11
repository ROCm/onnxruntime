# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_compile_definitions(USE_AMDGPU=1)
  file(GLOB_RECURSE onnxruntime_providers_amdgpu_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/amdgpu/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/amdgpu/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )
  onnxruntime_add_shared_library(onnxruntime_providers_amdgpu ${onnxruntime_providers_amdgpu_cc_srcs})
  add_dependencies(onnxruntime_providers_amdgpu ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_amdgpu PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_amdgpu PROPERTIES FOLDER "ONNXRuntime")
  target_compile_definitions(onnxruntime_providers_amdgpu PRIVATE ONNX_ML=1 ONNX_NAMESPACE=onnx)
  if(MSVC)
    set_property(TARGET onnxruntime_providers_amdgpu APPEND_STRING PROPERTY LINK_FLAGS /DEF:${ONNXRUNTIME_ROOT}/core/providers/amdgpu/symbols.def)
  else()
    set_property(TARGET onnxruntime_providers_amdgpu APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/amdgpu/version_script.lds -Xlinker --gc-sections")
  endif()

  install(TARGETS onnxruntime_providers_amdgpu
          EXPORT onnxruntime_providers_amdgpuTargets
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
