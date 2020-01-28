// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/environment.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"

std::unique_ptr<Ort::Env> ort_env;

GTEST_API_ int main(int argc, char** argv) {  
  int status = 0;
  try {
    ::testing::InitGoogleTest(&argc, argv);
    ort_env.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Default"));
    status = RUN_ALL_TESTS();
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    status = -1;
  }
  //TODO: Fix the C API issue
  ort_env.reset(); //If we don't do this, it will crash
  return status;
}
