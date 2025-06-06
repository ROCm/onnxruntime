// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include <numeric>

namespace onnxruntime {
namespace test {

// SequenceLength
TEST(SequenceOpsTest, SequenceLengthPositiveFloat) {
  OpTester test("SequenceLength", 11);
  SeqTensors<float> input;
  input.AddTensor({3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  input.AddTensor({3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddSeqInput("S", input);
  test.AddOutput<int64_t>("I", {}, {2});
  test.Run();
}

TEST(SequenceOpsTest, SequenceLengthPositiveInt64) {
  OpTester test("SequenceLength", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddOutput<int64_t>("I", {}, {2});
  test.Run();
}

// SequenceAt
TEST(SequenceOpsTest, SequenceAtPositiveIdx) {
  OpTester test("SequenceAt", 11);
  SeqTensors<float> input;
  std::vector<float> output_vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> output_shape{3, 3};
  input.AddTensor({3, 2}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  input.AddTensor(output_shape, output_vec);
  test.AddSeqInput("S", input);
  test.AddInput("I", {}, {1});
  test.AddOutput<float>("T", output_shape, output_vec);
  test.Run();
}

TEST(SequenceOpsTest, SequenceAtNegativeIdx) {
  OpTester test("SequenceAt", 11);
  SeqTensors<float> input;
  std::vector<float> output_vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> output_shape{3, 3};
  input.AddTensor({3, 2}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  input.AddTensor(output_shape, output_vec);
  test.AddSeqInput("S", input);
  test.AddInput("I", {}, {-1});
  test.AddOutput<float>("T", output_shape, output_vec);
  test.Run();
}

TEST(SequenceOpsTest, SequenceAtInvalidPositiveIdx) {
  OpTester test("SequenceAt", 11);
  SeqTensors<float> input;
  std::vector<float> output_vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> output_shape{3, 3};
  input.AddTensor({3, 2}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  input.AddTensor(output_shape, output_vec);
  test.AddSeqInput("S", input);
  test.AddInput("I", {}, {10});
  test.AddOutput<float>("T", output_shape, output_vec);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

TEST(SequenceOpsTest, SequenceAtInvalidNegativeIdx) {
  OpTester test("SequenceAt", 11);
  SeqTensors<float> input;
  std::vector<float> output_vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> output_shape{3, 3};
  input.AddTensor({3, 2}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  input.AddTensor(output_shape, output_vec);
  test.AddSeqInput("S", input);
  test.AddInput("I", {}, {-10});
  test.AddOutput<float>("T", output_shape, output_vec);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

// SequenceEmpty
TEST(SequenceOpsTest, SequenceEmptyDefault) {
  OpTester test("SequenceEmpty", 11);
  test.AddSeqOutput("S", SeqTensors<float>());
  test.Run();
}

TEST(SequenceOpsTest, SequenceEmptyInt64) {
  OpTester test("SequenceEmpty", 11);
  test.AddAttribute("dtype", static_cast<int64_t>(7));
  test.AddSeqOutput("S", SeqTensors<int64_t>());
  test.Run();
}

// SequenceInsert
TEST(SequenceOpsTest, SequenceInsertPositiveDefaultFloat) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<float> input;
  input.AddTensor({3, 2}, {1., 2., 3., 4., 5., 6.});
  input.AddTensor({3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  test.AddSeqInput("S", input);
  test.AddInput<float>("T", {3, 2}, {10., 20., 30., 40., 50., 60.});

  SeqTensors<float> output;
  output.AddTensor({3, 2}, {1., 2., 3., 4., 5., 6.});
  output.AddTensor({3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  output.AddTensor({3, 2}, {10., 20., 30., 40., 50., 60.});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceInsertPositiveDefault) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceInsertEmptyLast) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddInput<int64_t>("I", {1}, {0});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceInsertLast) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddInput<int64_t>("I", {1}, {2});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceInsertValidPositiveIdx) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddInput<int64_t>("I", {}, {1});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceInsertValidNegativeIdx) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddInput<int64_t>("I", {}, {-2});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceInsertInvalidPositiveIdx) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddInput<int64_t>("I", {}, {99});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqOutput("S2", output);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

TEST(SequenceOpsTest, SequenceInsertInvalidNegativeIdx) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddInput<int64_t>("I", {}, {-99});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqOutput("S2", output);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

// SequenceErase
TEST(SequenceOpsTest, SequenceErasePositiveDefault) {
  OpTester test("SequenceErase", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqInput("S", input);

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceEraseValidPositiveIdx) {
  OpTester test("SequenceErase", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("I", {}, {1});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceEraseValidNegativeIdx) {
  OpTester test("SequenceErase", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  input.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("I", {}, {-2});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  output.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceEraseInvalidPositiveIdx) {
  OpTester test("SequenceErase", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  input.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("I", {}, {99});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqOutput("S2", output);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

TEST(SequenceOpsTest, SequenceEraseInvalidNegativeIdx) {
  OpTester test("SequenceErase", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  input.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("I", {}, {-99});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqOutput("S2", output);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

// SequenceConstruct
TEST(SequenceOpsTest, SequenceConstructPositive) {
  OpTester test("SequenceConstruct", 11);
  test.AddInput<int64_t>("input_1", {3, 2}, {1, 2, 3, 4, 5, 6});
  test.AddInput<int64_t>("input_2", {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddInput<int64_t>("input_3", {3, 2}, {10, 20, 30, 40, 50, 60});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqOutput("S2", output);
  test.Run();
}

// SplitToSequence
template <typename T>
static std::vector<T> GetConsecutiveVector(T start, size_t num) {
  std::vector<T> inputv(num);
  std::iota(inputv.begin(), inputv.end(), start);
  return inputv;
}

template <>
std::vector<MLFloat16> GetConsecutiveVector<MLFloat16>(MLFloat16 start, size_t num) {
  std::vector<MLFloat16> inputv;
  inputv.reserve(num);
  float start_f = start.ToFloat();
  for (size_t i = 0; i < num; ++i) {
    inputv.push_back(MLFloat16{start_f + static_cast<float>(i)});
  }
  return inputv;
}

TEST(SequenceOpsTest, SplitToSequence_DefaultAxis0EqualSplitFloat) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<float>("input", {4, 2}, GetConsecutiveVector<float>(1.f, 8));
  test.AddInput<int64_t>("split", {1, 2}, {2, 2});
  SeqTensors<float> output;
  output.AddTensor({2, 2}, {1.f, 2.f, 3.f, 4.f});
  output.AddTensor({2, 2}, {5.f, 6.f, 7.f, 8.f});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_DefaultAxis0EqualSplitMLFloat16) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<MLFloat16>("input", {4, 2}, GetConsecutiveVector<MLFloat16>(MLFloat16::One, 8));
  test.AddInput<int64_t>("split", {1, 2}, {2, 2});
  SeqTensors<MLFloat16> output;

  std::vector<MLFloat16> tensor_1;
  const auto data_1 = {1.f, 2.f, 3.f, 4.f};
  for (auto f : data_1)
    tensor_1.push_back(MLFloat16{f});

  std::vector<MLFloat16> tensor_2;
  const auto data_2 = {5.f, 6.f, 7.f, 8.f};
  for (auto f : data_2)
    tensor_2.push_back(MLFloat16{f});

  output.AddTensor({2, 2}, tensor_1);
  output.AddTensor({2, 2}, tensor_2);
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_DefaultAxis0EqualSplitLong) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<int64_t>("input", {4, 2}, GetConsecutiveVector<int64_t>(1, 8));
  test.AddInput<int64_t>("split", {1, 2}, {2, 2});
  SeqTensors<int64_t> output;
  output.AddTensor({2, 2}, {1, 2, 3, 4});
  output.AddTensor({2, 2}, {5, 6, 7, 8});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_DefaultAxis0EqualSplitFloatScalarSplit) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<float>("input", {4, 2}, GetConsecutiveVector<float>(1.f, 8));
  test.AddInput<int64_t>("split", {}, {2});
  SeqTensors<float> output;
  output.AddTensor({2, 2}, {1.f, 2.f, 3.f, 4.f});
  output.AddTensor({2, 2}, {5.f, 6.f, 7.f, 8.f});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_Axis0DefaultSplitFloatSetAxisExplicitly) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<float>("input", {4, 2}, GetConsecutiveVector<float>(1.f, 8));
  int64_t axis = 0;
  test.AddAttribute("axis", axis);
  SeqTensors<float> output;
  output.AddTensor({1, 2}, {1.f, 2.f});
  output.AddTensor({1, 2}, {3.f, 4.f});
  output.AddTensor({1, 2}, {5.f, 6.f});
  output.AddTensor({1, 2}, {7.f, 8.f});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_PositiveAxisScalarSplit) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<float>("input", {2, 2, 6}, GetConsecutiveVector<float>(1.f, 2 * 2 * 6));
  int64_t axis = 2;
  test.AddAttribute("axis", axis);
  test.AddInput<int64_t>("split", {}, {2});
  SeqTensors<float> output;
  output.AddTensor({2, 2, 2}, {1.f, 2.f,
                               7.f, 8.f,

                               13.f, 14.f,
                               19.f, 20.f});
  output.AddTensor({2, 2, 2}, {3.f, 4.f,
                               9.f, 10.f,

                               15.f, 16.f,
                               21.f, 22.f});
  output.AddTensor({2, 2, 2}, {5.f, 6.f,
                               11.f, 12.f,

                               17.f, 18.f,
                               23.f, 24.f});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_StringSplit) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<std::string>("input", {3}, std::vector<std::string>({"Test string", "Another string", "A third and much longer string"}));
  int64_t axis = 0;
  test.AddAttribute("axis", axis);
  SeqTensors<std::string> output;
  output.AddTensor({1}, {"Test string"});
  output.AddTensor({1}, {"Another string"});
  output.AddTensor({1}, {"A third and much longer string"});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_DefaultAxis0UnevenSplitFloat) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<float>("input", {5, 2}, GetConsecutiveVector<float>(1.f, 10));
  test.AddInput<int64_t>("split", {}, {2});
  SeqTensors<float> output;
  output.AddTensor({2, 2}, GetConsecutiveVector<float>(1.f, 4));
  output.AddTensor({2, 2}, GetConsecutiveVector<float>(5.f, 4));
  output.AddTensor({1, 2}, {9.f, 10.f});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_DefaultAxis0UnevenSplitFloat2) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<float>("input", {17, 2}, GetConsecutiveVector<float>(1.f, 34));
  test.AddInput<int64_t>("split", {}, {3});
  SeqTensors<float> output;
  output.AddTensor({3, 2}, GetConsecutiveVector<float>(1.f, 6));
  output.AddTensor({3, 2}, GetConsecutiveVector<float>(7.f, 6));
  output.AddTensor({3, 2}, GetConsecutiveVector<float>(13.f, 6));
  output.AddTensor({3, 2}, GetConsecutiveVector<float>(19.f, 6));
  output.AddTensor({3, 2}, GetConsecutiveVector<float>(25.f, 6));
  output.AddTensor({2, 2}, GetConsecutiveVector<float>(31.f, 4));
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_PositiveAxisUnevenSplit) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<float>("input", {2, 5}, GetConsecutiveVector<float>(1.f, 10));
  test.AddInput<int64_t>("split", {}, {2});
  int64_t axis = 1;
  test.AddAttribute("axis", axis);
  SeqTensors<float> output;
  output.AddTensor({2, 2}, {1.f, 2.f, 6.f, 7.f});
  output.AddTensor({2, 2}, {3.f, 4.f, 8.f, 9.f});
  output.AddTensor({2, 1}, {5.f, 10.f});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_Axis0DefaultSplitFloatSetAxisExplicitlyDontKeepDims3Dim) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<float>("input", {2, 3, 4}, GetConsecutiveVector<float>(1.f, 2 * 3 * 4));
  test.AddAttribute<int64_t>("keepdims", 0);
  int64_t axis = 0;
  test.AddAttribute("axis", axis);
  SeqTensors<float> output;
  output.AddTensor({3, 4}, GetConsecutiveVector<float>(1.f, 12));
  output.AddTensor({3, 4}, GetConsecutiveVector<float>(13.f, 12));
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_Axis0DefaultSplitFloatSetAxisExplicitlyDontKeepDims2Dim) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<float>("input", {2, 3}, GetConsecutiveVector<float>(1.f, 2 * 3));
  test.AddAttribute<int64_t>("keepdims", 0);
  int64_t axis = 0;
  test.AddAttribute("axis", axis);
  SeqTensors<float> output;
  output.AddTensor({3}, GetConsecutiveVector<float>(1.f, 3));
  output.AddTensor({3}, GetConsecutiveVector<float>(4.f, 3));
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_PositiveAxisDontKeepDims) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<float>("input", {2, 3, 4}, GetConsecutiveVector<float>(1.f, 2 * 3 * 4));
  test.AddAttribute<int64_t>("keepdims", 0);
  int64_t axis = 2;
  test.AddAttribute("axis", axis);
  SeqTensors<float> output;
  output.AddTensor({2, 3}, {1.f, 5.f, 9.f, 13.f, 17.f, 21.f});
  output.AddTensor({2, 3}, {2.f, 6.f, 10.f, 14.f, 18.f, 22.f});
  output.AddTensor({2, 3}, {3.f, 7.f, 11.f, 15.f, 19.f, 23.f});
  output.AddTensor({2, 3}, {4.f, 8.f, 12.f, 16.f, 20.f, 24.f});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SplitToSequence_BoolSplit) {
  OpTester test("SplitToSequence", 11);
  test.AddInput<bool>("input", {4, 2}, std::initializer_list<bool>({1, 1, 1, 1, 0, 0, 0, 0}));
  int64_t axis = 0;
  test.AddAttribute("axis", axis);
  SeqTensors<bool> output;
  output.AddTensor({1, 2}, {1, 1});
  output.AddTensor({1, 2}, {1, 1});
  output.AddTensor({1, 2}, {0, 0});
  output.AddTensor({1, 2}, {0, 0});
  test.AddSeqOutput("S2", output);
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
