#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# For live logging, use the command: pytest -o log_cli=true --log-cli-level=DEBUG

import os
import unittest

import pytest
from test_optimizer import _get_test_model_path


class TestBertProfiler(unittest.TestCase):
    def setUp(self):
        from onnxruntime import get_available_providers  # noqa: PLC0415

        self.test_cuda = "CUDAExecutionProvider" in get_available_providers()

    def run_profile(self, arguments: str):
        from onnxruntime.transformers.profiler import parse_arguments, run  # noqa: PLC0415

        args = parse_arguments(arguments.split())
        results = run(args)
        self.assertTrue(len(results) > 1)

    @pytest.mark.slow
    def test_profiler_gpu(self):
        input_model_path = _get_test_model_path("bert_keras_squad")
        if self.test_cuda:
            self.run_profile(f"--model {input_model_path} --batch_size 1 --sequence_length 7 --use_gpu")

    @pytest.mark.slow
    def test_profiler_cpu(self):
        input_model_path = _get_test_model_path("bert_keras_squad")
        self.run_profile(f"--model {input_model_path} --batch_size 1 --sequence_length 7 --dummy_inputs default")


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    unittest.main()
