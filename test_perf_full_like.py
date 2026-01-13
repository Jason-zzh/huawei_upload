#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" full_like op performance test case """
# pylint: disable=unused-variable
# pylint: disable=W0622,W0613
import time
import numpy as np
import mindspore as ms
from mindspore import mint
from mindspore.common.api import _pynative_executor
from tests.utils.test_op_utils import BACKGROUND_NOISE
from tests.utils.mark_utils import arg_mark
import torch
import pytest


def full_like_forward_perf(input_tensor, fill_value):
    """Get MindSpore full_like forward performance."""
    # Warm-up
    for _ in range(1000):
        _ = mint.full_like(input_tensor, fill_value)

    _pynative_executor.sync()
    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = mint.full_like(input_tensor, fill_value)
    _pynative_executor.sync()
    end_time = time.time()

    return end_time - start_time


def generate_expect_full_like_forward_perf(input_tensor, fill_value):
    """Get PyTorch full_like forward performance."""
    # Warm-up
    for _ in range(1000):
        _ = torch.full_like(input_tensor, fill_value)

    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = torch.full_like(input_tensor, fill_value)
    end_time = time.time()

    return end_time - start_time


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_full_like_perf_large_tensor(mode):
    """
    Feature: standard forward performance for full_like with large tensor.
    Description: test full_like op performance.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Create a large tensor for performance test
    np.random.seed(0)
    input_tensor_torch = torch.randn(1000, 1000)
    input_tensor_ms = ms.Tensor(input_tensor_torch.numpy())
    fill_value = 3.14

    ms_perf = full_like_forward_perf(input_tensor_ms, fill_value)
    expect_perf = generate_expect_full_like_forward_perf(input_tensor_torch, fill_value)
    assert ms_perf - BACKGROUND_NOISE <= expect_perf * 1.1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_full_like_perf_small_tensor(mode):
    """
    Feature: standard forward performance for full_like with small tensor.
    Description: test full_like op performance.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Create a small tensor for performance test
    input_tensor_torch = torch.randn(10, 10)
    input_tensor_ms = ms.Tensor(input_tensor_torch.numpy())
    fill_value = 2.71

    ms_perf = full_like_forward_perf(input_tensor_ms, fill_value)
    expect_perf = generate_expect_full_like_forward_perf(input_tensor_torch, fill_value)
    assert ms_perf - BACKGROUND_NOISE <= expect_perf * 1.1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_full_like_perf_medium_tensor(mode):
    """
    Feature: standard forward performance for full_like with medium tensor.
    Description: test full_like op performance.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Create a medium tensor for performance test
    input_tensor_torch = torch.randn(100, 200)
    input_tensor_ms = ms.Tensor(input_tensor_torch.numpy())
    fill_value = -1.5

    ms_perf = full_like_forward_perf(input_tensor_ms, fill_value)
    expect_perf = generate_expect_full_like_forward_perf(input_tensor_torch, fill_value)
    assert ms_perf - BACKGROUND_NOISE <= expect_perf * 1.1
