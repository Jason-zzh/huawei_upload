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
""" avg_pool3d op performance test case """
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


def avg_pool3d_forward_perf(input_tensor, kernel_size, stride=None, padding=0, ceil_mode=False, 
                           count_include_pad=True, divisor_override=None):
    """Get MindSpore avg_pool3d forward performance."""
    ms_input = ms.Tensor(input_tensor)
    
    # Warm-up
    for _ in range(100):
        _ = mint.nn.functional.avg_pool3d(
            ms_input, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            ceil_mode=ceil_mode, 
            count_include_pad=count_include_pad, 
            divisor_override=divisor_override
        )

    _pynative_executor.sync()
    start_time = time.time()
    # Performance test
    for _ in range(100):
        _ = mint.nn.functional.avg_pool3d(
            ms_input, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            ceil_mode=ceil_mode, 
            count_include_pad=count_include_pad, 
            divisor_override=divisor_override
        )
    _pynative_executor.sync()
    end_time = time.time()

    return end_time - start_time


def generate_expect_avg_pool3d_forward_perf(input_tensor, kernel_size, stride=None, padding=0, ceil_mode=False, 
                                          count_include_pad=True, divisor_override=None):
    """Get PyTorch avg_pool3d forward performance."""
    torch_input = torch.tensor(input_tensor)
    
    # Warm-up
    for _ in range(100):
        _ = torch.nn.functional.avg_pool3d(
            torch_input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override
        )

    start_time = time.time()
    # Performance test
    for _ in range(100):
        _ = torch.nn.functional.avg_pool3d(
            torch_input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override
        )
    end_time = time.time()

    return end_time - start_time


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_avg_pool3d_perf(mode):
    """
    Feature: standard forward performance for avg_pool3d.
    Description: test avg_pool3d op performance.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate test input
    np.random.seed(0)
    input_tensor = np.random.uniform(-1, 1, (2, 4, 16, 16, 16)).astype(np.float32)
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    padding = (0, 0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None

    ms_perf = avg_pool3d_forward_perf(input_tensor, kernel_size, stride, padding, ceil_mode, 
                                     count_include_pad, divisor_override)
    expect_perf = generate_expect_avg_pool3d_forward_perf(input_tensor, kernel_size, stride, padding, ceil_mode, 
                                                        count_include_pad, divisor_override)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_avg_pool3d_perf_with_padding(mode):
    """
    Feature: performance for avg_pool3d with padding.
    Description: test avg_pool3d op performance with padding.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate test input
    np.random.seed(0)
    input_tensor = np.random.uniform(-1, 1, (1, 3, 12, 12, 12)).astype(np.float32)
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None

    ms_perf = avg_pool3d_forward_perf(input_tensor, kernel_size, stride, padding, ceil_mode, 
                                     count_include_pad, divisor_override)
    expect_perf = generate_expect_avg_pool3d_forward_perf(input_tensor, kernel_size, stride, padding, ceil_mode, 
                                                        count_include_pad, divisor_override)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_avg_pool3d_perf_with_ceil_mode(mode):
    """
    Feature: performance for avg_pool3d with ceil_mode.
    Description: test avg_pool3d op performance with ceil_mode.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate test input
    np.random.seed(0)
    input_tensor = np.random.uniform(-1, 1, (1, 2, 10, 10, 10)).astype(np.float32)
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (0, 0, 0)
    ceil_mode = True
    count_include_pad = True
    divisor_override = None

    ms_perf = avg_pool3d_forward_perf(input_tensor, kernel_size, stride, padding, ceil_mode, 
                                     count_include_pad, divisor_override)
    expect_perf = generate_expect_avg_pool3d_forward_perf(input_tensor, kernel_size, stride, padding, ceil_mode, 
                                                        count_include_pad, divisor_override)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()
