#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" full op performance test case """
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


def full_forward_perf(size, fill_value):
    """Get MindSpore full forward performance."""
    # Warm-up
    for _ in range(1000):
        _ = mint.full(size, fill_value)

    _pynative_executor.sync()
    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = mint.full(size, fill_value)
    _pynative_executor.sync()
    end_time = time.time()

    return end_time - start_time


def generate_expect_full_forward_perf(size, fill_value):
    """Get PyTorch full forward performance."""
    # Warm-up
    for _ in range(1000):
        _ = torch.full(size, fill_value)

    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = torch.full(size, fill_value)
    end_time = time.time()

    return end_time - start_time


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_full_perf(mode):
    """
    Feature: standard forward performance for full.
    Description: test full op performance.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate test parameters for performance test
    size = (1000, 100)  # Large enough to measure performance
    fill_value = 3.14

    ms_perf = full_forward_perf(size, fill_value)
    expect_perf = generate_expect_full_forward_perf(size, fill_value)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
@pytest.mark.parametrize(
    "size, fill_value",
    [
        ((100, 100), 1.0),
        ((500, 50), 42),
        ((10, 10, 100), -2.5),
        ((2000,), 0.0),
    ],
)
def test_full_perf_various_sizes(mode, size, fill_value):
    """
    Feature: performance for various tensor sizes.
    Description: test full op performance with different sizes.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    ms_perf = full_forward_perf(size, fill_value)
    expect_perf = generate_expect_full_forward_perf(size, fill_value)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()
