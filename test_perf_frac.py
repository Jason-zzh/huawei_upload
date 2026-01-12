#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" frac op performance test case """
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


def frac_forward_perf(input_tensor):
    """Get MindSpore frac forward performance."""
    # Warm-up
    for _ in range(1000):
        _ = mint.frac(input_tensor)

    _pynative_executor.sync()
    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = mint.frac(input_tensor)
    _pynative_executor.sync()
    end_time = time.time()

    return end_time - start_time


def generate_expect_frac_forward_perf(input_tensor):
    """Get PyTorch frac forward performance."""
    # Warm-up
    for _ in range(1000):
        _ = torch.frac(input_tensor)

    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = torch.frac(input_tensor)
    end_time = time.time()

    return end_time - start_time


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_frac_perf(mode):
    """
    Feature: standard forward performance for frac.
    Description: test frac op performance.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate random input for performance test
    np.random.seed(0)
    input_tensor = ms.Tensor(np.random.uniform(-10.0, 10.0, size=(100000,)).astype(np.float32))
    torch_tensor = torch.tensor(input_tensor.asnumpy())

    ms_perf = frac_forward_perf(input_tensor)
    expect_perf = generate_expect_frac_forward_perf(torch_tensor)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()
